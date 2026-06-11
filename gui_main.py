"""
Main GUI Module
PyQt5 기반 메인 GUI 및 트레이 아이콘 구현
"""
import sys
import os
import subprocess
import time
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QScrollArea, QGridLayout,
    QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTextEdit, QGroupBox, QCheckBox, QSplitter, QLineEdit,
    QSizePolicy
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint, QRect
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen, QColor, QMouseEvent
import threading

from config_manager import ConfigManager, PatternConfig, ROI, CameraConfig
from logger import Logger
from rtsp_stream import RTSPStream, FrameBuffer, StreamState
from template_matching import TemplateMatcher, TriggerBuffer, TenengradeAnalyzer
from ftp_manager import FTPManager, FileStorageManager
from ai_classifier import AIClassifier, CombinedTrigger
from startup_manager import apply_startup


class ImageLabel(QLabel):
    """이미지 표시용 라벨 - ROI/Template 그리기 지원"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.current_image = None
        self._original_pixmap = QPixmap()
        self._updating_pixmap = False
        self._last_scaled_size = None
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self._update_scaled_pixmap)
        self.roi_rects = []  # ROI 영역 리스트
        self.template_rects = []  # Template 영역 리스트
        self.matched_rects = []  # 매칭된 영역 리스트 (x, y, width, height, score)
        self.drawing = False
        self.start_point = QPoint()
        self.current_point = QPoint()
        self.draw_mode = None  # 'roi', 'template', None
        self.roi_callback = None
        self.template_callback = None
        self.image_clicked_callback = None
        self.setScaledContents(False)  # 자동 스케일링 비활성화
        self.setMinimumSize(240, 180)  # 창 축소/리사이즈 중에도 안전한 최소 크기
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_image(self, image: np.ndarray):
        """이미지 설정"""
        if image is None:
            self.current_image = None
            self._original_pixmap = QPixmap()
            self._last_scaled_size = None
            self.clear()
            return

        if image.size == 0:
            return

        # 리사이즈 중 RTSP 스레드/버퍼가 같은 numpy 메모리를 갱신해도 QImage가
        # dangling pointer를 참조하지 않도록 연속 메모리 + deep copy로 분리한다.
        self.current_image = np.ascontiguousarray(image.copy())
        rgb_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self._original_pixmap = QPixmap.fromImage(qt_image)
        self._last_scaled_size = None
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        """현재 라벨 크기에 맞춰 pixmap을 안전하게 갱신한다."""
        if self._updating_pixmap:
            return
        if self._original_pixmap.isNull():
            return

        label_size = self.size()
        if label_size.width() <= 1 or label_size.height() <= 1:
            return
        size_key = (label_size.width(), label_size.height())
        if self._last_scaled_size == size_key and self.pixmap() is not None:
            return

        self._updating_pixmap = True
        try:
            scaled_pixmap = self._original_pixmap.scaled(
                label_size,
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            if not scaled_pixmap.isNull():
                self.setPixmap(scaled_pixmap)
                self._last_scaled_size = size_key
        finally:
            self._updating_pixmap = False

    def resizeEvent(self, event):
        """리사이즈 이벤트 - 저장된 pixmap만 다시 스케일링"""
        super().resizeEvent(event)
        # resizeEvent 안에서 OpenCV 변환까지 다시 수행하면 창 가장자리 드래그 중
        # 이벤트가 폭주할 때 불안정할 수 있어, 짧게 debounce한 뒤 pixmap만 갱신한다.
        self._resize_timer.start(16)

    def set_roi_rects(self, rects: List[Tuple[int, int, int, int]]):
        """ROI 영역 설정"""
        self.roi_rects = rects
        self.update()

    def set_template_rects(self, rects: List[Tuple[int, int, int, int]]):
        """Template 영역 설정"""
        self.template_rects = rects
        self.update()

    def set_matched_rects(self, rects: List[Tuple[int, int, int, int, float]]):
        """매칭된 영역 설정 (x, y, width, height, score)"""
        self.matched_rects = rects
        self.update()

    def set_draw_mode(self, mode: Optional[str]):
        """그리기 모드 설정 ('roi', 'template', None)"""
        self.draw_mode = mode

    def set_roi_callback(self, callback):
        """ROI 콜백 설정"""
        self.roi_callback = callback

    def set_template_callback(self, callback):
        """Template 콜백 설정"""
        self.template_callback = callback

    def set_image_clicked_callback(self, callback):
        """이미지 클릭 콜백 설정"""
        self.image_clicked_callback = callback

    def _pixmap_geometry(self) -> Optional[QRect]:
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull() or pixmap.width() <= 0 or pixmap.height() <= 0:
            return None
        offset_x = (self.width() - pixmap.width()) // 2
        offset_y = (self.height() - pixmap.height()) // 2
        return QRect(offset_x, offset_y, pixmap.width(), pixmap.height())

    def _screen_to_image_rect(self, start_point: QPoint, end_point: QPoint) -> Optional[Tuple[int, int, int, int]]:
        if self.current_image is None:
            return None
        geometry = self._pixmap_geometry()
        if geometry is None:
            return None

        img_h, img_w = self.current_image.shape[:2]
        if img_w <= 0 or img_h <= 0:
            return None

        img_scale_x = img_w / geometry.width()
        img_scale_y = img_h / geometry.height()
        x1 = int((start_point.x() - geometry.x()) * img_scale_x)
        y1 = int((start_point.y() - geometry.y()) * img_scale_y)
        x2 = int((end_point.x() - geometry.x()) * img_scale_x)
        y2 = int((end_point.y() - geometry.y()) * img_scale_y)

        x = max(0, min(min(x1, x2), img_w - 1))
        y = max(0, min(min(y1, y2), img_h - 1))
        width = min(abs(x2 - x1), img_w - x)
        height = min(abs(y2 - y1), img_h - y)
        if width <= 0 or height <= 0:
            return None
        return x, y, width, height

    def mousePressEvent(self, event: QMouseEvent):
        """마우스 누름 이벤트"""
        if event.button() == Qt.LeftButton:
            if self.draw_mode and self.current_image is not None:
                self.drawing = True
                self.start_point = event.pos()
                self.current_point = event.pos()
            elif self.image_clicked_callback:
                self.image_clicked_callback()

    def mouseMoveEvent(self, event: QMouseEvent):
        """마우스 이동 이벤트"""
        if self.drawing:
            self.current_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent):
        """마우스 떼기 이벤트"""
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            image_rect = self._screen_to_image_rect(self.start_point, event.pos())
            if image_rect:
                if self.draw_mode == 'roi' and self.roi_callback:
                    self.roi_callback(image_rect)
                elif self.draw_mode == 'template' and self.template_callback:
                    self.template_callback(image_rect)
            self.draw_mode = None

    def paintEvent(self, event):
        """페인트 이벤트 - ROI/Template 그리기"""
        super().paintEvent(event)

        if self.current_image is None:
            return
        geometry = self._pixmap_geometry()
        if geometry is None:
            return

        img_h, img_w = self.current_image.shape[:2]
        if img_w <= 0 or img_h <= 0:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 이미지 좌표를 화면 좌표로 변환하는 함수
        def img_to_screen(img_x, img_y, img_w_rect, img_h_rect):
            scale_x = geometry.width() / img_w
            scale_y = geometry.height() / img_h
            screen_x = int(geometry.x() + img_x * scale_x)
            screen_y = int(geometry.y() + img_y * scale_y)
            screen_w = int(img_w_rect * scale_x)
            screen_h = int(img_h_rect * scale_y)
            return screen_x, screen_y, screen_w, screen_h

        # ROI 영역 그리기 (녹색)
        pen = QPen(QColor(0, 255, 0), 2)
        painter.setPen(pen)
        for roi in self.roi_rects:
            x, y, w, h = img_to_screen(*roi)
            painter.drawRect(x, y, w, h)

        # Template 영역 그리기 (빨간색)
        pen = QPen(QColor(255, 0, 0), 2)
        painter.setPen(pen)
        for template in self.template_rects:
            x, y, w, h = img_to_screen(*template)
            painter.drawRect(x, y, w, h)

        # 매칭된 영역 그리기 (노란색 + 점선)
        pen = QPen(QColor(255, 255, 0), 3)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        for matched in self.matched_rects:
            x, y, w, h, score = matched
            screen_x, screen_y, screen_w, screen_h = img_to_screen(x, y, w, h)
            painter.drawRect(screen_x, screen_y, screen_w, screen_h)
            painter.setPen(QColor(255, 255, 0))
            painter.drawText(screen_x, screen_y - 5, f"{score:.2f}")
            pen = QPen(QColor(255, 255, 0), 3)
            pen.setStyle(Qt.DashLine)
            painter.setPen(pen)

        # 현재 그리기 중인 영역 표시
        if self.drawing and self.draw_mode:
            if self.draw_mode == 'roi':
                pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
            else:
                pen = QPen(QColor(255, 0, 255), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawRect(
                self.start_point.x(),
                self.start_point.y(),
                self.current_point.x() - self.start_point.x(),
                self.current_point.y() - self.start_point.y()
            )


class ThumbnailLabel(QLabel):
    """썸네일 이미지 라벨 - 클릭 가능"""

    clicked = pyqtSignal()

    def __init__(self, image: np.ndarray, index: int, parent=None):
        super().__init__(parent)
        self.image = image
        self.index = index
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

        # 썸네일 생성
        self.set_thumbnail()

    def set_thumbnail(self):
        """썸네일 설정"""
        if self.image is not None:
            # 리사이즈
            h, w = self.image.shape[:2]
            max_size = 150
            scale = min(max_size / w, max_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(self.image, (new_w, new_h))

            # QPixmap으로 변환
            rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.setPixmap(pixmap)

    def mousePressEvent(self, event: QMouseEvent):
        """마우스 클릭 이벤트"""
        if event.button() == Qt.LeftButton:
            self.clicked.emit()


class CameraWidget(QWidget):
    """카메라 1대의 라이브/설정/로그/검출 런타임을 독립 보유하는 위젯"""

    def __init__(self, camera_index: int, config_manager: ConfigManager, logger: Logger, app_restart_callback=None, parent=None):
        super().__init__(parent)
        self.camera_index = camera_index
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        self.camera = self.config.cameras[camera_index]
        self.logger = logger
        self.app_restart_callback = app_restart_callback

        self.rtsp_stream = RTSPStream(self.camera.rtsp_url, self.camera.rtsp_reconnect_interval)
        self.template_matcher = TemplateMatcher()
        self.trigger_buffer = TriggerBuffer(self.camera.buffer_size)
        self.frame_buffer = FrameBuffer(self.camera.max_frames)
        self.ftp_manager = FTPManager(
            host=self.camera.ftp_host,
            port=self.camera.ftp_port,
            username=self.camera.ftp_username,
            password=self.camera.ftp_password,
            remote_dir=self.camera.ftp_remote_dir,
            timeout=self.camera.ftp_timeout,
            enabled=self.camera.ftp_enabled
        )
        self.file_storage = self.create_file_storage()
        self.ai_classifier = AIClassifier(self.camera.ai_model_path, self.camera.ai_threshold)
        self.combined_trigger = CombinedTrigger(self.camera.ai_condition_type)

        self.is_running = False
        self.is_gathering = False
        self.gathered_images = []
        self.selected_image = None
        self.current_roi = None
        self.current_template = None
        self.show_roi_regions = True
        self.background_frame_cnt = 0
        self.image_save_done = False
        self.last_health_frame = None
        self.last_health_change_time = time.time()
        self.consecutive_health_failures = 0
        self.stream_restart_history = []

        self.init_ui()
        self.setup_callbacks()
        self.load_templates()

        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(30)

        self.health_timer = QTimer(self)
        self.health_timer.timeout.connect(self.check_health)
        self.health_timer.start(max(1, self.camera.healthcheck.check_interval_sec) * 1000)

        self.cleanup_timer = QTimer(self)
        self.cleanup_timer.timeout.connect(self.cleanup_old_files)
        self.cleanup_timer.start(3600000)
        self.cleanup_old_files()

    def create_file_storage(self) -> FileStorageManager:
        opts = self.camera.image_save
        return FileStorageManager(
            save_dir=self.camera.save_dir,
            retention_days=self.camera.retention_days,
            equipment_no=opts.equipment_no,
            mount=opts.mount,
            image_format=opts.image_format,
            quality=opts.quality,
            resize_enabled=opts.resize_enabled,
            resize_width=opts.resize_width,
            resize_height=opts.resize_height,
            keep_aspect_ratio=opts.keep_aspect_ratio,
            filename_format=opts.filename_format,
        )

    def log_info(self, message: str):
        text = f"[{self.camera.name}] {message}"
        self.logger.info(text)
        if hasattr(self, 'log_text'):
            self.log_text.append(f"{datetime.now().strftime('%H:%M:%S')} {text}")

    def log_error(self, message: str):
        text = f"[{self.camera.name}] {message}"
        self.logger.error(text)
        if hasattr(self, 'log_text'):
            self.log_text.append(f"{datetime.now().strftime('%H:%M:%S')} ERROR {text}")

    def init_ui(self):
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Horizontal)
        layout.addWidget(splitter)
        splitter.addWidget(self.create_left_panel())
        splitter.addWidget(self.create_right_panel())
        splitter.setSizes([340, 900])

    def create_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.inner_tab_widget = QTabWidget()
        layout.addWidget(self.inner_tab_widget)
        self.inner_tab_widget.addTab(self.create_live_tab(), "라이브")
        self.inner_tab_widget.addTab(self.create_config_tab(), "설정")
        self.inner_tab_widget.addTab(self.create_log_tab(), "로그")
        return panel

