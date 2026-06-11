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

    def create_live_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)

        control_group = QGroupBox("제어")
        control_layout = QVBoxLayout()
        self.start_stop_btn = QPushButton("START")
        self.start_stop_btn.setCheckable(True)
        self.start_stop_btn.clicked.connect(self.toggle_start_stop)
        self.start_stop_btn.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }
            QPushButton:checked { background-color: #f44336; }
        """)
        control_layout.addWidget(self.start_stop_btn)

        self.gather_btn = QPushButton("이미지 수집")
        self.gather_btn.clicked.connect(self.start_image_gathering)
        control_layout.addWidget(self.gather_btn)

        self.template_reg_btn = QPushButton("템플릿 등록")
        self.template_reg_btn.clicked.connect(self.show_template_registration)
        control_layout.addWidget(self.template_reg_btn)
        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        status_group = QGroupBox("상태")
        status_layout = QVBoxLayout()
        self.status_label = QLabel("상태: 대기중")
        self.fps_label = QLabel("FPS: 0.0")
        self.buffer_label = QLabel("버퍼: 0/0")
        self.health_label = QLabel("Health: 대기중")
        status_layout.addWidget(self.status_label)
        status_layout.addWidget(self.fps_label)
        status_layout.addWidget(self.buffer_label)
        status_layout.addWidget(self.health_label)
        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        thumbnail_group = QGroupBox("수집된 이미지")
        thumbnail_layout = QVBoxLayout()
        self.thumbnail_scroll = QScrollArea()
        self.thumbnail_scroll.setWidgetResizable(True)
        self.thumbnail_widget = QWidget()
        self.thumbnail_layout = QGridLayout(self.thumbnail_widget)
        self.thumbnail_scroll.setWidget(self.thumbnail_widget)
        thumbnail_layout.addWidget(self.thumbnail_scroll)
        thumbnail_group.setLayout(thumbnail_layout)
        layout.addWidget(thumbnail_group)
        return tab

    def create_config_tab(self) -> QWidget:
        tab = QWidget()
        outer_layout = QVBoxLayout(tab)
        outer_layout.setContentsMargins(4, 4, 4, 4)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setFrameShape(QFrame.NoFrame)

        content = QWidget()
        content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        layout = QVBoxLayout(content)

        cam_group = QGroupBox("카메라 설정")
        cam_layout = QVBoxLayout()
        cam_layout.addWidget(QLabel("카메라 이름:"))
        self.camera_name_edit = QLineEdit(self.camera.name)
        cam_layout.addWidget(self.camera_name_edit)
        self.camera_enabled_check = QCheckBox("카메라 사용")
        self.camera_enabled_check.setChecked(self.camera.enabled)
        cam_layout.addWidget(self.camera_enabled_check)
        cam_group.setLayout(cam_layout)
        layout.addWidget(cam_group)

        rtsp_group = QGroupBox("RTSP 설정")
        rtsp_layout = QVBoxLayout()
        rtsp_layout.addWidget(QLabel("RTSP URL:"))
        self.rtsp_url_edit = QTextEdit()
        self.rtsp_url_edit.setMaximumHeight(60)
        self.rtsp_url_edit.setPlainText(self.camera.rtsp_url)
        rtsp_layout.addWidget(self.rtsp_url_edit)
        rtsp_layout.addWidget(QLabel("재연결 간격(초):"))
        self.reconnect_spin = QSpinBox()
        self.reconnect_spin.setRange(1, 300)
        self.reconnect_spin.setValue(self.camera.rtsp_reconnect_interval)
        rtsp_layout.addWidget(self.reconnect_spin)
        rtsp_group.setLayout(rtsp_layout)
        layout.addWidget(rtsp_group)

        pattern_group = QGroupBox("패턴 설정")
        pattern_layout = QVBoxLayout()
        pattern_layout.addWidget(QLabel("패턴 인덱스:"))
        self.pattern_index_spin = QSpinBox()
        self.pattern_index_spin.setMinimum(1)
        self.pattern_index_spin.setMaximum(10)
        self.pattern_index_spin.setValue(1)
        pattern_layout.addWidget(self.pattern_index_spin)
        pattern_layout.addWidget(QLabel("매칭 임계값:"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setMinimum(0.0)
        self.threshold_spin.setMaximum(1.0)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.85)
        pattern_layout.addWidget(self.threshold_spin)
        self.require_all_check = QCheckBox("모든 패턴 매칭 필요")
        self.require_all_check.setChecked(self.camera.require_all_patterns)
        pattern_layout.addWidget(self.require_all_check)
        pattern_group.setLayout(pattern_layout)
        layout.addWidget(pattern_group)

        buffer_group = QGroupBox("버퍼/수집 설정")
        buffer_layout = QVBoxLayout()
        buffer_layout.addWidget(QLabel("트리거 버퍼 크기:"))
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setMinimum(1)
        self.buffer_size_spin.setMaximum(100)
        self.buffer_size_spin.setValue(self.camera.buffer_size)
        buffer_layout.addWidget(self.buffer_size_spin)
        buffer_layout.addWidget(QLabel("수집 시간 (초):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setMinimum(1)
        self.duration_spin.setMaximum(60)
        self.duration_spin.setValue(self.camera.frame_save_duration)
        buffer_layout.addWidget(self.duration_spin)
        buffer_group.setLayout(buffer_layout)
        layout.addWidget(buffer_group)

        file_group = QGroupBox("이미지 저장 설정")
        file_layout = QVBoxLayout()
        file_layout.addWidget(QLabel("저장 폴더:"))
        self.save_dir_edit = QLineEdit(self.camera.save_dir)
        file_layout.addWidget(self.save_dir_edit)
        file_layout.addWidget(QLabel("보관 기간(일):"))
        self.retention_spin = QSpinBox()
        self.retention_spin.setRange(1, 3650)
        self.retention_spin.setValue(self.camera.retention_days)
        file_layout.addWidget(self.retention_spin)
        file_layout.addWidget(QLabel("설비 No:"))
        self.equipment_edit = QLineEdit(self.camera.image_save.equipment_no)
        file_layout.addWidget(self.equipment_edit)
        file_layout.addWidget(QLabel("Mount:"))
        self.mount_edit = QLineEdit(self.camera.image_save.mount)
        file_layout.addWidget(self.mount_edit)
        file_layout.addWidget(QLabel("저장 포맷:"))
        self.image_format_combo = QComboBox()
        self.image_format_combo.addItems(["BMP", "JPG", "PNG"])
        self.image_format_combo.setCurrentText(self.camera.image_save.image_format if self.camera.image_save.image_format in ["BMP", "JPG", "PNG"] else "BMP")
        file_layout.addWidget(self.image_format_combo)
        file_layout.addWidget(QLabel("품질/압축률(10~90):"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(10, 90)
        self.quality_spin.setValue(self.camera.image_save.quality)
        file_layout.addWidget(self.quality_spin)
        self.resize_check = QCheckBox("저장 이미지 크기 변경")
        self.resize_check.setChecked(self.camera.image_save.resize_enabled)
        file_layout.addWidget(self.resize_check)
        file_layout.addWidget(QLabel("저장 Width(0=원본/자동):"))
        self.resize_width_spin = QSpinBox()
        self.resize_width_spin.setRange(0, 10000)
        self.resize_width_spin.setValue(self.camera.image_save.resize_width)
        file_layout.addWidget(self.resize_width_spin)
        file_layout.addWidget(QLabel("저장 Height(0=원본/자동):"))
        self.resize_height_spin = QSpinBox()
        self.resize_height_spin.setRange(0, 10000)
        self.resize_height_spin.setValue(self.camera.image_save.resize_height)
        file_layout.addWidget(self.resize_height_spin)
        self.keep_aspect_check = QCheckBox("비율 유지")
        self.keep_aspect_check.setChecked(self.camera.image_save.keep_aspect_ratio)
        file_layout.addWidget(self.keep_aspect_check)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        health_group = QGroupBox("HealthCheck 설정")
        health_layout = QVBoxLayout()
        self.health_enabled_check = QCheckBox("HealthCheck 사용")
        self.health_enabled_check.setChecked(self.camera.healthcheck.enabled)
        health_layout.addWidget(self.health_enabled_check)
        health_layout.addWidget(QLabel("영상 정지 판단 시간(초):"))
        self.health_timeout_spin = QSpinBox()
        self.health_timeout_spin.setRange(1, 3600)
        self.health_timeout_spin.setValue(self.camera.healthcheck.timeout_sec)
        health_layout.addWidget(self.health_timeout_spin)
        self.restart_stream_check = QCheckBox("이상 시 RTSP 재연결")
        self.restart_stream_check.setChecked(self.camera.healthcheck.restart_stream)
        health_layout.addWidget(self.restart_stream_check)
        self.restart_app_check = QCheckBox("재연결 반복 실패 시 프로그램 재시작")
        self.restart_app_check.setChecked(self.camera.healthcheck.restart_app)
        health_layout.addWidget(self.restart_app_check)
        self.restart_limit_check = QCheckBox("무한 재시작 방지 사용")
        self.restart_limit_check.setChecked(self.camera.healthcheck.restart_limit_enabled)
        health_layout.addWidget(self.restart_limit_check)
        health_layout.addWidget(QLabel("1시간 최대 프로그램 재시작 횟수:"))
        self.max_restart_spin = QSpinBox()
        self.max_restart_spin.setRange(1, 100)
        self.max_restart_spin.setValue(self.camera.healthcheck.max_restart_per_hour)
        health_layout.addWidget(self.max_restart_spin)
        health_group.setLayout(health_layout)
        layout.addWidget(health_group)

        # Cam1에만 전역 자동실행 설정 노출
        if self.camera_index == 0:
            system_group = QGroupBox("시스템 공통 설정")
            system_layout = QVBoxLayout()
            self.auto_start_check = QCheckBox("Windows 로그인 후 자동 실행")
            self.auto_start_check.setChecked(self.config.auto_start_enabled)
            system_layout.addWidget(self.auto_start_check)
            self.auto_run_check = QCheckBox("프로그램 시작 후 자동 START")
            self.auto_run_check.setChecked(self.config.auto_run_detection)
            system_layout.addWidget(self.auto_run_check)
            system_layout.addWidget(QLabel("자동 실행 방식:"))
            self.auto_start_method_combo = QComboBox()
            self.auto_start_method_combo.addItems(["task_scheduler", "registry"])
            self.auto_start_method_combo.setCurrentText(self.config.auto_start_method)
            system_layout.addWidget(self.auto_start_method_combo)
            system_group.setLayout(system_layout)
            layout.addWidget(system_group)

        layout.addStretch()
        scroll_area.setWidget(content)
        outer_layout.addWidget(scroll_area)

        save_btn = QPushButton("설정 저장")
        save_btn.clicked.connect(self.save_config)
        save_btn.setMinimumHeight(36)
        outer_layout.addWidget(save_btn)
        return tab

    def create_log_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        return tab

    def create_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        self.main_image_label = ImageLabel()
        self.main_image_label.setAlignment(Qt.AlignCenter)
        self.main_image_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        layout.addWidget(self.main_image_label)

        index_layout = QHBoxLayout()
        index_layout.addWidget(QLabel("템플릿 인덱스:"))
        self.template_index_spin = QSpinBox()
        self.template_index_spin.setMinimum(1)
        self.template_index_spin.setMaximum(10)
        self.template_index_spin.setValue(1)
        self.template_index_spin.setFixedWidth(80)
        index_layout.addWidget(self.template_index_spin)
        index_layout.addStretch()
        layout.addLayout(index_layout)

        control_layout = QHBoxLayout()
        self.roi_btn = QPushButton("ROI 설정")
        self.roi_btn.setCheckable(True)
        self.roi_btn.clicked.connect(self.toggle_roi_mode)
        control_layout.addWidget(self.roi_btn)
        self.template_btn = QPushButton("Template 설정")
        self.template_btn.setCheckable(True)
        self.template_btn.clicked.connect(self.toggle_template_mode)
        control_layout.addWidget(self.template_btn)
        self.clear_btn = QPushButton("영역 지우기")
        self.clear_btn.clicked.connect(self.clear_regions)
        control_layout.addWidget(self.clear_btn)
        layout.addLayout(control_layout)
        self.pattern_info_label = QLabel("현재 패턴: 없음")
        layout.addWidget(self.pattern_info_label)
        return panel

    def setup_callbacks(self):
        self.rtsp_stream.set_error_callback(self.on_rtsp_error)
        self.rtsp_stream.set_state_callback(self.on_rtsp_state_change)
        self.main_image_label.set_roi_callback(self.on_roi_selected)
        self.main_image_label.set_template_callback(self.on_template_selected)
        self.ftp_manager.set_upload_callback(self.on_ftp_upload)

    def load_templates(self):
        self.template_matcher.clear_all()
        for pattern in self.camera.patterns:
            if pattern.template_path and Path(pattern.template_path).exists():
                self.template_matcher.load_template(
                    index=pattern.index,
                    template_path=pattern.template_path,
                    roi=(pattern.roi.x, pattern.roi.y, pattern.roi.width, pattern.roi.height),
                    threshold=pattern.score_threshold
                )
                self.log_info(f"템플릿 로드됨: Index={pattern.index}, Path={pattern.template_path}")

    def toggle_start_stop(self):
        if self.start_stop_btn.isChecked():
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        if not self.camera.enabled:
            self.start_stop_btn.setChecked(False)
            QMessageBox.warning(self, "경고", f"{self.camera.name} 카메라가 비활성화되어 있습니다.")
            return
        self.is_running = True
        self.start_stop_btn.setText("STOP")
        self.rtsp_stream.rtsp_url = self.camera.rtsp_url
        self.rtsp_stream.reconnect_interval = self.camera.rtsp_reconnect_interval
        self.rtsp_stream.start()
        self.log_info("검출 시작")
        self.status_label.setText("상태: 실행중")
        self.last_health_change_time = time.time()

    def stop_detection(self):
        self.is_running = False
        self.start_stop_btn.setChecked(False)
        self.start_stop_btn.setText("START")
        self.rtsp_stream.stop()
        self.log_info("검출 중지")
        self.status_label.setText("상태: 대기중")
        self.health_label.setText("Health: 대기중")

    def start_image_gathering(self):
        if self.is_gathering:
            return
        self.is_gathering = True
        self.gather_btn.setEnabled(False)
        self.gather_btn.setText("수집중...")
        thread = threading.Thread(target=self._gather_images, daemon=True)
        thread.start()

    def _gather_images(self):
        self.frame_buffer.clear()
        self.gathered_images = []
        duration = self.camera.frame_save_duration
        start_time = datetime.now()
        save_dir = Path("gathered_images") / self.camera.name / start_time.strftime("%Y%m%d_%H%M%S")
        save_dir.mkdir(parents=True, exist_ok=True)
        self.log_info(f"이미지 수집 시작: Duration={duration}s, 저장 경로: {save_dir}")
        frame_count = 0
        while self.is_gathering:
            frame_info = self.rtsp_stream.get_frame()
            if frame_info:
                self.frame_buffer.add_frame(frame_info.frame, frame_info.timestamp)
                frame_time = datetime.fromtimestamp(frame_info.timestamp)
                filename = frame_time.strftime("%Y%m%d_%H%M%S") + f"_{frame_count:03d}.bmp"
                cv2.imwrite(str(save_dir / filename), frame_info.frame)
                frame_count += 1
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration or self.frame_buffer.is_full():
                break
        self.gathered_images = self.frame_buffer.get_frames()
        self.is_gathering = False
        self.gather_btn.setEnabled(True)
        self.gather_btn.setText("이미지 수집")
        self.logger.log_image_gathering(duration, len(self.gathered_images))
        self.log_info(f"이미지 저장 완료: {save_dir}, 총 {frame_count}개 파일")
        self.show_thumbnails()

    def show_thumbnails(self):
        for i in reversed(range(self.thumbnail_layout.count())):
            widget = self.thumbnail_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)
        cols = 2
        for i, img_data in enumerate(self.gathered_images):
            thumbnail = ThumbnailLabel(img_data['frame'], i)
            thumbnail.clicked.connect(lambda idx=i: self.select_image(idx))
            self.thumbnail_layout.addWidget(thumbnail, i // cols, i % cols)

    def select_image(self, index: int):
        if 0 <= index < len(self.gathered_images):
            self.selected_image = self.gathered_images[index]['frame']
            self.main_image_label.set_image(self.selected_image)
            self.main_image_label.set_roi_rects([])
            self.main_image_label.set_template_rects([])
            self.log_info(f"이미지 선택됨: Index={index}")

    def show_template_registration(self):
        if not self.gathered_images:
            QMessageBox.warning(self, "경고", "먼저 이미지를 수집해주세요.")
            return
        self.show_thumbnails()
        self.log_info("템플릿 등록 모드")

    def toggle_roi_mode(self):
        if self.roi_btn.isChecked():
            self.template_btn.setChecked(False)
            self.main_image_label.set_draw_mode('roi')
            self.log_info("ROI 설정 모드 활성화")
        else:
            self.main_image_label.set_draw_mode(None)

    def toggle_template_mode(self):
        if self.template_btn.isChecked():
            self.roi_btn.setChecked(False)
            self.main_image_label.set_draw_mode('template')
            self.log_info("Template 설정 모드 활성화")
        else:
            self.main_image_label.set_draw_mode(None)

    def clear_regions(self):
        self.main_image_label.set_roi_rects([])
        self.main_image_label.set_template_rects([])
        self.current_roi = None
        self.current_template = None
        self.show_roi_regions = False
        self.roi_btn.setChecked(False)
        self.template_btn.setChecked(False)
        self.main_image_label.set_draw_mode(None)
        self.log_info("영역 지우기 실행됨")

    def on_roi_selected(self, roi: Tuple[int, int, int, int]):
        self.current_roi = roi
        self.main_image_label.set_roi_rects([roi])
        self.roi_btn.setChecked(False)
        self.main_image_label.set_draw_mode(None)
        self.log_info(f"ROI 설정됨: {roi}")

    def on_template_selected(self, template: Tuple[int, int, int, int]):
        if self.selected_image is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 선택해주세요.")
            return
        self.current_template = template
        self.main_image_label.set_template_rects([template])
        self.template_btn.setChecked(False)
        self.main_image_label.set_draw_mode(None)
        self.save_template()
        self.log_info(f"Template 설정됨: {template}")
        self.selected_image = None
        self.main_image_label.set_roi_rects([])
        self.main_image_label.set_template_rects([])

    def save_template(self):
        if self.current_template is None or self.selected_image is None:
            return
        index = self.template_index_spin.value()
        threshold = self.threshold_spin.value()
        if self.current_roi:
            roi = self.current_roi
        else:
            h, w = self.selected_image.shape[:2]
            roi = (0, 0, w, h)
        x, y, w, h = self.current_template
        template_img = self.selected_image[y:y+h, x:x+w]
        template_dir = Path("templates") / self.camera.name
        template_dir.mkdir(parents=True, exist_ok=True)
        template_path = template_dir / f"template_{index}.png"
        cv2.imwrite(str(template_path), template_img)
        pattern_config = PatternConfig(
            index=index,
            template_path=str(template_path),
            score_threshold=threshold,
            roi=ROI(x=roi[0], y=roi[1], width=roi[2], height=roi[3])
        )
        self.config_manager.update_pattern(index, pattern_config, self.camera_index)
        self.config = self.config_manager.get_config()
        self.camera = self.config.cameras[self.camera_index]
        self.template_matcher.load_template_from_image(index, template_img, roi, threshold)
        self.logger.log_template_registration(index, str(template_path), roi)
        self.pattern_info_label.setText(f"현재 패턴: Index={index}, Threshold={threshold}")
        QMessageBox.information(self, "성공", f"템플릿이 저장되었습니다.\nIndex: {index}")

    def save_config(self):
        self.camera.name = self.camera_name_edit.text().strip() or f"Cam{self.camera_index + 1}"
        self.camera.enabled = self.camera_enabled_check.isChecked()
        self.camera.rtsp_url = self.rtsp_url_edit.toPlainText().strip()
        self.camera.rtsp_reconnect_interval = self.reconnect_spin.value()
        self.camera.require_all_patterns = self.require_all_check.isChecked()
        self.camera.buffer_size = self.buffer_size_spin.value()
        self.camera.frame_save_duration = self.duration_spin.value()
        self.camera.save_dir = self.save_dir_edit.text().strip() or f"saved_images/{self.camera.name}"
        self.camera.retention_days = self.retention_spin.value()
        self.camera.image_save.equipment_no = self.equipment_edit.text().strip() or "EQ01"
        self.camera.image_save.mount = self.mount_edit.text().strip() or "Mount"
        self.camera.image_save.image_format = self.image_format_combo.currentText()
        self.camera.image_save.quality = self.quality_spin.value()
        self.camera.image_save.resize_enabled = self.resize_check.isChecked()
        self.camera.image_save.resize_width = self.resize_width_spin.value()
        self.camera.image_save.resize_height = self.resize_height_spin.value()
        self.camera.image_save.keep_aspect_ratio = self.keep_aspect_check.isChecked()
        self.camera.healthcheck.enabled = self.health_enabled_check.isChecked()
        self.camera.healthcheck.timeout_sec = self.health_timeout_spin.value()
        self.camera.healthcheck.restart_stream = self.restart_stream_check.isChecked()
        self.camera.healthcheck.restart_app = self.restart_app_check.isChecked()
        self.camera.healthcheck.restart_limit_enabled = self.restart_limit_check.isChecked()
        self.camera.healthcheck.max_restart_per_hour = self.max_restart_spin.value()

        if self.camera_index == 0:
            self.config.auto_start_enabled = self.auto_start_check.isChecked()
            self.config.auto_run_detection = self.auto_run_check.isChecked()
            self.config.auto_start_method = self.auto_start_method_combo.currentText()
            ok, msg = apply_startup(self.config.auto_start_enabled, self.config.auto_start_method)
            if ok:
                self.log_info(msg)
            else:
                self.logger.warning(msg)
                self.log_info(f"자동 실행 설정 확인 필요: {msg}")

        self.config.cameras[self.camera_index] = self.camera
        main_window = self.window()
        if hasattr(main_window, 'camera_tabs'):
            main_window.camera_tabs.setTabText(self.camera_index, self.camera.name)
        self.config_manager.save_config()
        self.config = self.config_manager.get_config()
        self.camera = self.config.cameras[self.camera_index]
        self.trigger_buffer = TriggerBuffer(self.camera.buffer_size)
        self.frame_buffer = FrameBuffer(self.camera.max_frames)
        self.file_storage = self.create_file_storage()
        self.rtsp_stream.rtsp_url = self.camera.rtsp_url
        self.rtsp_stream.reconnect_interval = self.camera.rtsp_reconnect_interval
        self.health_timer.setInterval(max(1, self.camera.healthcheck.check_interval_sec) * 1000)
        self.log_info("설정 저장됨")
        QMessageBox.information(self, "성공", "설정이 저장되었습니다.")

