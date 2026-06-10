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
    QTabWidget, QTextEdit, QGroupBox, QCheckBox, QSplitter, QLineEdit
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint
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
        self.setMinimumSize(400, 300)  # 최소 크기 설정

    def set_image(self, image: np.ndarray):
        """이미지 설정"""
        self.current_image = image
        if image is not None:
            # OpenCV 이미지를 QPixmap으로 변환
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            # 현재 라벨 크기 사용
            label_size = self.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = pixmap.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.setPixmap(scaled_pixmap)
        else:
            self.clear()

    def resizeEvent(self, event):
        """리사이즈 이벤트 - 이미지 다시 스케일링"""
        super().resizeEvent(event)
        if self.current_image is not None:
            self.set_image(self.current_image)

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
            end_point = event.pos()

            # 이미지 좌표로 변환
            if self.pixmap() and self.current_image is not None:
                pixmap = self.pixmap()
                img_scale_x = self.current_image.shape[1] / pixmap.width()
                img_scale_y = self.current_image.shape[0] / pixmap.height()

                # 라벨 내에서의 오프셋 계산
                label_width = self.width()
                label_height = self.height()
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                offset_x = (label_width - pixmap_width) // 2
                offset_y = (label_height - pixmap_height) // 2

                # 실제 이미지 좌표 계산
                x1 = int((self.start_point.x() - offset_x) * img_scale_x)
                y1 = int((self.start_point.y() - offset_y) * img_scale_y)
                x2 = int((end_point.x() - offset_x) * img_scale_x)
                y2 = int((end_point.y() - offset_y) * img_scale_y)

                # 정규화 (x1 < x2, y1 < y2)
                x = min(x1, x2)
                y = min(y1, y2)
                width = abs(x2 - x1)
                height = abs(y2 - y1)

                # 이미지 범위 체크
                x = max(0, min(x, self.current_image.shape[1] - 1))
                y = max(0, min(y, self.current_image.shape[0] - 1))
                width = min(width, self.current_image.shape[1] - x)
                height = min(height, self.current_image.shape[0] - y)
                print(str(self.draw_mode))

                if width > 0 and height > 0:
                    if self.draw_mode == 'roi' and self.roi_callback:
                        self.roi_callback((x, y, width, height))
                    elif self.draw_mode == 'template' and self.template_callback:
                        self.template_callback((x, y, width, height))

            self.draw_mode = None

    def paintEvent(self, event):
        """페인트 이벤트 - ROI/Template 그리기"""
        super().paintEvent(event)

        if not self.pixmap() or self.current_image is None:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # 라벨 내에서의 오프셋 계산
        label_width = self.width()
        label_height = self.height()
        pixmap_width = self.pixmap().width()
        pixmap_height = self.pixmap().height()
        offset_x = (label_width - pixmap_width) // 2
        offset_y = (label_height - pixmap_height) // 2

        # 이미지 좌표를 화면 좌표로 변환하는 함수
        def img_to_screen(img_x, img_y, img_w, img_h):
            scale_x = pixmap_width / self.current_image.shape[1]
            scale_y = pixmap_height / self.current_image.shape[0]
            screen_x = int(offset_x + img_x * scale_x)
            screen_y = int(offset_y + img_y * scale_y)
            screen_w = int(img_w * scale_x)
            screen_h = int(img_h * scale_y)
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
            # 매칭 점수 표시
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
        layout = QVBoxLayout(tab)

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

        save_btn = QPushButton("설정 저장")
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)
        layout.addStretch()
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

    def update_display(self):
        fps = self.rtsp_stream.get_fps()
        self.fps_label.setText(f"FPS: {fps:.1f}")
        self.buffer_label.setText(f"버퍼: {self.trigger_buffer.size()}/{self.camera.buffer_size}")
        if self.is_running and self.selected_image is None:
            frame_info = self.rtsp_stream.get_latest_frame()
            if frame_info:
                frame = frame_info.frame
                matched_rects = []
                if self.template_matcher.get_template_count() > 0:
                    matched, results = self.template_matcher.match_all(frame, require_all=self.camera.require_all_patterns)
                    for result in results:
                        if result.matched:
                            template_size = self.template_matcher.get_template_size(result.pattern_index)
                            if template_size:
                                template_w, template_h = template_size
                                matched_rects.append((result.location[0], result.location[1], template_w, template_h, result.score))
                    if matched:
                        if not self.image_save_done:
                            if not self.trigger_buffer.is_full():
                                self.trigger_buffer.add_frame(frame)
                            else:
                                self.process_buffer()
                                self.image_save_done = True
                        self.background_frame_cnt = 0
                    else:
                        self.background_frame_cnt += 1
                        if self.background_frame_cnt > 5 and self.image_save_done:
                            self.image_save_done = False
                            self.trigger_buffer.clear()
                roi_rects = []
                template_rects = []
                if self.show_roi_regions:
                    for pattern in self.camera.patterns:
                        roi_rects.append((pattern.roi.x, pattern.roi.y, pattern.roi.width, pattern.roi.height))
                self.main_image_label.set_roi_rects(roi_rects)
                self.main_image_label.set_template_rects(template_rects)
                self.main_image_label.set_matched_rects(matched_rects)
                self.main_image_label.set_image(frame)

    def process_buffer(self):
        best_image = self.trigger_buffer.get_best_frame()
        if best_image is not None:
            tenengrade_score = TenengradeAnalyzer.calculate(best_image)
            filepath = self.file_storage.save_image(best_image)
            if filepath:
                self.logger.log_image_saved(filepath, tenengrade_score)
                self.log_info(f"이미지 저장: {filepath}")
                if self.ftp_manager.is_enabled():
                    date_str = datetime.now().strftime("%Y%m%d")
                    remote_subdir = f"{self.camera.name}/{date_str}"
                    self.ftp_manager.upload_file_async(filepath, remote_subdir)

    def _frame_changed(self, frame) -> bool:
        if frame is None:
            return False
        try:
            small = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
            if self.last_health_frame is None:
                self.last_health_frame = gray
                self.last_health_change_time = time.time()
                return True
            diff = float(np.mean(cv2.absdiff(gray, self.last_health_frame)))
            self.last_health_frame = gray
            if diff > self.camera.healthcheck.freeze_diff_threshold:
                self.last_health_change_time = time.time()
                return True
            return False
        except Exception as exc:
            self.log_error(f"HealthCheck 프레임 비교 오류: {exc}")
            return True

    def check_health(self):
        hc = self.camera.healthcheck
        if not hc.enabled or not self.is_running:
            return
        now = time.time()
        last_ts = self.rtsp_stream.get_last_frame_timestamp()
        no_frame_timeout = (not last_ts) or (now - last_ts > hc.timeout_sec)
        current_frame = self.main_image_label.current_image
        self._frame_changed(current_frame)
        freeze_timeout = now - self.last_health_change_time > hc.timeout_sec

        if not no_frame_timeout and not freeze_timeout:
            self.consecutive_health_failures = 0
            self.health_label.setText("Health: 정상")
            return

        reason = "프레임 수신 없음" if no_frame_timeout else "영상 변화 없음"
        self.health_label.setText(f"Health: 이상 감지 - {reason}")
        self.log_error(f"HealthCheck 이상 감지: {reason}")
        self.consecutive_health_failures += 1

        if hc.restart_stream:
            self.restart_stream()
            self.last_health_change_time = now

        if hc.restart_app and self.consecutive_health_failures >= 3 and self.app_restart_callback:
            self.log_error("재연결 반복 실패: 프로그램 재시작 요청")
            self.app_restart_callback(self.camera, hc)

    def restart_stream(self):
        now = time.time()
        self.stream_restart_history = [t for t in self.stream_restart_history if now - t < 3600]
        self.stream_restart_history.append(now)
        self.health_label.setText("Health: RTSP 재연결 중")
        self.log_info("RTSP 재연결 시도")
        self.rtsp_stream.restart()

    def cleanup_old_files(self):
        self.file_storage.cleanup_old_files()

    def on_rtsp_error(self, error_msg: str):
        self.log_error(f"RTSP 오류: {error_msg}")
        self.status_label.setText(f"상태: 오류 - {error_msg}")

    def on_rtsp_state_change(self, state: StreamState):
        self.logger.log_rtsp_status(state == StreamState.CONNECTED, self.rtsp_stream.get_fps())
        if state == StreamState.CONNECTED:
            self.status_label.setText("상태: 연결됨")
        elif state == StreamState.DISCONNECTED:
            self.status_label.setText("상태: 연결 끊김")

    def on_ftp_upload(self, filepath: str, success: bool, error: Optional[str]):
        self.logger.log_ftp_upload(filepath, success, error)

    def shutdown(self):
        self.stop_detection()
        self.ftp_manager.disconnect()


class MainWindow(QMainWindow):
    """메인 윈도우 - 최상위 Cam1~Cam4 탭 관리"""

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.config_manager.ensure_directories()
        self.logger = Logger(
            log_dir=self.config.log_dir,
            log_level=self.config.log_level,
            retention_days=self.config.log_retention_days,
            max_file_size_mb=self.config.log_max_file_size_mb
        )
        self.camera_widgets = []
        self.app_restart_history = []
        self.init_ui()
        if self.config.auto_run_detection:
            QTimer.singleShot(1500, self.start_enabled_cameras)

    def init_ui(self):
        self.setWindowTitle(self.config.window_title)
        self.setGeometry(100, 100, self.config.window_width, self.config.window_height)
        self.camera_tabs = QTabWidget()
        self.setCentralWidget(self.camera_tabs)
        for index, camera in enumerate(self.config.cameras[:4]):
            widget = CameraWidget(index, self.config_manager, self.logger, self.restart_application, self)
            self.camera_widgets.append(widget)
            self.camera_tabs.addTab(widget, camera.name or f"Cam{index + 1}")

    def start_enabled_cameras(self):
        for widget in self.camera_widgets:
            if widget.camera.enabled:
                widget.start_stop_btn.setChecked(True)
                widget.start_detection()

    def restart_application(self, camera: CameraConfig, healthcheck):
        now = time.time()
        self.app_restart_history = [t for t in self.app_restart_history if now - t < 3600]
        if healthcheck.restart_limit_enabled and len(self.app_restart_history) >= healthcheck.max_restart_per_hour:
            self.logger.error(
                f"[{camera.name}] 프로그램 재시작 제한 초과: "
                f"1시간 {healthcheck.max_restart_per_hour}회"
            )
            return
        self.app_restart_history.append(now)
        self.logger.error(f"[{camera.name}] HealthCheck 요청으로 프로그램 재시작")
        for widget in self.camera_widgets:
            widget.shutdown()
        if getattr(sys, 'frozen', False):
            subprocess.Popen([sys.executable] + sys.argv[1:])
        else:
            subprocess.Popen([sys.executable, str(Path(__file__).resolve())] + sys.argv[1:])
        QApplication.quit()

    def closeEvent(self, event):
        for widget in self.camera_widgets:
            widget.shutdown()
        event.accept()


class TrayIcon:
    """시스템 트레이 아이콘"""

    def __init__(self, main_window: MainWindow):
        from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction

        self.main_window = main_window
        self.tray_icon = QSystemTrayIcon(main_window)
        icon_path = Path("icon.ico")
        if icon_path.exists():
            self.tray_icon.setIcon(QIcon(str(icon_path)))
        else:
            self.tray_icon.setIcon(main_window.style().standardIcon(main_window.style().SP_ComputerIcon))

        menu = QMenu()
        show_action = QAction("보이기", main_window)
        show_action.triggered.connect(main_window.show)
        menu.addAction(show_action)
        hide_action = QAction("숨기기", main_window)
        hide_action.triggered.connect(main_window.hide)
        menu.addAction(hide_action)
        menu.addSeparator()
        quit_action = QAction("종료", main_window)
        quit_action.triggered.connect(main_window.close)
        menu.addAction(quit_action)
        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self.on_activated)
        if main_window.config.show_tray_icon:
            self.tray_icon.show()

    def on_activated(self, reason):
        from PyQt5.QtWidgets import QSystemTrayIcon
        if reason == QSystemTrayIcon.DoubleClick:
            if self.main_window.isVisible():
                self.main_window.hide()
            else:
                self.main_window.show()
                self.main_window.activateWindow()

    def show_message(self, title: str, message: str):
        from PyQt5.QtWidgets import QSystemTrayIcon
        self.tray_icon.showMessage(title, message, QSystemTrayIcon.Information, 3000)


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    icon_path = Path("icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    main_window = MainWindow()
    main_window.show()
    tray_icon = TrayIcon(main_window)
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
