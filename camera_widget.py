"""Per-camera GUI runtime widget for VideoTriggerDetection."""

import shutil
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFrame,
    QScrollArea, QGridLayout, QMessageBox, QComboBox, QSpinBox,
    QDoubleSpinBox, QTabWidget, QTextEdit, QGroupBox, QCheckBox,
    QSplitter, QLineEdit, QSizePolicy,
)

from ai_classifier import AIClassifier, CombinedTrigger
from config_manager import ConfigManager, PatternConfig, ROI
from ftp_manager import FTPManager, FileStorageManager
from gui_widgets import ImageLabel, ThumbnailLabel
from logger import Logger
from rtsp_stream import RTSPStream, FrameBuffer, StreamState
from startup_manager import apply_startup
from template_matching import TemplateMatcher, TriggerBuffer, TenengradeAnalyzer


class CameraWidget(QWidget):
    """카메라 1대의 라이브/설정/로그/검출 런타임을 독립 보유하는 위젯"""

    rtsp_error_signal = pyqtSignal(str)
    rtsp_state_signal = pyqtSignal(object)
    log_append_signal = pyqtSignal(str)
    image_gathered_signal = pyqtSignal(object, str, int, int)

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
        self.previous_health_gray = None
        self.last_health_diff = None
        self.last_health_change_time = time.time()
        self.freeze_same_count = 0
        self.last_freeze_check_at = 0.0
        self.health_started_at = time.time()
        self.first_frame_received = False
        self.last_stream_restart_at = 0.0
        self.health_restart_in_progress = False
        self.consecutive_health_failures = 0
        self.last_health_failure_counted_at = 0.0
        self.stream_restart_history = []
        self.last_rtsp_state_log = None
        self.last_rtsp_state_log_at = 0.0
        self.last_health_error_reason = None
        self.last_health_error_logged_at = 0.0

        self.rtsp_error_signal.connect(self.on_rtsp_error)
        self.rtsp_state_signal.connect(self.on_rtsp_state_change)
        self.log_append_signal.connect(self.append_log_text)
        self.image_gathered_signal.connect(self.on_image_gathered)

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
        self.log_append_signal.emit(f"{datetime.now().strftime('%H:%M:%S')} {text}")

    def log_error(self, message: str):
        text = f"[{self.camera.name}] {message}"
        self.logger.error(text)
        self.log_append_signal.emit(f"{datetime.now().strftime('%H:%M:%S')} ERROR {text}")

    def append_log_text(self, line: str):
        if hasattr(self, 'log_text'):
            self.log_text.append(line)

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

        self.live_view_btn = QPushButton("라이브 화면으로 돌아가기")
        self.live_view_btn.clicked.connect(self.return_to_live_view)
        control_layout.addWidget(self.live_view_btn)
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
        file_layout.addWidget(QLabel("품질/압축률(10~100):"))
        self.quality_spin = QSpinBox()
        self.quality_spin.setRange(10, 100)
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
        health_layout.addWidget(QLabel("프레임 미수신/시작 대기 제한 시간(초):"))
        self.health_timeout_spin = QSpinBox()
        self.health_timeout_spin.setRange(1, 3600)
        self.health_timeout_spin.setValue(self.camera.healthcheck.timeout_sec)
        health_layout.addWidget(self.health_timeout_spin)
        health_layout.addWidget(QLabel("영상 정지 검사 주기(초):"))
        self.freeze_interval_spin = QSpinBox()
        self.freeze_interval_spin.setRange(1, 3600)
        self.freeze_interval_spin.setValue(self.camera.healthcheck.freeze_check_interval_sec)
        health_layout.addWidget(self.freeze_interval_spin)
        health_layout.addWidget(QLabel("영상 정지 연속 동일 판정 횟수:"))
        self.freeze_count_spin = QSpinBox()
        self.freeze_count_spin.setRange(1, 100)
        self.freeze_count_spin.setValue(self.camera.healthcheck.freeze_consecutive_count)
        health_layout.addWidget(self.freeze_count_spin)
        health_layout.addWidget(QLabel("영상 변화 감지 임계값:"))
        self.freeze_diff_spin = QDoubleSpinBox()
        self.freeze_diff_spin.setRange(0.0, 255.0)
        self.freeze_diff_spin.setDecimals(3)
        self.freeze_diff_spin.setSingleStep(0.1)
        self.freeze_diff_spin.setValue(self.camera.healthcheck.freeze_diff_threshold)
        health_layout.addWidget(self.freeze_diff_spin)
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
            self.start_minimized_check = QCheckBox("프로그램 시작 시 최소화/트레이로 숨김")
            self.start_minimized_check.setChecked(self.config.start_minimized)
            system_layout.addWidget(self.start_minimized_check)
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
        self.rtsp_stream.set_error_callback(self.rtsp_error_signal.emit)
        self.rtsp_stream.set_state_callback(self.rtsp_state_signal.emit)
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
        now = time.time()
        self.health_started_at = now
        self.first_frame_received = False
        self.last_health_change_time = now
        self.last_stream_restart_at = 0.0
        self.health_restart_in_progress = False
        self.last_health_failure_counted_at = 0.0
        self.previous_health_gray = None
        self.last_freeze_check_at = 0.0
        self.freeze_same_count = 0
        self.last_health_diff = None
        self.selected_image = None
        self.main_image_label.set_status_message("RTSP 연결 중...", QColor(255, 220, 80))
        self.rtsp_stream.start()
        self.log_info("검출 시작")
        self.status_label.setText("상태: 실행중")

    def stop_detection(self):
        self.is_running = False
        self.start_stop_btn.setChecked(False)
        self.start_stop_btn.setText("START")
        self.health_restart_in_progress = False
        self.rtsp_stream.stop()
        self.main_image_label.set_status_message("")
        self.log_info("검출 중지")
        self.status_label.setText("상태: 대기중")
        self.health_label.setText("Health: 대기중")

    def start_image_gathering(self):
        if self.is_gathering:
            return
        self.clear_gathered_image_files()
        self.is_gathering = True
        self.gathered_images = []
        self.selected_image = None
        self.show_thumbnails()
        self.gather_btn.setEnabled(False)
        self.gather_btn.setText("수집중...")
        thread = threading.Thread(target=self._gather_images, daemon=True)
        thread.start()

    def clear_gathered_image_files(self):
        """이미지 수집 버튼으로 저장된 이전 수집 이미지만 삭제한다."""
        gather_root = Path("gathered_images") / self.camera.name
        if not gather_root.exists():
            return
        deleted_count = 0
        try:
            for item in gather_root.iterdir():
                if item.is_dir():
                    file_count = sum(1 for child in item.rglob("*") if child.is_file())
                    shutil.rmtree(item)
                    deleted_count += file_count
                elif item.is_file():
                    item.unlink()
                    deleted_count += 1
            if deleted_count:
                self.log_info(f"기존 수집 이미지 삭제 완료: {gather_root}, 총 {deleted_count}개 파일")
        except OSError as exc:
            self.log_error(f"기존 수집 이미지 삭제 실패: {exc}")

    def _gather_images(self):
        self.frame_buffer.clear()
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
        frames = self.frame_buffer.get_frames()
        self.image_gathered_signal.emit(frames, str(save_dir), frame_count, duration)

    def on_image_gathered(self, frames: list, save_dir: str, frame_count: int, duration: int):
        self.gathered_images = frames
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
            self.main_image_label.set_status_message("")
            self.main_image_label.set_image(self.selected_image)
            self.main_image_label.set_roi_rects([])
            self.main_image_label.set_template_rects([])
            self.log_info(f"이미지 선택됨: Index={index}")

    def return_to_live_view(self):
        self.selected_image = None
        self.roi_btn.setChecked(False)
        self.template_btn.setChecked(False)
        self.main_image_label.set_draw_mode(None)
        self.main_image_label.set_roi_rects([])
        self.main_image_label.set_template_rects([])
        if self.rtsp_stream.state == StreamState.CONNECTED:
            self.main_image_label.set_status_message("")
            frame_info = self.rtsp_stream.get_latest_frame()
            if frame_info:
                self.main_image_label.set_image(frame_info.frame)
        else:
            self.main_image_label.set_status_message("RTSP ERROR / 연결 끊김")
        self.log_info("라이브 화면으로 복귀")

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
        self.camera.healthcheck.freeze_check_interval_sec = self.freeze_interval_spin.value()
        self.camera.healthcheck.freeze_consecutive_count = self.freeze_count_spin.value()
        self.camera.healthcheck.freeze_diff_threshold = self.freeze_diff_spin.value()
        self.camera.healthcheck.restart_stream = self.restart_stream_check.isChecked()
        self.camera.healthcheck.restart_app = self.restart_app_check.isChecked()
        self.camera.healthcheck.restart_limit_enabled = self.restart_limit_check.isChecked()
        self.camera.healthcheck.max_restart_per_hour = self.max_restart_spin.value()

        if self.camera_index == 0:
            self.config.auto_start_enabled = self.auto_start_check.isChecked()
            self.config.auto_run_detection = self.auto_run_check.isChecked()
            self.config.start_minimized = self.start_minimized_check.isChecked()
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
                if not self.first_frame_received:
                    self.first_frame_received = True
                    self.consecutive_health_failures = 0
                    self.last_health_failure_counted_at = 0.0
                    self.health_restart_in_progress = False
                    self.last_health_error_reason = None
                    self.last_health_error_logged_at = 0.0
                    self.main_image_label.set_status_message("")
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
            current_gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if len(small.shape) == 3 else small
            if self.previous_health_gray is None or current_gray.shape != self.previous_health_gray.shape:
                self.previous_health_gray = current_gray.copy()
                self.last_health_diff = None
                self.last_health_change_time = time.time()
                self.freeze_same_count = 0
                return True
            diff = float(np.mean(cv2.absdiff(current_gray, self.previous_health_gray)))
            self.last_health_diff = diff
            self.previous_health_gray = current_gray.copy()
            if diff > self.camera.healthcheck.freeze_diff_threshold:
                self.last_health_change_time = time.time()
                self.freeze_same_count = 0
                return True
            self.freeze_same_count += 1
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
        reconnecting_states = (
            StreamState.CONNECTING,
            StreamState.DISCONNECTED,
            StreamState.ERROR,
        )
        stream_reconnecting = self.rtsp_stream.state in reconnecting_states
        reconnect_or_restart_pending = stream_reconnecting or self.health_restart_in_progress

        in_startup_grace = not self.first_frame_received and now - self.health_started_at < hc.timeout_sec
        no_frame_timeout = (not last_ts) or (now - last_ts > hc.timeout_sec)
        current_frame = None if self.selected_image is not None else self.main_image_label.current_image
        freeze_check_due = now - self.last_freeze_check_at >= hc.freeze_check_interval_sec
        if not reconnect_or_restart_pending and freeze_check_due:
            self.last_freeze_check_at = now
            self._frame_changed(current_frame)
        freeze_timeout = (
            freeze_check_due
            and current_frame is not None
            and self.selected_image is None
            and self.first_frame_received
            and self.freeze_same_count >= hc.freeze_consecutive_count
        )

        if in_startup_grace:
            if reconnect_or_restart_pending:
                self.health_label.setText("Health: RTSP 재연결 중")
            else:
                self.health_label.setText("Health: 첫 프레임 대기중")
            return

        if reconnect_or_restart_pending:
            self.health_label.setText("Health: RTSP 재연결 중")

        if not reconnect_or_restart_pending and not no_frame_timeout and not freeze_timeout:
            self.consecutive_health_failures = 0
            self.last_health_failure_counted_at = 0.0
            self.health_label.setText("Health: 정상")
            self.last_health_error_reason = None
            self.last_health_error_logged_at = 0.0
            return

        if reconnect_or_restart_pending and not no_frame_timeout:
            return

        reason = "프레임 수신 없음" if no_frame_timeout else "영상 변화 없음"
        self.health_label.setText(f"Health: 이상 감지 - {reason}")
        if self.last_health_error_reason != reason or now - self.last_health_error_logged_at >= 5.0:
            last_frame_age = now - last_ts if last_ts else None
            state = self.rtsp_stream.state
            state_name = state.name if hasattr(state, "name") else str(state)
            last_health_diff = (
                f"{self.last_health_diff:.3f}"
                if self.last_health_diff is not None
                else "None"
            )
            last_frame_age_text = f"{last_frame_age:.3f}" if last_frame_age is not None else "None"
            self.log_error(
                "HealthCheck 이상 감지: "
                f"reason={reason}, "
                f"rtsp_stream.state={state_name}, "
                f"fps={self.rtsp_stream.get_fps():.3f}, "
                f"frame_number={self.rtsp_stream.get_frame_number()}, "
                f"last_frame_age={last_frame_age_text}, "
                f"last_health_diff={last_health_diff}, "
                f"first_frame_received={self.first_frame_received}, "
                f"health_restart_in_progress={self.health_restart_in_progress}"
            )
            self.last_health_error_reason = reason
            self.last_health_error_logged_at = now

        reconnect_interval = self.camera.rtsp_reconnect_interval
        reconnect_waiting = now - self.last_stream_restart_at < reconnect_interval

        # RTSPStream._capture_loop가 CONNECTED가 아닐 때 자동 재연결을 담당한다.
        # HealthCheck는 재연결 중인 스트림을 다시 끊지 않고 실패 주기만 집계해
        # 반복 실패 시 프로그램 재시작 옵션이 동작하도록 한다.
        if reconnect_or_restart_pending:
            if not reconnect_waiting:
                self._count_health_reconnect_failure(now)
        elif hc.restart_stream and not reconnect_waiting:
            # CONNECTED 상태인데 프레임/변화가 멈춘 경우에만 캡처 스레드에 강제 재연결을 요청한다.
            if self.restart_stream():
                self._count_health_reconnect_failure(now)

        if hc.restart_app and self.consecutive_health_failures >= 3 and self.app_restart_callback:
            self.log_error("재연결 반복 실패: 프로그램 재시작 요청")
            self.app_restart_callback(self.camera, hc)

    def _count_health_reconnect_failure(self, now: float) -> bool:
        """재연결 시도/대기 주기당 HealthCheck 실패를 최대 1회 집계한다."""
        reconnect_interval = max(1, self.camera.rtsp_reconnect_interval)
        if now - self.last_health_failure_counted_at < reconnect_interval:
            return False
        self.last_health_failure_counted_at = now
        self.consecutive_health_failures += 1
        return True

    def restart_stream(self):
        now = time.time()
        if now - self.last_stream_restart_at < self.camera.rtsp_reconnect_interval:
            return False
        self.last_stream_restart_at = now
        self.health_started_at = now
        self.first_frame_received = False
        self.last_health_change_time = now
        self.freeze_same_count = 0
        self.previous_health_gray = None
        self.last_freeze_check_at = 0.0
        self.last_health_diff = None
        self.health_restart_in_progress = True
        self.stream_restart_history = [t for t in self.stream_restart_history if now - t < 3600]
        self.stream_restart_history.append(now)
        self.health_label.setText("Health: RTSP 재연결 중")
        self.log_info("RTSP 재연결 시도")
        self.rtsp_stream.request_restart()
        return True

    def cleanup_old_files(self):
        self.file_storage.cleanup_old_files()

    def on_rtsp_error(self, error_msg: str):
        self.log_error(f"RTSP 오류: {error_msg}")
        self.status_label.setText(f"상태: 오류 - {error_msg}")

    def on_rtsp_state_change(self, state: StreamState):
        now = time.time()
        should_log = not (
            self.last_rtsp_state_log == state
            and now - self.last_rtsp_state_log_at < 5.0
        )
        if should_log and state in (StreamState.CONNECTED, StreamState.DISCONNECTED):
            self.logger.log_rtsp_status(state == StreamState.CONNECTED, self.rtsp_stream.get_fps())
            self.last_rtsp_state_log = state
            self.last_rtsp_state_log_at = now
        if state == StreamState.CONNECTED:
            self.health_started_at = now
            self.first_frame_received = False
            self.last_health_change_time = now
            self.freeze_same_count = 0
            self.previous_health_gray = None
            self.last_freeze_check_at = 0.0
            self.last_health_diff = None
            self.health_label.setText("Health: 첫 프레임 대기중")
            self.main_image_label.set_status_message("")
            self.status_label.setText("상태: 연결됨")
        elif state == StreamState.DISCONNECTED:
            self.health_label.setText("Health: RTSP 재연결 중")
            self.main_image_label.set_status_message("RTSP 재연결 중", QColor(255, 220, 80))
            self.status_label.setText("상태: RTSP 재연결 중")
        elif state == StreamState.CONNECTING:
            self.health_label.setText("Health: RTSP 재연결 중")
            self.main_image_label.set_status_message("RTSP 재연결 중", QColor(255, 220, 80))
            self.status_label.setText("상태: RTSP 재연결 중")
        elif state == StreamState.ERROR:
            self.health_label.setText("Health: RTSP 재연결 중")
            self.main_image_label.set_status_message("RTSP 재연결 중", QColor(255, 220, 80))
            self.status_label.setText("상태: RTSP 재연결 중")

    def on_ftp_upload(self, filepath: str, success: bool, error: Optional[str]):
        self.logger.log_ftp_upload(filepath, success, error)

    def shutdown(self):
        self.stop_detection()
        self.ftp_manager.disconnect()
