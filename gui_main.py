"""
Main GUI Module
PyQt5 기반 메인 GUI 및 트레이 아이콘 구현
"""
import sys
import cv2
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFrame, QScrollArea, QGridLayout,
    QFileDialog, QMessageBox, QComboBox, QSpinBox, QDoubleSpinBox,
    QTabWidget, QTextEdit, QGroupBox, QCheckBox, QSplitter
)
from PyQt5.QtCore import Qt, QTimer, pyqtSignal, QThread, QPoint
from PyQt5.QtGui import QImage, QPixmap, QIcon, QPainter, QPen, QColor, QMouseEvent
import threading

from config_manager import ConfigManager, PatternConfig, ROI
from logger import Logger
from rtsp_stream import RTSPStream, FrameBuffer, StreamState
from template_matching import TemplateMatcher, TriggerBuffer, TenengradeAnalyzer
from ftp_manager import FTPManager, FileStorageManager
from ai_classifier import AIClassifier, CombinedTrigger


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


class MainWindow(QMainWindow):
    """메인 윈도우"""

    def __init__(self):
        super().__init__()
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.logger = Logger(
            log_dir=self.config.log_dir,
            log_level=self.config.log_level,
            retention_days=self.config.log_retention_days,
            max_file_size_mb=self.config.log_max_file_size_mb
        )

        # 컴포넌트 초기화
        self.rtsp_stream = RTSPStream(
            self.config.rtsp_url,
            self.config.rtsp_reconnect_interval
        )
        self.template_matcher = TemplateMatcher()
        self.trigger_buffer = TriggerBuffer(self.config.buffer_size)
        self.backgorund_frame_cnt = 0
        self.image_save_done = False
        self.frame_buffer = FrameBuffer(self.config.max_frames)
        self.ftp_manager = FTPManager(
            host=self.config.ftp_host,
            port=self.config.ftp_port,
            username=self.config.ftp_username,
            password=self.config.ftp_password,
            remote_dir=self.config.ftp_remote_dir,
            timeout=self.config.ftp_timeout,
            enabled=self.config.ftp_enabled
        )
        self.file_storage = FileStorageManager(
            save_dir=self.config.save_dir,
            retention_days=self.config.retention_days
        )
        self.ai_classifier = AIClassifier(
            model_path=self.config.ai_model_path,
            threshold=self.config.ai_threshold
        )
        self.combined_trigger = CombinedTrigger(self.config.ai_condition_type)

        # 상태 변수
        self.is_running = False
        self.is_gathering = False
        self.current_pattern_index = 1
        self.gathered_images = []
        self.selected_image = None
        self.current_roi = None
        self.current_template = None
        self.show_roi_regions = True  # ROI 영역 표시 플래그

        # UI 초기화
        self.init_ui()
        self.setup_callbacks()

        # 타이머 설정
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_display)
        self.update_timer.start(30)  # 30ms (약 33 FPS)

        # 로그 정리 타이머
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self.cleanup_old_files)
        self.cleanup_timer.start(3600000)  # 1시간마다

        # 템플릿 로드
        self.load_templates()

    def init_ui(self):
        """UI 초기화"""
        self.setWindowTitle(self.config.window_title)
        self.setGeometry(100, 100, self.config.window_width, self.config.window_height)

        # 메인 위젯
        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        # 메인 레이아웃
        main_layout = QHBoxLayout(main_widget)

        # 스플리터
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)

        # 왼쪽 패널 (썸네일/설정)
        left_panel = self.create_left_panel()
        splitter.addWidget(left_panel)

        # 오른쪽 패널 (메인 화면)
        right_panel = self.create_right_panel()
        splitter.addWidget(right_panel)

        # 스플리터 비율 설정
        splitter.setSizes([300, 900])

    def create_left_panel(self) -> QWidget:
        """왼쪽 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 탭 위젯
        tab_widget = QTabWidget()
        layout.addWidget(tab_widget)

        # 라이브 탭
        live_tab = self.create_live_tab()
        tab_widget.addTab(live_tab, "라이브")

        # 설정 탭
        config_tab = self.create_config_tab()
        tab_widget.addTab(config_tab, "설정")

        # 로그 탭
        log_tab = self.create_log_tab()
        tab_widget.addTab(log_tab, "로그")

        return panel

    def create_live_tab(self) -> QWidget:
        """라이브 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 제어 버튼 그룹
        control_group = QGroupBox("제어")
        control_layout = QVBoxLayout()

        # START/STOP 버튼
        self.start_stop_btn = QPushButton("START")
        self.start_stop_btn.setCheckable(True)
        self.start_stop_btn.clicked.connect(self.toggle_start_stop)
        self.start_stop_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                font-size: 14px;
                padding: 10px;
            }
            QPushButton:checked {
                background-color: #f44336;
            }
        """)
        control_layout.addWidget(self.start_stop_btn)

        # 이미지 수집 버튼
        self.gather_btn = QPushButton("이미지 수집")
        self.gather_btn.clicked.connect(self.start_image_gathering)
        control_layout.addWidget(self.gather_btn)

        # 템플릿 등록 버튼
        self.template_reg_btn = QPushButton("템플릿 등록")
        self.template_reg_btn.clicked.connect(self.show_template_registration)
        control_layout.addWidget(self.template_reg_btn)

        control_group.setLayout(control_layout)
        layout.addWidget(control_group)

        # 상태 정보
        status_group = QGroupBox("상태")
        status_layout = QVBoxLayout()

        self.status_label = QLabel("상태: 대기중")
        status_layout.addWidget(self.status_label)

        self.fps_label = QLabel("FPS: 0.0")
        status_layout.addWidget(self.fps_label)

        self.buffer_label = QLabel("버퍼: 0/0")
        status_layout.addWidget(self.buffer_label)

        status_group.setLayout(status_layout)
        layout.addWidget(status_group)

        # 썸네일 영역
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
        """설정 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # RTSP 설정
        rtsp_group = QGroupBox("RTSP 설정")
        rtsp_layout = QVBoxLayout()

        rtsp_layout.addWidget(QLabel("RTSP URL:"))
        self.rtsp_url_edit = QTextEdit()
        self.rtsp_url_edit.setMaximumHeight(60)
        self.rtsp_url_edit.setPlainText(self.config.rtsp_url)
        rtsp_layout.addWidget(self.rtsp_url_edit)

        rtsp_group.setLayout(rtsp_layout)
        layout.addWidget(rtsp_group)

        # 패턴 설정
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

        pattern_layout.addWidget(QLabel("모든 패턴 매칭 필요:"))
        self.require_all_check = QCheckBox()
        self.require_all_check.setChecked(self.config.require_all_patterns)
        pattern_layout.addWidget(self.require_all_check)

        pattern_group.setLayout(pattern_layout)
        layout.addWidget(pattern_group)

        # 버퍼 설정
        buffer_group = QGroupBox("버퍼 설정")
        buffer_layout = QVBoxLayout()

        buffer_layout.addWidget(QLabel("버퍼 크기:"))
        self.buffer_size_spin = QSpinBox()
        self.buffer_size_spin.setMinimum(1)
        self.buffer_size_spin.setMaximum(100)
        self.buffer_size_spin.setValue(self.config.buffer_size)
        buffer_layout.addWidget(self.buffer_size_spin)

        buffer_layout.addWidget(QLabel("수집 시간 (초):"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setMinimum(1)
        self.duration_spin.setMaximum(60)
        self.duration_spin.setValue(self.config.frame_save_duration)
        buffer_layout.addWidget(self.duration_spin)

        buffer_group.setLayout(buffer_layout)
        layout.addWidget(buffer_group)

        # 저장 버튼
        save_btn = QPushButton("설정 저장")
        save_btn.clicked.connect(self.save_config)
        layout.addWidget(save_btn)

        layout.addStretch()

        return tab

    def create_log_tab(self) -> QWidget:
        """로그 탭 생성"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        return tab

    def create_right_panel(self) -> QWidget:
        """오른쪽 패널 생성"""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 메인 이미지 라벨
        self.main_image_label = ImageLabel()
        self.main_image_label.setAlignment(Qt.AlignCenter)
        self.main_image_label.setStyleSheet("border: 2px solid #333; background-color: #000;")
        layout.addWidget(self.main_image_label)

        # 템플릿 인덱스 선택
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

        # 제어 버튼 (ROI/Template)
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

        # 현재 패턴 정보
        self.pattern_info_label = QLabel("현재 패턴: 없음")
        layout.addWidget(self.pattern_info_label)

        return panel

    def setup_callbacks(self):
        """콜백 설정"""
        # RTSP 스트림 콜백
        self.rtsp_stream.set_error_callback(self.on_rtsp_error)
        self.rtsp_stream.set_state_callback(self.on_rtsp_state_change)

        # 이미지 라벨 콜백
        self.main_image_label.set_roi_callback(self.on_roi_selected)
        self.main_image_label.set_template_callback(self.on_template_selected)

        # FTP 콜백
        self.ftp_manager.set_upload_callback(self.on_ftp_upload)

    def load_templates(self):
        """설정에서 템플릿 로드"""
        patterns = self.config_manager.get_all_patterns()
        for pattern in patterns:
            if pattern.template_path and Path(pattern.template_path).exists():
                self.template_matcher.load_template(
                    index=pattern.index,
                    template_path=pattern.template_path,
                    roi=(pattern.roi.x, pattern.roi.y, pattern.roi.width, pattern.roi.height),
                    threshold=pattern.score_threshold
                )
                self.logger.info(f"템플릿 로드됨: Index={pattern.index}, Path={pattern.template_path}")

    def toggle_start_stop(self):
        """START/STOP 토글"""
        if self.start_stop_btn.isChecked():
            self.start_detection()
        else:
            self.stop_detection()

    def start_detection(self):
        """검출 시작"""
        self.is_running = True
        self.start_stop_btn.setText("STOP")

        # RTSP 스트림 시작
        self.rtsp_stream.start()

        self.logger.info("검출 시작")
        self.status_label.setText("상태: 실행중")

    def stop_detection(self):
        """검출 중지"""
        self.is_running = False
        self.start_stop_btn.setText("START")

        # RTSP 스트림 중지
        self.rtsp_stream.stop()

        self.logger.info("검출 중지")
        self.status_label.setText("상태: 대기중")

    def start_image_gathering(self):
        """이미지 수집 시작"""
        if self.is_gathering:
            return

        self.is_gathering = True
        self.gather_btn.setEnabled(False)
        self.gather_btn.setText("수집중...")

        # 수집 스레드 시작
        thread = threading.Thread(target=self._gather_images, daemon=True)
        thread.start()

    def _gather_images(self):
        """이미지 수집 스레드"""
        self.frame_buffer.clear()
        self.gathered_images = []

        duration = self.config.frame_save_duration
        start_time = datetime.now()

        # 저장 디렉토리 생성: yyyyMMdd_hhmmss
        save_dir_name = start_time.strftime("%Y%m%d_%H%M%S")
        save_dir = Path("gathered_images") / save_dir_name
        save_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"이미지 수집 시작: Duration={duration}s, 저장 경로: {save_dir}")

        frame_count = 0

        while self.is_gathering:
            frame_info = self.rtsp_stream.get_frame()
            if frame_info:
                # 메모리에 저장
                self.frame_buffer.add_frame(frame_info.frame, frame_info.timestamp)

                # 파일로 저장 (yyyyMMdd_hhmmss_000.bmp 형식)
                frame_time = datetime.fromtimestamp(frame_info.timestamp)
                filename = frame_time.strftime("%Y%m%d_%H%M%S") + f"_{frame_count:03d}.bmp"
                filepath = save_dir / filename
                cv2.imwrite(str(filepath), frame_info.frame)
                frame_count += 1

            # 시간 확인
            elapsed = (datetime.now() - start_time).total_seconds()
            if elapsed >= duration or self.frame_buffer.is_full():
                break

        self.gathered_images = self.frame_buffer.get_frames()
        self.is_gathering = False

        # UI 업데이트 (메인 스레드에서)
        self.gather_btn.setEnabled(True)
        self.gather_btn.setText("이미지 수집")

        self.logger.log_image_gathering(duration, len(self.gathered_images))
        self.logger.info(f"이미지 저장 완료: {save_dir}, 총 {frame_count}개 파일")

        # 썸네일 표시
        self.show_thumbnails()

    def show_thumbnails(self):
        """썸네일 표시"""
        # 기존 위젯 제거
        for i in reversed(range(self.thumbnail_layout.count())):
            self.thumbnail_layout.itemAt(i).widget().setParent(None)

        # 썸네일 추가
        cols = 2
        for i, img_data in enumerate(self.gathered_images):
            thumbnail = ThumbnailLabel(img_data['frame'], i)
            thumbnail.clicked.connect(lambda idx=i: self.select_image(idx))
            self.thumbnail_layout.addWidget(thumbnail, i // cols, i % cols)

    def select_image(self, index: int):
        """이미지 선택 - 메인 화면에 표시하고 라이브 업데이트 중지"""
        if 0 <= index < len(self.gathered_images):
            self.selected_image = self.gathered_images[index]['frame']
            self.main_image_label.set_image(self.selected_image)
            self.main_image_label.set_roi_rects([])
            self.main_image_label.set_template_rects([])
            self.logger.info(f"이미지 선택됨: Index={index}")

    def show_template_registration(self):
        """템플릿 등록 화면 표시"""
        if not self.gathered_images:
            QMessageBox.warning(self, "경고", "먼저 이미지를 수집해주세요.")
            return

        # 썸네일 표시
        self.show_thumbnails()

        self.logger.info("템플릿 등록 모드")

    def toggle_roi_mode(self):
        """ROI 모드 토글"""
        if self.roi_btn.isChecked():
            self.template_btn.setChecked(False)
            self.main_image_label.set_draw_mode('roi')
            self.logger.info("ROI 설정 모드 활성화")
        else:
            self.main_image_label.set_draw_mode(None)

    def toggle_template_mode(self):
        """Template 모드 토글"""
        if self.template_btn.isChecked():
            self.roi_btn.setChecked(False)
            self.main_image_label.set_draw_mode('template')
            self.logger.info("Template 설정 모드 활성화")
        else:
            self.main_image_label.set_draw_mode(None)

    def clear_regions(self):
        """영역 지우기"""
        self.main_image_label.set_roi_rects([])
        self.main_image_label.set_template_rects([])
        self.current_roi = None
        self.current_template = None
        self.show_roi_regions = False  # ROI 표시 비활성화
        self.roi_btn.setChecked(False)
        self.template_btn.setChecked(False)
        self.main_image_label.set_draw_mode(None)
        self.logger.info("영역 지우기 실행됨")

    def on_roi_selected(self, roi: Tuple[int, int, int, int]):
        """ROI 선택 콜백"""
        self.current_roi = roi
        self.main_image_label.set_roi_rects([roi])
        self.roi_btn.setChecked(False)
        self.main_image_label.set_draw_mode(None)
        self.logger.info(f"ROI 설정됨: {roi}")

    def on_template_selected(self, template: Tuple[int, int, int, int]):
        """Template 선택 콜백"""
        if self.selected_image is None:
            QMessageBox.warning(self, "경고", "먼저 이미지를 선택해주세요.")
            return
        self.logger.info(f"Template X, Y, Width, Height : {template}")
        self.current_template = template
        self.main_image_label.set_template_rects([template])
        self.template_btn.setChecked(False)
        self.main_image_label.set_draw_mode(None)

        # 템플릿 저장
        self.save_template()
        self.logger.info(f"Template 설정됨: {template}")

        # 템플릿 저장 후 선택된 이미지 해제 (라이브 화면으로 복귀)
        self.selected_image = None
        self.main_image_label.set_roi_rects([])
        self.main_image_label.set_template_rects([])

    def save_template(self):
        """템플릿 저장"""
        if self.current_template is None or self.selected_image is None:
            return

        index = self.template_index_spin.value()
        threshold = self.threshold_spin.value()

        # ROI 설정 (없으면 전체 이미지)
        if self.current_roi:
            roi = self.current_roi
        else:
            h, w = self.selected_image.shape[:2]
            roi = (0, 0, w, h)

        # 템플릿 이미지 추출
        x, y, w, h = self.current_template
        template_img = self.selected_image[y:y+h, x:x+w]

        # 템플릿 파일 저장
        template_dir = Path("templates")
        template_dir.mkdir(exist_ok=True)
        template_path = template_dir / f"template_{index}.png"
        cv2.imwrite(str(template_path), template_img)

        # 설정 업데이트
        pattern_config = PatternConfig(
            index=index,
            template_path=str(template_path),
            score_threshold=threshold,
            roi=ROI(x=roi[0], y=roi[1], width=roi[2], height=roi[3])
        )
        self.config_manager.update_pattern(index, pattern_config)

        # 템플릿 매처에 로드
        self.template_matcher.load_template_from_image(
            index=index,
            template_image=template_img,
            roi=roi,
            threshold=threshold
        )

        self.logger.log_template_registration(index, str(template_path), roi)
        self.pattern_info_label.setText(f"현재 패턴: Index={index}, Threshold={threshold}")

        QMessageBox.information(self, "성공", f"템플릿이 저장되었습니다.\nIndex: {index}")

    def save_config(self):
        """설정 저장"""
        # RTSP URL
        self.config_manager.config.rtsp_url = self.rtsp_url_edit.toPlainText().strip()

        # 패턴 설정
        self.config_manager.config.require_all_patterns = self.require_all_check.isChecked()

        # 버퍼 설정
        self.config_manager.config.buffer_size = self.buffer_size_spin.value()
        self.config_manager.config.frame_save_duration = self.duration_spin.value()

        # 저장
        self.config_manager.save_config()
        self.config = self.config_manager.get_config()

        # 버퍼 크기 업데이트
        self.trigger_buffer = TriggerBuffer(self.config.buffer_size)

        self.logger.info("설정 저장됨")
        QMessageBox.information(self, "성공", "설정이 저장되었습니다.")

    def update_display(self):
        """디스플레이 업데이트"""
        # FPS 업데이트
        fps = self.rtsp_stream.get_fps()
        self.fps_label.setText(f"FPS: {fps:.1f}")

        # 버퍼 상태 업데이트
        buffer_size = self.trigger_buffer.size()
        self.buffer_label.setText(f"버퍼: {buffer_size}/{self.config.buffer_size}")

        # 라이브 이미지 업데이트 (실행 중이고 선택된 이미지가 없을 때만)
        if self.is_running and self.selected_image is None:
            frame_info = self.rtsp_stream.get_latest_frame()
            if frame_info:
                frame = frame_info.frame

                # Template Matching 수행
                matched_rects = []  # 매칭된 영역 리스트
                if self.template_matcher.get_template_count() > 0:
                    matched, results = self.template_matcher.match_all(
                        frame,
                        require_all=self.config.require_all_patterns
                    )

                    #KMS
                    # 매칭 결과 로그
                    # for result in results:
                    #     self.logger.log_pattern_match(
                    #         result.pattern_index,
                    #         result.score,
                    #         result.threshold,
                    #         result.matched
                    #     )

                    # 매칭된 영역 수집 (매칭된 것만)
                    for result in results:
                        if result.matched:
                            # 템플릿 크기 가져오기
                            template_size = self.template_matcher.get_template_size(result.pattern_index)
                            if template_size:
                                template_w, template_h = template_size
                                matched_rects.append((
                                    result.location[0],  # x
                                    result.location[1],  # y
                                    template_w,         # width
                                    template_h,         # height
                                    result.score        # score
                                ))

                    # 매칭되면 버퍼에 추가
                    if matched:
                        print("matching!", results[0].score)
                        if self.image_save_done == False:
                            if self.trigger_buffer.is_full() == False:
                                self.trigger_buffer.add_frame(frame)
                            else:
                                self.process_buffer()
                                self.image_save_done = True
                        self.backgorund_frame_cnt = 0
                        print("buffer Full", self.image_save_done)
                        print("buffer Full", self.backgorund_frame_cnt)
                    else:
                        self.backgorund_frame_cnt += 1
                        if self.backgorund_frame_cnt > 5 and self.image_save_done == True:
                            self.image_save_done = False
                            self.trigger_buffer.clear()
                            print("buffer clear", self.image_save_done)
                            print("buffer clear", self.backgorund_frame_cnt)

                            # 버퍼 비우기
                            
                # ROI 영역 표시 (플래그가 True일 때만)
                roi_rects = []
                template_rects = []
                if self.show_roi_regions:
                    patterns = self.config_manager.get_all_patterns()
                    for pattern in patterns:
                        roi_rects.append((pattern.roi.x, pattern.roi.y, pattern.roi.width, pattern.roi.height))

                self.main_image_label.set_roi_rects(roi_rects)
                self.main_image_label.set_template_rects(template_rects)
                self.main_image_label.set_matched_rects(matched_rects)
                self.main_image_label.set_image(frame)

    def process_buffer(self):
        """버퍼 처리"""
        # 가장 선명한 이미지 선택
        best_image = self.trigger_buffer.get_best_frame()
        if best_image is not None:
            # 선명도 계산
            tenengrade_score = TenengradeAnalyzer.calculate(best_image)

            # 이미지 저장
            filepath = self.file_storage.save_image(best_image)
            if filepath:
                self.logger.log_image_saved(filepath, tenengrade_score)

                # FTP 업로드
                if self.ftp_manager.is_enabled():
                    date_str = datetime.now().strftime("%Y%m%d")
                    self.ftp_manager.upload_file_async(filepath, date_str)



    def cleanup_old_files(self):
        """오래된 파일 정리"""
        self.file_storage.cleanup_old_files()
        # 로그 정리는 Logger 클래스에서 자동 처리

    def on_rtsp_error(self, error_msg: str):
        """RTSP 에러 콜백"""
        self.logger.error(f"RTSP 오류: {error_msg}")
        self.status_label.setText(f"상태: 오류 - {error_msg}")

    def on_rtsp_state_change(self, state: StreamState):
        """RTSP 상태 변경 콜백"""
        self.logger.log_rtsp_status(state == StreamState.CONNECTED, self.rtsp_stream.get_fps())

    def on_ftp_upload(self, filepath: str, success: bool, error: Optional[str]):
        """FTP 업로드 콜백"""
        self.logger.log_ftp_upload(filepath, success, error)

    def closeEvent(self, event):
        """종료 이벤트"""
        self.stop_detection()
        self.ftp_manager.disconnect()
        event.accept()


class TrayIcon:
    """시스템 트레이 아이콘"""

    def __init__(self, main_window: MainWindow):
        from PyQt5.QtWidgets import QSystemTrayIcon, QMenu, QAction

        self.main_window = main_window
        self.tray_icon = QSystemTrayIcon(main_window)

        # 아이콘 설정 (기본 아이콘 사용)
        self.tray_icon.setIcon(main_window.style().standardIcon(main_window.style().SP_ComputerIcon))

        # 메뉴 생성
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

        # 더블클릭 이벤트
        self.tray_icon.activated.connect(self.on_activated)

        # 트레이 아이콘 표시
        if main_window.config.show_tray_icon:
            self.tray_icon.show()

    def on_activated(self, reason):
        """트레이 아이콘 활성화 이벤트"""
        if reason == QSystemTrayIcon.DoubleClick:
            if self.main_window.isVisible():
                self.main_window.hide()
            else:
                self.main_window.show()
                self.main_window.activateWindow()

    def show_message(self, title: str, message: str):
        """트레이 메시지 표시"""
        self.tray_icon.showMessage(title, message, QSystemTrayIcon.Information, 3000)


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')

    # 메인 윈도우 생성
    main_window = MainWindow()
    main_window.show()

    # 트레이 아이콘 생성
    tray_icon = TrayIcon(main_window)

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
