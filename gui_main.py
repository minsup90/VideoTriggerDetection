"""Main application entry point and top-level window for VideoTriggerDetection."""

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

from PyQt5.QtCore import QPoint, Qt, QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import (
    QApplication, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSizePolicy, QStyle, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
)

from camera_widget import CameraWidget
from config_manager import ConfigManager, CameraConfig
from logger import Logger


class TitleBar(QWidget):
    """Custom title bar so the top chrome matches the app theme."""

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self.drag_start = QPoint()
        self.setObjectName("titleBar")
        self.setFixedHeight(42)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 8, 0)
        layout.setSpacing(8)

        icon_label = QLabel()
        icon_label.setObjectName("titleIcon")
        icon = main_window.windowIcon() or main_window.style().standardIcon(QStyle.SP_ComputerIcon)
        icon_label.setPixmap(icon.pixmap(18, 18))
        layout.addWidget(icon_label)

        self.title_label = QLabel(main_window.windowTitle())
        self.title_label.setObjectName("titleLabel")
        self.title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self.title_label)

        self.minimize_btn = self._create_button("−", "최소화")
        self.maximize_btn = self._create_button("□", "최대화/복원")
        self.close_btn = self._create_button("×", "닫기")
        self.close_btn.setObjectName("titleCloseButton")
        layout.addWidget(self.minimize_btn)
        layout.addWidget(self.maximize_btn)
        layout.addWidget(self.close_btn)

        self.minimize_btn.clicked.connect(main_window.showMinimized)
        self.maximize_btn.clicked.connect(self.toggle_maximized)
        self.close_btn.clicked.connect(main_window.close)

    def _create_button(self, text: str, tooltip: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("titleButton")
        button.setToolTip(tooltip)
        button.setFixedSize(34, 28)
        button.setFocusPolicy(Qt.NoFocus)
        return button

    def set_title(self, title: str):
        self.title_label.setText(title)

    def toggle_maximized(self):
        if self.main_window.isMaximized():
            self.main_window.showNormal()
            self.maximize_btn.setText("□")
        else:
            self.main_window.showMaximized()
            self.maximize_btn.setText("❐")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start = event.globalPos() - self.main_window.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton and not self.main_window.isMaximized():
            self.main_window.move(event.globalPos() - self.drag_start)
            event.accept()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.toggle_maximized()
            event.accept()


def resource_path(filename: str) -> Path:
    """Return path to a bundled resource in source or PyInstaller onedir mode."""
    if getattr(sys, 'frozen', False):
        return Path(sys.executable).resolve().parent / filename
    return Path(__file__).resolve().parent / filename


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
        self.initial_visibility_applied = False
        self.log_cleanup_timer = QTimer(self)
        self.log_cleanup_timer.timeout.connect(self.logger.cleanup_old_logs)
        self.log_cleanup_timer.start(3600000)
        self.init_ui()
        if self.config.auto_run_detection:
            QTimer.singleShot(1500, self.start_enabled_cameras)

    def init_ui(self):
        self.setWindowTitle(self.config.window_title)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setGeometry(100, 100, self.config.window_width, self.config.window_height)
        self.setObjectName("mainWindow")
        self.central_container = QWidget()
        self.central_container.setObjectName("centralContainer")
        central_layout = QVBoxLayout(self.central_container)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.title_bar = TitleBar(self)
        central_layout.addWidget(self.title_bar)
        self.camera_tabs = QTabWidget()
        self.camera_tabs.setObjectName("cameraTabs")
        central_layout.addWidget(self.camera_tabs, 1)
        self.setCentralWidget(self.central_container)
        for index, camera in enumerate(self.config.cameras[:4]):
            widget = CameraWidget(index, self.config_manager, self.logger, self.restart_application, self)
            self.camera_widgets.append(widget)
            self.camera_tabs.addTab(widget, camera.name or f"Cam{index + 1}")
        self.create_menu_bar()

    def create_menu_bar(self):
        """Create the main menu bar."""
        help_menu = self.menuBar().addMenu("도움말")
        license_action = help_menu.addAction("라이선스 / 오픈소스 고지")
        license_action.triggered.connect(self.show_license_dialog)

    def read_notice_file(self, filename: str) -> str:
        """Read a bundled notice file, returning a user-facing error if it is missing."""
        path = resource_path(filename)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return f"{filename} 파일을 찾을 수 없습니다: {path}"
        except OSError as exc:
            return f"{filename} 파일을 읽을 수 없습니다: {exc}"

    def show_license_dialog(self):
        """Show project license and third-party notices."""
        license_text = self.read_notice_file("LICENSE")
        third_party_text = self.read_notice_file("THIRD_PARTY_NOTICES.md")
        message = (
            "VideoTriggerDetection 프로젝트 라이선스: GPL-3.0\n"
            "LICENSE 파일에는 GNU General Public License v3.0 전문이 포함되어 있어야 합니다.\n"
            "현재 배포 폴더에 포함된 LICENSE 파일 내용을 아래에 표시합니다.\n"
            "전체 라이선스 전문은 배포 폴더에 포함된 LICENSE 파일을 참조하세요.\n\n"
            "===== LICENSE =====\n"
            f"{license_text}\n\n"
            "===== THIRD_PARTY_NOTICES.md =====\n"
            f"{third_party_text}"
        )

        dialog = QDialog(self)
        dialog.setWindowTitle("라이선스 / 오픈소스 고지")
        dialog.resize(800, 600)
        layout = QVBoxLayout(dialog)
        text_edit = QTextEdit(dialog)
        text_edit.setReadOnly(True)
        text_edit.setPlainText(message)
        layout.addWidget(text_edit)
        button_box = QDialogButtonBox(QDialogButtonBox.Close, dialog)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)
        dialog.exec_()

    def apply_initial_visibility(self):
        """프로그램 시작 직후 한 번만 최소화/숨김 옵션을 적용한다."""
        if self.initial_visibility_applied:
            return
        self.initial_visibility_applied = True
        if not self.config.start_minimized:
            self.show()
            return
        if self.config.show_tray_icon:
            self.hide()
            self.logger.info("시작 최소화 옵션 적용: 트레이로 숨김")
        else:
            self.showMinimized()
            self.logger.info("시작 최소화 옵션 적용: 작업 표시줄로 최소화")

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
        self.tray_icon.setToolTip(main_window.config.window_title)

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


def resource_path(relative_path: str) -> Path:
    """Return a resource path that works in source and PyInstaller bundles."""
    base_path = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return base_path / relative_path


def apply_app_theme(app: QApplication):
    """Apply a cohesive modern dark theme to the whole desktop UI."""
    checkmark_icon = resource_path("assets/checkmark-indicator.svg").as_posix()
    stylesheet = """
        QWidget {
            background: #0f172a;
            color: #e5e7eb;
            font-family: "Segoe UI", "Noto Sans CJK KR", "Malgun Gothic", sans-serif;
            font-size: 10.5pt;
        }

        QMainWindow#mainWindow, QWidget#centralContainer {
            background: #0b1120;
        }

        QWidget#titleBar {
            background: #1e293b;
            border-bottom: 1px solid #38bdf8;
        }

        QLabel#titleLabel {
            color: #f8fafc;
            font-size: 10pt;
            font-weight: 800;
            padding-left: 2px;
        }

        QLabel#titleIcon {
            background: transparent;
        }

        QPushButton#titleButton, QPushButton#titleCloseButton {
            background: transparent;
            border: none;
            border-radius: 6px;
            color: #cbd5e1;
            font-size: 13pt;
            font-weight: 800;
            padding: 0;
        }

        QPushButton#titleButton:hover {
            background: #334155;
            color: #ffffff;
        }

        QPushButton#titleCloseButton:hover {
            background: #ef4444;
            color: #ffffff;
        }

        QMenuBar {
            background: #0b1120;
            color: #cbd5e1;
            padding: 4px;
            border-bottom: 1px solid #1e293b;
        }

        QMenuBar::item:selected, QMenu {
            background: #1e293b;
            color: #f8fafc;
        }

        QTabWidget::pane {
            border: 1px solid #1e293b;
            border-radius: 14px;
            background: #111827;
            top: -1px;
        }

        QTabBar::tab {
            background: #111827;
            color: #94a3b8;
            border: 1px solid #1e293b;
            padding: 9px 16px;
            margin-right: 4px;
            border-top-left-radius: 10px;
            border-top-right-radius: 10px;
        }

        QTabBar::tab:selected {
            background: #1e293b;
            color: #f8fafc;
            border-bottom-color: #38bdf8;
        }

        QTabBar::tab:hover {
            color: #e0f2fe;
            background: #172033;
        }

        QSplitter::handle {
            background: #1e293b;
            margin: 8px 3px;
            border-radius: 3px;
        }

        QGroupBox, QFrame#previewCard, QFrame#toolbarCard {
            background: #111827;
            border: 1px solid #243244;
            border-radius: 16px;
            margin-top: 14px;
            padding: 14px;
        }

        QGroupBox::title {
            subcontrol-origin: margin;
            left: 14px;
            padding: 0 8px;
            color: #7dd3fc;
            font-weight: 700;
        }

        QLabel[role="sectionTitle"] {
            color: #f8fafc;
            font-size: 14pt;
            font-weight: 800;
        }

        QLabel[role="hint"] {
            color: #94a3b8;
            font-size: 9.5pt;
        }

        QLabel[role="metric"] {
            background: #172033;
            border: 1px solid #28364a;
            border-radius: 12px;
            padding: 10px 12px;
            color: #dbeafe;
            font-weight: 650;
        }

        QLabel[role="metric"][state="ok"] {
            color: #bbf7d0;
            border-color: #166534;
            background: #10251d;
        }

        QLabel[role="metric"][state="warning"] {
            color: #fef08a;
            border-color: #854d0e;
            background: #2a2110;
        }

        QLabel[role="metric"][state="error"] {
            color: #fecaca;
            border-color: #991b1b;
            background: #2a1114;
        }

        QLabel[role="metric"][state="idle"] {
            color: #cbd5e1;
            border-color: #334155;
        }

        QPushButton {
            background: #1e293b;
            color: #e5e7eb;
            border: 1px solid #334155;
            border-radius: 11px;
            padding: 9px 13px;
            font-weight: 700;
        }

        QPushButton:hover {
            background: #27364b;
            border-color: #38bdf8;
        }

        QPushButton:disabled {
            background: #111827;
            color: #64748b;
            border-color: #1e293b;
        }

        QPushButton[variant="primary"] {
            background: #0284c7;
            border-color: #0ea5e9;
            color: white;
        }

        QPushButton[variant="success"] {
            background: #16a34a;
            border-color: #22c55e;
            color: white;
        }

        QPushButton[variant="danger"], QPushButton:checked {
            background: #dc2626;
            border-color: #ef4444;
            color: white;
        }

        QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
            background: #0b1120;
            color: #e5e7eb;
            border: 1px solid #334155;
            border-radius: 10px;
            padding: 7px 9px;
            selection-background-color: #0ea5e9;
        }

        QLineEdit:focus, QTextEdit:focus, QPlainTextEdit:focus,
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border-color: #38bdf8;
        }

        QSpinBox, QDoubleSpinBox {
            padding-right: 34px;
        }

        QSpinBox::up-button, QDoubleSpinBox::up-button,
        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-origin: border;
            width: 28px;
            background: #334155;
            border-left: 1px solid #64748b;
        }

        QSpinBox::up-button, QDoubleSpinBox::up-button {
            subcontrol-position: top right;
            border-top-right-radius: 9px;
            border-bottom: 1px solid #64748b;
        }

        QSpinBox::down-button, QDoubleSpinBox::down-button {
            subcontrol-position: bottom right;
            border-bottom-right-radius: 9px;
        }

        QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
        QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
            background: #0ea5e9;
            border-left-color: #7dd3fc;
        }

        QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed,
        QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
            background: #0369a1;
        }

        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-bottom: 7px solid #f8fafc;
        }

        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
            width: 0;
            height: 0;
            border-left: 5px solid transparent;
            border-right: 5px solid transparent;
            border-top: 7px solid #f8fafc;
        }

        QCheckBox {
            spacing: 10px;
            color: #dbeafe;
            min-height: 24px;
        }

        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 5px;
            border: 2px solid #94a3b8;
            background: #f8fafc;
        }

        QCheckBox::indicator:hover {
            border-color: #38bdf8;
            background: #e0f2fe;
        }

        QCheckBox::indicator:checked {
            border-color: #fde047;
            background: #facc15;
            image: url("CHECKMARK_ICON");
        }

        QCheckBox::indicator:checked:hover {
            border-color: #fef08a;
            background: #fde047;
        }

        QCheckBox::indicator:disabled {
            border-color: #475569;
            background: #1e293b;
        }

        QScrollArea, QScrollArea > QWidget > QWidget {
            background: transparent;
            border: none;
        }

        QScrollBar:vertical {
            background: #0b1120;
            width: 11px;
            margin: 2px;
            border-radius: 5px;
        }

        QScrollBar::handle:vertical {
            background: #334155;
            border-radius: 5px;
            min-height: 32px;
        }

        QScrollBar::handle:vertical:hover {
            background: #475569;
        }

        QFrame#previewCard {
            padding: 10px;
        }

        QLabel#mainImageLabel {
            background: #020617;
            border: 1px solid #1e293b;
            border-radius: 14px;
        }

        QLabel#thumbnailLabel {
            background: #020617;
            border: 2px solid #243244;
            border-radius: 12px;
            padding: 4px;
        }

        QLabel#thumbnailLabel:hover {
            border-color: #38bdf8;
        }

        QLabel#thumbnailLabel[selected="true"] {
            border-color: #facc15;
            background: #2a2110;
        }
    """
    app.setStyleSheet(stylesheet.replace("CHECKMARK_ICON", checkmark_icon))


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    apply_app_theme(app)
    icon_path = Path("icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    main_window = MainWindow()
    tray_icon = TrayIcon(main_window)
    QTimer.singleShot(0, main_window.apply_initial_visibility)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
