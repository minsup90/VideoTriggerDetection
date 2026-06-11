"""Main application entry point and top-level window for VideoTriggerDetection."""

import subprocess
import sys
import time
from pathlib import Path

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from camera_widget import CameraWidget
from config_manager import ConfigManager, CameraConfig
from logger import Logger


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
