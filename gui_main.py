"""Main application entry point and top-level window for VideoTriggerDetection."""

import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "16")

from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QDialog, QDialogButtonBox, QMainWindow, QTabWidget, QTextEdit, QVBoxLayout

from camera_widget import CameraWidget
from config_manager import ConfigManager, CameraConfig
from logger import Logger


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


def main():
    """메인 함수"""
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    icon_path = Path("icon.ico")
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    main_window = MainWindow()
    tray_icon = TrayIcon(main_window)
    QTimer.singleShot(0, main_window.apply_initial_visibility)
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
