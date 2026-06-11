"""Windows 자동 실행 등록 관리.1"""
import os
import platform
import subprocess
import sys
try:
    import winreg
except ImportError:
    winreg = None
from pathlib import Path

TASK_NAME = "VideoTriggerDetection"
RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def get_executable_command() -> str:
    """현재 실행 형태에 맞는 자동실행 명령 문자열 반환."""
    if getattr(sys, 'frozen', False):
        return f'"{sys.executable}"'
    script = Path(__file__).with_name('gui_main.py')
    return f'"{sys.executable}" "{script}"'


def is_windows() -> bool:
    return platform.system().lower() == 'windows'


def enable_task_scheduler() -> tuple[bool, str]:
    """작업 스케줄러에 사용자 로그인 시 실행 작업 등록."""
    if not is_windows():
        return False, "Windows 환경이 아니므로 작업 스케줄러 등록을 건너뜁니다."

    cmd = get_executable_command()
    args = [
        'schtasks', '/Create',
        '/TN', TASK_NAME,
        '/TR', cmd,
        '/SC', 'ONLOGON',
        '/RL', 'HIGHEST',
        '/F'
    ]
    try:
        result = subprocess.run(args, capture_output=True, text=True, shell=False)
        if result.returncode == 0:
            return True, "작업 스케줄러 자동 실행 등록 완료"
        return False, (result.stderr or result.stdout or "작업 스케줄러 등록 실패").strip()
    except Exception as exc:
        return False, f"작업 스케줄러 등록 오류: {exc}"


def disable_task_scheduler() -> tuple[bool, str]:
    """작업 스케줄러 자동 실행 작업 삭제."""
    if not is_windows():
        return False, "Windows 환경이 아니므로 작업 스케줄러 삭제를 건너뜁니다."

    try:
        result = subprocess.run(
            ['schtasks', '/Delete', '/TN', TASK_NAME, '/F'],
            capture_output=True,
            text=True,
            shell=False
        )
        if result.returncode == 0:
            return True, "작업 스케줄러 자동 실행 삭제 완료"
        return False, (result.stderr or result.stdout or "작업 스케줄러 삭제 실패").strip()
    except Exception as exc:
        return False, f"작업 스케줄러 삭제 오류: {exc}"


def enable_registry_run() -> tuple[bool, str]:
    """HKCU Run 레지스트리에 사용자 로그인 후 자동 실행 등록."""
    if not is_windows():
        return False, "Windows 환경이 아니므로 레지스트리 등록을 건너뜁니다."

    if winreg is None:
        return False, "winreg 모듈을 사용할 수 없습니다."

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.SetValueEx(key, TASK_NAME, 0, winreg.REG_SZ, get_executable_command())
        return True, "레지스트리 자동 실행 등록 완료"
    except Exception as exc:
        return False, f"레지스트리 등록 오류: {exc}"


def disable_registry_run() -> tuple[bool, str]:
    """HKCU Run 자동 실행 값 삭제."""
    if not is_windows():
        return False, "Windows 환경이 아니므로 레지스트리 삭제를 건너뜁니다."

    if winreg is None:
        return False, "winreg 모듈을 사용할 수 없습니다."

    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, TASK_NAME)
        return True, "레지스트리 자동 실행 삭제 완료"
    except FileNotFoundError:
        return True, "레지스트리 자동 실행 값이 이미 없습니다."
    except Exception as exc:
        return False, f"레지스트리 삭제 오류: {exc}"


def apply_startup(enabled: bool, method: str = "task_scheduler") -> tuple[bool, str]:
    """설정값에 따라 로그인 후 자동 실행 등록/삭제."""
    method = (method or "task_scheduler").lower()
    if enabled:
        if method == "registry":
            return enable_registry_run()
        ok, msg = enable_task_scheduler()
        if ok:
            return ok, msg
        # 작업 스케줄러가 권한/정책 때문에 실패하면 registry 방식으로 보조 등록 시도
        reg_ok, reg_msg = enable_registry_run()
        return reg_ok, f"{msg} / Registry fallback: {reg_msg}"

    task_ok, task_msg = disable_task_scheduler()
    reg_ok, reg_msg = disable_registry_run()
    return task_ok or reg_ok, f"{task_msg} / {reg_msg}"
