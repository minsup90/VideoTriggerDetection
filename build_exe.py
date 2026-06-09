"""
Build Script for Windows EXE
PyInstaller를 사용하여 Windows 실행 파일 생성
"""
import PyInstaller.__main__
import os
import sys
from pathlib import Path


def build_exe():
    """Windows EXE 파일 빌드"""

    # 프로젝트 경로
    project_dir = Path(__file__).parent
    main_script = project_dir / "gui_main.py"

    # dist 디렉토리 설정
    dist_dir = project_dir / "dist"
    build_dir = project_dir / "build"

    # PyInstaller 옵션
    options = [
        str(main_script),  # 메인 스크립트
        '--name=VideoTriggerDetection',  # 실행 파일 이름
        '--onefile',  # 단일 파일로 패키징
        '--windowed',  # 콘솔 창 없이 실행 (GUI)
        '--manifest=app.manifest',
        '--icon=icon.ico',  # 아이콘 파일 (있는 경우)
        '--add-data=config.yaml;.',  # 설정 파일 포함
        # '--hidden-import=PyQt5',
        # '--hidden-import=PyQt5.QtCore',
        # '--hidden-import=PyQt5.QtGui',
        # '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=yaml',
        # '--collect-all=cv2',  # OpenCV 의존성 포함
        # '--collect-all=numpy',  # NumPy 의존성 포함
        '--clean',  # 빌드 캐시 정리
        '--noconfirm',  # 확인 없이 덮어쓰기
        '--exclude-module=PySide2',
        '--exclude-module=PySide6',
        '--exclude-module=PyQt6',
        f'--distpath={dist_dir}',
        f'--workpath={build_dir}',
        f'--specpath={project_dir}',
    ]

    print("=== Windows EXE 빌드 시작 ===")
    print(f"메인 스크립트: {main_script}")
    print(f"출력 디렉토리: {dist_dir}")
    print()

    # 빌드 실행
    PyInstaller.__main__.run(options)

    print()
    print("=== 빌드 완료 ===")
    print(f"실행 파일 위치: {dist_dir / 'VideoTriggerDetection.exe'}")
    print()
    print("참고:")
    print("- 실행 파일과 같은 디렉토리에 config.yaml 파일이 필요합니다.")
    print("- templates, saved_images, logs 디렉토리가 자동 생성됩니다.")
    print("- 첫 실행 시 필요한 디렉토리가 생성됩니다.")

def build_with_console():
    """콘솔 창이 있는 EXE 파일 빌드 (디버깅용)"""

    project_dir = Path(__file__).parent
    main_script = project_dir / "gui_main.py"

    options = [
        str(main_script),
        '--name=VideoTriggerDetection_Debug',
        '--onedir',
        '--console',  # 콘솔 창 표시
        '--icon=icon.ico',
        '--add-data=config.yaml;.',
        '--hidden-import=PyQt5',
        '--hidden-import=PyQt5.QtCore',
        '--hidden-import=PyQt5.QtGui',
        '--hidden-import=PyQt5.QtWidgets',
        '--hidden-import=cv2',
        '--hidden-import=numpy',
        '--hidden-import=yaml',
        '--collect-all=cv2',
        '--collect-all=numpy',
        '--clean',
        '--noconfirm',
        '--manifest=app.manifest',
        '--exclude-module PySide2',
        '--exclude-module PySide6',
        '--exclude-module PyQt6',
    ]

    print("=== Debug EXE 빌드 시작 (콘솔 포함) ===")
    PyInstaller.__main__.run(options)
    print("=== 빌드 완료 ===")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--debug":
        build_with_console()
    else:
        build_exe()
