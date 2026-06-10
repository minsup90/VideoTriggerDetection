# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
from PyInstaller.utils.hooks import collect_all


datas = [('config.yaml', '.')]
binaries = []
icon_file = Path('icon.ico')
icon_arg = ['icon.ico'] if icon_file.exists() else None
if icon_file.exists():
    datas.append(('icon.ico', '.'))

hiddenimports = [
    'cv2',
    'numpy',
    'yaml',
    'PyQt5',
    'PyQt5.QtCore',
    'PyQt5.QtGui',
    'PyQt5.QtWidgets',
]

# OpenCV/NumPy/PyQt5는 DLL/플러그인 누락 또는 다른 DLL과의 버전 충돌이 잦으므로
# 패키지의 데이터/바이너리/숨은 import를 명시적으로 수집한다.
for package_name in ('cv2', 'numpy', 'PyQt5'):
    package_datas, package_binaries, package_hiddenimports = collect_all(package_name)
    datas += package_datas
    binaries += package_binaries
    hiddenimports += package_hiddenimports


a = Analysis(
    ['gui_main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PySide2', 'PySide6', 'PyQt6'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='VideoTriggerDetection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_arg,
    manifest='app.manifest',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='VideoTriggerDetection',
)
