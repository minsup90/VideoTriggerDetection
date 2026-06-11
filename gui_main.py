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

