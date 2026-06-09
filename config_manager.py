"""
Configuration Manager
YAML 기반 설정 파일을 관리하는 모듈
"""
import os
import yaml
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ROI:
    """ROI 영역 정보"""
    x: int = 0
    y: int = 0
    width: int = 0
    height: int = 0

    def to_dict(self) -> Dict[str, int]:
        return {
            'x': self.x,
            'y': self.y,
            'width': self.width,
            'height': self.height
        }

    @classmethod
    def from_dict(cls, data: Dict[str, int]) -> 'ROI':
        return cls(
            x=data.get('x', 0),
            y=data.get('y', 0),
            width=data.get('width', 0),
            height=data.get('height', 0)
        )


@dataclass
class PatternConfig:
    """Template Matching 패턴 설정"""
    index: int = 1
    template_path: str = ""
    score_threshold: float = 0.85
    roi: ROI = field(default_factory=ROI)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'index': self.index,
            'template_path': self.template_path,
            'score_threshold': self.score_threshold,
            'roi': self.roi.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternConfig':
        return cls(
            index=data.get('index', 1),
            template_path=data.get('template_path', ''),
            score_threshold=data.get('score_threshold', 0.85),
            roi=ROI.from_dict(data.get('roi', {}))
        )


@dataclass
class Config:
    """전체 설정 클래스"""
    # RTSP 설정
    rtsp_url: str = ""
    rtsp_reconnect_interval: int = 5

    # AI Classification 설정
    ai_model_path: str = ""
    ai_enabled: bool = False
    ai_threshold: float = 0.8
    ai_condition_type: str = "and"  # 'and' 또는 'or'

    # 이미지 품질 설정
    tenengrade_threshold: float = 100.0
    optical_flow_threshold: float = 0.5

    # Template Matching 설정
    patterns: List[PatternConfig] = field(default_factory=list)
    require_all_patterns: bool = True

    # 이미지 수집 설정
    frame_save_duration: int = 10
    max_frames: int = 300

    # 버퍼 설정
    buffer_size: int = 10

    # FTP 설정
    ftp_enabled: bool = True
    ftp_host: str = ""
    ftp_port: int = 21
    ftp_username: str = ""
    ftp_password: str = ""
    ftp_remote_dir: str = ""
    ftp_timeout: int = 30

    # 파일 저장 설정
    save_dir: str = "saved_images"
    retention_days: int = 30
    filename_format: str = "yyyyMMdd_hhmmss_000.bmp"

    # 로그 설정
    log_dir: str = "logs"
    log_level: str = "INFO"
    log_retention_days: int = 30
    log_max_file_size_mb: int = 10

    # GUI 설정
    window_title: str = "Video Trigger Detection"
    window_width: int = 1280
    window_height: int = 720
    show_tray_icon: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        return {
            'rtsp': {
                'url': self.rtsp_url,
                'reconnect_interval': self.rtsp_reconnect_interval
            },
            'ai_classification': {
                'model_path': self.ai_model_path,
                'enabled': self.ai_enabled,
                'threshold': self.ai_threshold,
                'condition_type': self.ai_condition_type
            },
            'image_quality': {
                'tenengrade_threshold': self.tenengrade_threshold,
                'optical_flow_threshold': self.optical_flow_threshold
            },
            'template_matching': {
                'patterns': [p.to_dict() for p in self.patterns],
                'require_all_patterns': self.require_all_patterns
            },
            'image_gathering': {
                'frame_save_duration': self.frame_save_duration,
                'max_frames': self.max_frames
            },
            'buffer': {
                'size': self.buffer_size
            },
            'ftp': {
                'enabled': self.ftp_enabled,
                'host': self.ftp_host,
                'port': self.ftp_port,
                'username': self.ftp_username,
                'password': self.ftp_password,
                'remote_dir': self.ftp_remote_dir,
                'timeout': self.ftp_timeout
            },
            'file_storage': {
                'save_dir': self.save_dir,
                'retention_days': self.retention_days,
                'filename_format': self.filename_format
            },
            'logging': {
                'log_dir': self.log_dir,
                'log_level': self.log_level,
                'retention_days': self.log_retention_days,
                'max_file_size_mb': self.log_max_file_size_mb
            },
            'gui': {
                'window_title': self.window_title,
                'window_width': self.window_width,
                'window_height': self.window_height,
                'show_tray_icon': self.show_tray_icon
            }
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Config':
        """딕셔너리에서 설정 로드"""
        rtsp = data.get('rtsp', {})
        ai = data.get('ai_classification', {})
        quality = data.get('image_quality', {})
        tm = data.get('template_matching', {})
        gathering = data.get('image_gathering', {})
        buffer = data.get('buffer', {})
        ftp = data.get('ftp', {})
        storage = data.get('file_storage', {})
        logging = data.get('logging', {})
        gui = data.get('gui', {})

        return cls(
            rtsp_url=rtsp.get('url', ''),
            rtsp_reconnect_interval=rtsp.get('reconnect_interval', 5),
            ai_model_path=ai.get('model_path', ''),
            ai_enabled=ai.get('enabled', False),
            ai_threshold=ai.get('threshold', 0.8),
            ai_condition_type=ai.get('condition_type', 'and'),
            tenengrade_threshold=quality.get('tenengrade_threshold', 100.0),
            optical_flow_threshold=quality.get('optical_flow_threshold', 0.5),
            patterns=[PatternConfig.from_dict(p) for p in tm.get('patterns', [])],
            require_all_patterns=tm.get('require_all_patterns', True),
            frame_save_duration=gathering.get('frame_save_duration', 10),
            max_frames=gathering.get('max_frames', 300),
            buffer_size=buffer.get('size', 10),
            ftp_enabled=ftp.get('enabled', True),
            ftp_host=ftp.get('host', ''),
            ftp_port=ftp.get('port', 21),
            ftp_username=ftp.get('username', ''),
            ftp_password=ftp.get('password', ''),
            ftp_remote_dir=ftp.get('remote_dir', ''),
            ftp_timeout=ftp.get('timeout', 30),
            save_dir=storage.get('save_dir', 'saved_images'),
            retention_days=storage.get('retention_days', 30),
            filename_format=storage.get('filename_format', 'yyyyMMdd_hhmmss_000.bmp'),
            log_dir=logging.get('log_dir', 'logs'),
            log_level=logging.get('log_level', 'INFO'),
            log_retention_days=logging.get('retention_days', 30),
            log_max_file_size_mb=logging.get('max_file_size_mb', 10),
            window_title=gui.get('window_title', 'Video Trigger Detection'),
            window_width=gui.get('window_width', 1280),
            window_height=gui.get('window_height', 720),
            show_tray_icon=gui.get('show_tray_icon', True)
        )


class ConfigManager:
    """설정 관리자"""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = Config()
        self._load_config()

    def _load_config(self):
        """YAML 설정 파일 로드"""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data:
                        self.config = Config.from_dict(data)
            except Exception as e:
                print(f"설정 파일 로드 실패: {e}")
                # 기본 설정 사용

    def save_config(self):
        """설정을 YAML 파일로 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config.to_dict(), f, allow_unicode=True, default_flow_style=False)
            return True
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")
            return False

    def get_config(self) -> Config:
        """설정 객체 반환"""
        return self.config

    def update_rtsp_url(self, url: str):
        """RTSP URL 업데이트"""
        self.config.rtsp_url = url
        self.save_config()

    def update_pattern(self, index: int, pattern: PatternConfig):
        """패턴 설정 업데이트"""
        # 기존 패턴 찾기
        existing = None
        for i, p in enumerate(self.config.patterns):
            if p.index == index:
                existing = i
                break

        if existing is not None:
            self.config.patterns[existing] = pattern
        else:
            self.config.patterns.append(pattern)
            
            # 인덱스 순으로 정렬
            self.config.patterns.sort(key=lambda x: x.index)

        self.save_config()

    def remove_pattern(self, index: int):
        """패턴 삭제"""
        self.config.patterns = [p for p in self.config.patterns if p.index != index]
        self.save_config()

    def get_pattern(self, index: int) -> Optional[PatternConfig]:
        """특정 인덱스의 패턴 반환"""
        for p in self.config.patterns:
            if p.index == index:
                return p
        return None

    def get_all_patterns(self) -> List[PatternConfig]:
        """모든 패턴 반환"""
        return self.config.patterns.copy()

    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [
            self.config.save_dir,
            self.config.log_dir,
            "templates",
            "gathered_images"
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
