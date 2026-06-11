"""
Configuration Manager
YAML 기반 설정 파일을 관리하는 모듈
"""
import os
import copy
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
    def from_dict(cls, data: Optional[Dict[str, int]]) -> 'ROI':
        data = data or {}
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
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'PatternConfig':
        data = data or {}
        return cls(
            index=data.get('index', 1),
            template_path=data.get('template_path', ''),
            score_threshold=data.get('score_threshold', 0.85),
            roi=ROI.from_dict(data.get('roi', {}))
        )


@dataclass
class ImageSaveOptions:
    """이미지 저장 파일명/용량 옵션"""
    equipment_no: str = "EQ01"
    mount: str = "Mount"
    image_format: str = "BMP"  # BMP, JPG, PNG
    quality: int = 90  # JPG 품질 또는 PNG 압축률 UI 값(10~90)
    resize_enabled: bool = False
    resize_width: int = 0
    resize_height: int = 0
    keep_aspect_ratio: bool = True
    filename_format: str = "{equipment_no}.{mount}.{yyyyMMdd}_{hhmmss}_{seq:03d}.{ext}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            'equipment_no': self.equipment_no,
            'mount': self.mount,
            'image_format': self.image_format,
            'quality': self.quality,
            'resize_enabled': self.resize_enabled,
            'resize_width': self.resize_width,
            'resize_height': self.resize_height,
            'keep_aspect_ratio': self.keep_aspect_ratio,
            'filename_format': self.filename_format
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'ImageSaveOptions':
        data = data or {}
        return cls(
            equipment_no=data.get('equipment_no', 'EQ01'),
            mount=data.get('mount', data.get('mount_name', 'Mount')),
            image_format=str(data.get('image_format', 'BMP')).upper(),
            quality=max(10, min(90, int(data.get('quality', data.get('jpeg_quality', 90))))),
            resize_enabled=bool(data.get('resize_enabled', False)),
            resize_width=max(0, int(data.get('resize_width', 0) or 0)),
            resize_height=max(0, int(data.get('resize_height', 0) or 0)),
            keep_aspect_ratio=bool(data.get('keep_aspect_ratio', True)),
            filename_format=data.get(
                'filename_format',
                '{equipment_no}.{mount}.{yyyyMMdd}_{hhmmss}_{seq:03d}.{ext}'
            )
        )


@dataclass
class HealthCheckConfig:
    """영상 HealthCheck 및 재시작 정책"""
    enabled: bool = True
    timeout_sec: int = 10
    check_interval_sec: int = 1
    freeze_diff_threshold: float = 1.0
    restart_stream: bool = True
    restart_app: bool = True
    restart_limit_enabled: bool = True
    max_restart_per_hour: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            'enabled': self.enabled,
            'timeout_sec': self.timeout_sec,
            'check_interval_sec': self.check_interval_sec,
            'freeze_diff_threshold': self.freeze_diff_threshold,
            'restart_stream': self.restart_stream,
            'restart_app': self.restart_app,
            'restart_limit_enabled': self.restart_limit_enabled,
            'max_restart_per_hour': self.max_restart_per_hour
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'HealthCheckConfig':
        data = data or {}
        return cls(
            enabled=bool(data.get('enabled', True)),
            timeout_sec=max(1, int(data.get('timeout_sec', 10))),
            check_interval_sec=max(1, int(data.get('check_interval_sec', 1))),
            freeze_diff_threshold=float(data.get('freeze_diff_threshold', 1.0)),
            restart_stream=bool(data.get('restart_stream', True)),
            restart_app=bool(data.get('restart_app', True)),
            restart_limit_enabled=bool(data.get('restart_limit_enabled', True)),
            max_restart_per_hour=max(1, int(data.get('max_restart_per_hour', 5)))
        )


@dataclass
class CameraConfig:
    """카메라별 독립 설정"""
    id: str = "cam1"
    name: str = "Cam1"
    enabled: bool = True
    rtsp_url: str = ""
    rtsp_reconnect_interval: int = 5
    ai_model_path: str = ""
    ai_enabled: bool = False
    ai_threshold: float = 0.8
    ai_condition_type: str = "and"
    tenengrade_threshold: float = 100.0
    optical_flow_threshold: float = 0.5
    patterns: List[PatternConfig] = field(default_factory=list)
    require_all_patterns: bool = True
    frame_save_duration: int = 10
    max_frames: int = 300
    buffer_size: int = 10
    ftp_enabled: bool = True
    ftp_host: str = ""
    ftp_port: int = 21
    ftp_username: str = ""
    ftp_password: str = ""
    ftp_remote_dir: str = ""
    ftp_timeout: int = 30
    save_dir: str = "saved_images/Cam1"
    retention_days: int = 30
    image_save: ImageSaveOptions = field(default_factory=ImageSaveOptions)
    healthcheck: HealthCheckConfig = field(default_factory=HealthCheckConfig)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'enabled': self.enabled,
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
                **self.image_save.to_dict()
            },
            'healthcheck': self.healthcheck.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]], default_index: int = 1) -> 'CameraConfig':
        data = data or {}
        rtsp = data.get('rtsp', {})
        ai = data.get('ai_classification', {})
        quality = data.get('image_quality', {})
        tm = data.get('template_matching', {})
        gathering = data.get('image_gathering', {})
        buffer = data.get('buffer', {})
        ftp = data.get('ftp', {})
        storage = data.get('file_storage', {})
        name = data.get('name', f'Cam{default_index}')
        return cls(
            id=data.get('id', f'cam{default_index}'),
            name=name,
            enabled=bool(data.get('enabled', True)),
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
            save_dir=storage.get('save_dir', f'saved_images/{name}'),
            retention_days=storage.get('retention_days', 30),
            image_save=ImageSaveOptions.from_dict(storage),
            healthcheck=HealthCheckConfig.from_dict(data.get('healthcheck', {}))
        )


@dataclass
class Config:
    """전체 설정 클래스"""
    cameras: List[CameraConfig] = field(default_factory=list)

    # 시스템/자동실행 설정
    auto_start_enabled: bool = False
    auto_start_method: str = "task_scheduler"  # task_scheduler 또는 registry
    auto_run_detection: bool = False

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

    def __post_init__(self):
        if not self.cameras:
            self.cameras = self.default_cameras()
        self._sync_legacy_attrs()

    @staticmethod
    def default_cameras() -> List[CameraConfig]:
        cameras = []
        for i in range(1, 5):
            cameras.append(CameraConfig(
                id=f'cam{i}',
                name=f'Cam{i}',
                enabled=(i == 1),
                save_dir=f'saved_images/Cam{i}'
            ))
        return cameras

    def _sync_legacy_attrs(self):
        """기존 단일 카메라 코드와의 호환용 속성 유지"""
        cam = self.cameras[0] if self.cameras else CameraConfig()
        self.rtsp_url = cam.rtsp_url
        self.rtsp_reconnect_interval = cam.rtsp_reconnect_interval
        self.ai_model_path = cam.ai_model_path
        self.ai_enabled = cam.ai_enabled
        self.ai_threshold = cam.ai_threshold
        self.ai_condition_type = cam.ai_condition_type
        self.tenengrade_threshold = cam.tenengrade_threshold
        self.optical_flow_threshold = cam.optical_flow_threshold
        self.patterns = cam.patterns
        self.require_all_patterns = cam.require_all_patterns
        self.frame_save_duration = cam.frame_save_duration
        self.max_frames = cam.max_frames
        self.buffer_size = cam.buffer_size
        self.ftp_enabled = cam.ftp_enabled
        self.ftp_host = cam.ftp_host
        self.ftp_port = cam.ftp_port
        self.ftp_username = cam.ftp_username
        self.ftp_password = cam.ftp_password
        self.ftp_remote_dir = cam.ftp_remote_dir
        self.ftp_timeout = cam.ftp_timeout
        self.save_dir = cam.save_dir
        self.retention_days = cam.retention_days
        self.filename_format = cam.image_save.filename_format

    def to_dict(self) -> Dict[str, Any]:
        """설정을 딕셔너리로 변환"""
        self._sync_legacy_attrs()
        return {
            'system': {
                'auto_start_enabled': self.auto_start_enabled,
                'auto_start_method': self.auto_start_method,
                'auto_run_detection': self.auto_run_detection
            },
            'cameras': [camera.to_dict() for camera in self.cameras[:4]],
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
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> 'Config':
        """딕셔너리에서 설정 로드. 기존 단일 카메라 config도 Cam1로 자동 변환."""
        data = data or {}
        system = data.get('system', {})
        logging = data.get('logging', {})
        gui = data.get('gui', {})

        cameras_data = data.get('cameras')
        if cameras_data:
            cameras = [CameraConfig.from_dict(cam, i + 1) for i, cam in enumerate(cameras_data[:4])]
            while len(cameras) < 4:
                idx = len(cameras) + 1
                cameras.append(CameraConfig(id=f'cam{idx}', name=f'Cam{idx}', enabled=False, save_dir=f'saved_images/Cam{idx}'))
        else:
            # 기존 config.yaml 구조를 Cam1로 마이그레이션
            legacy_cam = CameraConfig.from_dict({
                'id': 'cam1',
                'name': 'Cam1',
                'rtsp': data.get('rtsp', {}),
                'ai_classification': data.get('ai_classification', {}),
                'image_quality': data.get('image_quality', {}),
                'template_matching': data.get('template_matching', {}),
                'image_gathering': data.get('image_gathering', {}),
                'buffer': data.get('buffer', {}),
                'ftp': data.get('ftp', {}),
                'file_storage': data.get('file_storage', {}),
                'healthcheck': data.get('healthcheck', {})
            }, 1)
            legacy_cam.save_dir = data.get('file_storage', {}).get('save_dir', 'saved_images/Cam1')
            cameras = [legacy_cam]
            for idx in range(2, 5):
                cam = copy.deepcopy(legacy_cam)
                cam.id = f'cam{idx}'
                cam.name = f'Cam{idx}'
                cam.enabled = False
                cam.rtsp_url = ''
                cam.patterns = []
                cam.save_dir = f'saved_images/Cam{idx}'
                cameras.append(cam)

        return cls(
            cameras=cameras,
            auto_start_enabled=bool(system.get('auto_start_enabled', False)),
            auto_start_method=system.get('auto_start_method', 'task_scheduler'),
            auto_run_detection=bool(system.get('auto_run_detection', False)),
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

    def save_config(self):
        """설정을 YAML 파일로 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config.to_dict(), f, allow_unicode=True, default_flow_style=False, sort_keys=False)
            return True
        except Exception as e:
            print(f"설정 파일 저장 실패: {e}")
            return False

    def get_config(self) -> Config:
        """설정 객체 반환"""
        return self.config

    def get_camera(self, index: int) -> CameraConfig:
        return self.config.cameras[index]

    def update_camera(self, index: int, camera: CameraConfig):
        self.config.cameras[index] = camera
        self.config._sync_legacy_attrs()
        self.save_config()

    def update_rtsp_url(self, url: str):
        """Cam1 RTSP URL 업데이트(기존 호환)"""
        self.config.cameras[0].rtsp_url = url
        self.config._sync_legacy_attrs()
        self.save_config()

    def update_pattern(self, index: int, pattern: PatternConfig, camera_index: int = 0):
        """패턴 설정 업데이트"""
        patterns = self.config.cameras[camera_index].patterns
        existing = None
        for i, p in enumerate(patterns):
            if p.index == index:
                existing = i
                break

        if existing is not None:
            patterns[existing] = pattern
        else:
            patterns.append(pattern)
            patterns.sort(key=lambda x: x.index)

        self.config._sync_legacy_attrs()
        self.save_config()

    def remove_pattern(self, index: int, camera_index: int = 0):
        """패턴 삭제"""
        camera = self.config.cameras[camera_index]
        camera.patterns = [p for p in camera.patterns if p.index != index]
        self.config._sync_legacy_attrs()
        self.save_config()

    def get_pattern(self, index: int, camera_index: int = 0) -> Optional[PatternConfig]:
        """특정 인덱스의 패턴 반환"""
        for p in self.config.cameras[camera_index].patterns:
            if p.index == index:
                return p
        return None

    def get_all_patterns(self, camera_index: int = 0) -> List[PatternConfig]:
        """모든 패턴 반환"""
        return self.config.cameras[camera_index].patterns.copy()

    def ensure_directories(self):
        """필요한 디렉토리 생성"""
        directories = [self.config.log_dir, "templates", "gathered_images"]
        for camera in self.config.cameras:
            directories.extend([
                camera.save_dir,
                str(Path("templates") / camera.name),
                str(Path("gathered_images") / camera.name),
            ])
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
