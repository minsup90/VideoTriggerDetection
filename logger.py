"""
Logger Module
날짜별 로그 파일 관리 및 자동 삭제 기능
"""
import os
import logging
import logging.handlers
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional


class Logger:
    """로그 관리 클래스"""

    def __init__(
        self,
        log_dir: str = "logs",
        log_level: str = "INFO",
        retention_days: int = 30,
        max_file_size_mb: int = 10
    ):
        self.log_dir = Path(log_dir)
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.retention_days = retention_days
        self.max_file_size_mb = max_file_size_mb
        self.logger: Optional[logging.Logger] = None

        # 로그 디렉토리 생성
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 오래된 로그 파일 삭제
        self._cleanup_old_logs()

        # 로거 초기화
        self._setup_logger()

    def _cleanup_old_logs(self):
        """보관 기간이 지난 로그 파일 삭제"""
        if not self.log_dir.exists():
            return

        cutoff_date = datetime.now() - timedelta(days=self.retention_days)

        for log_file in self.log_dir.glob("*.log"):
            try:
                # 파일 수정 시간 확인
                file_mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                if file_mtime < cutoff_date:
                    log_file.unlink()
                    print(f"오래된 로그 파일 삭제: {log_file}")
            except Exception as e:
                print(f"로그 파일 삭제 중 오류: {e}")

    def _setup_logger(self):
        """로거 설정"""
        # 오늘 날짜의 로그 파일명
        today = datetime.now().strftime("%Y%m%d")
        log_file = self.log_dir / f"app_{today}.log"

        # 로거 생성
        self.logger = logging.getLogger("VideoTriggerDetection")
        self.logger.setLevel(self.log_level)

        # 기존 핸들러 제거
        self.logger.handlers.clear()

        # 파일 핸들러 (날짜별 로그 파일)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=self.max_file_size_mb * 1024 * 1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(self.log_level)

        # 콘솔 핸들러
        console_handler = logging.StreamHandler()
        console_handler.setLevel(self.log_level)

        # 포맷터
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        # 핸들러 추가
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def debug(self, message: str):
        """DEBUG 레벨 로그"""
        if self.logger:
            self.logger.debug(message)

    def info(self, message: str):
        """INFO 레벨 로그"""
        if self.logger:
            self.logger.info(message)

    def warning(self, message: str):
        """WARNING 레벨 로그"""
        if self.logger:
            self.logger.warning(message)

    def error(self, message: str, exc_info: bool = False):
        """ERROR 레벨 로그"""
        if self.logger:
            self.logger.error(message, exc_info=exc_info)

    def critical(self, message: str, exc_info: bool = False):
        """CRITICAL 레벨 로그"""
        if self.logger:
            self.logger.critical(message, exc_info=exc_info)

    def log_rtsp_status(self, connected: bool, fps: Optional[float] = None):
        """RTSP 연결 상태 로그"""
        status = "연결됨" if connected else "연결 끊김"
        fps_info = f", FPS: {fps:.2f}" if fps else ""
        self.info(f"RTSP 상태: {status}{fps_info}")

    def log_pattern_match(self, pattern_index: int, score: float, threshold: float, matched: bool):
        """패턴 매칭 결과 로그"""
        status = "매칭 성공" if matched else "매칭 실패"
        self.info(f"패턴 {pattern_index}: Score={score:.4f}, Threshold={threshold:.4f}, {status}")

    def log_image_saved(self, filepath: str, tenengrade_score: float):
        """이미지 저장 로그"""
        self.info(f"이미지 저장: {filepath}, Tenengrade Score: {tenengrade_score:.2f}")

    def log_ftp_upload(self, filepath: str, success: bool, error: Optional[str] = None):
        """FTP 업로드 로그"""
        if success:
            self.info(f"FTP 업로드 성공: {filepath}")
        else:
            self.error(f"FTP 업로드 실패: {filepath}, 오류: {error}")

    def log_buffer_status(self, current_size: int, max_size: int):
        """버퍼 상태 로그"""
        self.debug(f"버퍼 상태: {current_size}/{max_size}")

    def log_template_registration(self, pattern_index: int, template_path: str, roi: tuple):
        """템플릿 등록 로그"""
        self.info(f"템플릿 등록: Index={pattern_index}, Path={template_path}, ROI={roi}")

    def log_image_gathering(self, duration: int, frame_count: int):
        """이미지 수집 로그"""
        self.info(f"이미지 수집 완료: Duration={duration}s, Frames={frame_count}")


# 전역 로거 인스턴스
_global_logger: Optional[Logger] = None


def get_logger(
    log_dir: str = "logs",
    log_level: str = "INFO",
    retention_days: int = 30,
    max_file_size_mb: int = 10
) -> Logger:
    """전역 로거 인스턴스 반환"""
    global _global_logger
    if _global_logger is None:
        _global_logger = Logger(log_dir, log_level, retention_days, max_file_size_mb)
    return _global_logger


def init_logger(config):
    """설정에서 로거 초기화"""
    global _global_logger
    _global_logger = Logger(
        log_dir=config.log_dir,
        log_level=config.log_level,
        retention_days=config.log_retention_days,
        max_file_size_mb=config.log_max_file_size_mb
    )
    return _global_logger
