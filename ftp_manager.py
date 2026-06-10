"""
FTP Manager Module
FTP 서버로 이미지 전송 기능
"""
import os
import ftplib
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import threading


class FTPManager:
    """FTP 전송 관리 클래스"""

    def __init__(
        self,
        host: str,
        port: int = 21,
        username: str = "",
        password: str = "",
        remote_dir: str = "/",
        timeout: int = 30,
        enabled: bool = True
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.remote_dir = remote_dir
        self.timeout = timeout
        self.enabled = enabled
        self.ftp: Optional[ftplib.FTP] = None
        self.lock = threading.Lock()
        self.upload_callback: Optional[Callable[[str, bool, Optional[str]], None]] = None

    def set_upload_callback(self, callback: Callable[[str, bool, Optional[str]], None]):
        """업로드 콜백 설정"""
        self.upload_callback = callback

    def connect(self) -> bool:
        """FTP 서버 연결"""
        if not self.enabled:
            return False

        try:
            with self.lock:
                self.ftp = ftplib.FTP()
                self.ftp.connect(self.host, self.port, timeout=self.timeout)
                self.ftp.login(self.username, self.password)
                self.ftp.encoding = 'utf-8'

                # 원격 디렉토리 생성 및 이동
                self._ensure_remote_directory(self.remote_dir)

                return True

        except Exception as e:
            print(f"FTP 연결 실패: {e}")
            self.ftp = None
            return False

    def disconnect(self):
        """FTP 연결 해제"""
        with self.lock:
            if self.ftp:
                try:
                    self.ftp.quit()
                except:
                    try:
                        self.ftp.close()
                    except:
                        pass
                self.ftp = None

    def _ensure_remote_directory(self, dir_path: str):
        """원격 디렉토리 생성"""
        if not self.ftp:
            return

        try:
            # 디렉토리 경로를 분리
            parts = [p for p in dir_path.split('/') if p]
            current_path = ""

            for part in parts:
                current_path = f"{current_path}/{part}" if current_path else part
                try:
                    self.ftp.cwd(current_path)
                except ftplib.error_perm:
                    # 디렉토리가 없으면 생성
                    try:
                        self.ftp.mkd(current_path)
                        self.ftp.cwd(current_path)
                    except ftplib.error_perm as e:
                        print(f"원격 디렉토리 생성 실패: {current_path}, {e}")

        except Exception as e:
            print(f"원격 디렉토리 설정 오류: {e}")

    def upload_file(self, local_path: str, remote_subdir: Optional[str] = None) -> bool:
        """
        파일 업로드
        Args:
            local_path: 로컬 파일 경로
            remote_subdir: 원격 서브 디렉토리 (선택사항)
        Returns:
            성공 여부
        """
        if not self.enabled:
            return False

        if not os.path.exists(local_path):
            error_msg = f"파일이 존재하지 않음: {local_path}"
            if self.upload_callback:
                self.upload_callback(local_path, False, error_msg)
            return False

        # 연결 확인 및 재연결
        if not self.ftp:
            if not self.connect():
                error_msg = "FTP 연결 실패"
                if self.upload_callback:
                    self.upload_callback(local_path, False, error_msg)
                return False

        try:
            with self.lock:
                # 원격 디렉토리 설정
                if remote_subdir:
                    self._ensure_remote_directory(f"{self.remote_dir}/{remote_subdir}")
                else:
                    self.ftp.cwd(self.remote_dir)

                # 파일명 추출
                filename = os.path.basename(local_path)

                # 바이너리 모드로 업로드
                with open(local_path, 'rb') as f:
                    self.ftp.storbinary(f'STOR {filename}', f)

                if self.upload_callback:
                    self.upload_callback(local_path, True, None)

                return True

        except Exception as e:
            error_msg = f"FTP 업로드 오류: {str(e)}"
            print(error_msg)

            # 연결 재시도
            try:
                self.disconnect()
                self.connect()
            except:
                pass

            if self.upload_callback:
                self.upload_callback(local_path, False, error_msg)

            return False

    def upload_file_async(self, local_path: str, remote_subdir: Optional[str] = None):
        """
        비동기 파일 업로드
        별도 스레드에서 업로드 수행
        """
        if not self.enabled:
            return

        thread = threading.Thread(
            target=self.upload_file,
            args=(local_path, remote_subdir),
            daemon=True
        )
        thread.start()

    def test_connection(self) -> tuple[bool, str]:
        """FTP 연결 테스트"""
        if not self.enabled:
            return False, "FTP가 비활성화됨"

        try:
            with self.lock:
                test_ftp = ftplib.FTP()
                test_ftp.connect(self.host, self.port, timeout=self.timeout)
                test_ftp.login(self.username, self.password)
                test_ftp.quit()
                return True, "연결 성공"
        except Exception as e:
            return False, f"연결 실패: {str(e)}"

    def is_enabled(self) -> bool:
        """FTP 활성화 상태 확인"""
        return self.enabled

    def set_enabled(self, enabled: bool):
        """FTP 활성화/비활성화 설정"""
        self.enabled = enabled
        if not enabled:
            self.disconnect()


class FileStorageManager:
    """파일 저장 관리 클래스"""

    def __init__(
        self,
        save_dir: str = "saved_images",
        retention_days: int = 30,
        equipment_no: str = "EQ01",
        mount: str = "Mount",
        image_format: str = "BMP",
        quality: int = 90,
        resize_enabled: bool = False,
        resize_width: int = 0,
        resize_height: int = 0,
        keep_aspect_ratio: bool = True,
        filename_format: str = "{equipment_no}.{mount}.{yyyyMMdd}_{hhmmss}_{seq:03d}.{ext}"
    ):
        self.save_dir = Path(save_dir)
        self.retention_days = retention_days
        self.equipment_no = equipment_no or "EQ01"
        self.mount = mount or "Mount"
        self.image_format = (image_format or "BMP").upper()
        self.quality = max(10, min(90, int(quality)))
        self.resize_enabled = resize_enabled
        self.resize_width = max(0, int(resize_width or 0))
        self.resize_height = max(0, int(resize_height or 0))
        self.keep_aspect_ratio = keep_aspect_ratio
        self.filename_format = filename_format or "{equipment_no}.{mount}.{yyyyMMdd}_{hhmmss}_{seq:03d}.{ext}"
        self.lock = threading.Lock()

        # 저장 디렉토리 생성
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def get_date_dir(self) -> Path:
        """오늘 날짜의 디렉토리 경로 반환"""
        today = datetime.now().strftime("%Y%m%d")
        date_dir = self.save_dir / today
        date_dir.mkdir(parents=True, exist_ok=True)
        return date_dir

    def _extension(self) -> str:
        if self.image_format == "JPG" or self.image_format == "JPEG":
            return "jpg"
        if self.image_format == "PNG":
            return "png"
        return "bmp"

    def _build_filename(self, date_dir: Path) -> str:
        """설비No.Mount.yyyyMMdd_hhmmss_000 형태의 파일명 생성"""
        now = datetime.now()
        ext = self._extension()
        context = {
            'equipment_no': self.equipment_no,
            'mount': self.mount,
            'yyyyMMdd': now.strftime('%Y%m%d'),
            'hhmmss': now.strftime('%H%M%S'),
            'ext': ext,
        }

        seq = 0
        while seq < 1000:
            context['seq'] = seq
            try:
                filename = self.filename_format.format(**context)
            except Exception:
                filename = f"{self.equipment_no}.{self.mount}.{context['yyyyMMdd']}_{context['hhmmss']}_{seq:03d}.{ext}"
            if not Path(filename).suffix:
                filename = f"{filename}.{ext}"
            if not (date_dir / filename).exists():
                return filename
            seq += 1
        return f"{self.equipment_no}.{self.mount}.{context['yyyyMMdd']}_{context['hhmmss']}_{now.microsecond:06d}.{ext}"

    def _resize_image(self, image):
        if not self.resize_enabled or (self.resize_width <= 0 and self.resize_height <= 0):
            return image

        import cv2
        h, w = image.shape[:2]
        target_w = self.resize_width if self.resize_width > 0 else w
        target_h = self.resize_height if self.resize_height > 0 else h

        if self.keep_aspect_ratio:
            if self.resize_width > 0 and self.resize_height > 0:
                scale = min(self.resize_width / w, self.resize_height / h)
            elif self.resize_width > 0:
                scale = self.resize_width / w
            else:
                scale = self.resize_height / h
            target_w = max(1, int(w * scale))
            target_h = max(1, int(h * scale))

        return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _imwrite_params(self) -> list:
        import cv2
        if self.image_format in ("JPG", "JPEG"):
            return [cv2.IMWRITE_JPEG_QUALITY, self.quality]
        if self.image_format == "PNG":
            # UI 입력 10~90을 PNG 압축 1~9로 매핑한다. 높을수록 더 압축한다.
            png_compression = max(1, min(9, round(self.quality / 10)))
            return [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
        return []

    def save_image(self, image, filename: str = None) -> Optional[str]:
        """
        이미지 저장
        Args:
            image: OpenCV 이미지
            filename: 파일명 (None이면 자동 생성)
        Returns:
            저장된 파일 경로
        """
        try:
            date_dir = self.get_date_dir()

            if filename is None:
                filename = self._build_filename(date_dir)

            filepath = date_dir / filename
            output_image = self._resize_image(image)

            # 이미지 저장
            import cv2
            ok = cv2.imwrite(str(filepath), output_image, self._imwrite_params())
            if not ok:
                raise RuntimeError(f"cv2.imwrite 실패: {filepath}")

            return str(filepath)

        except Exception as e:
            print(f"이미지 저장 오류: {e}")
            return None

    def cleanup_old_files(self):
        """보관 기간이 지난 YYYYMMDD 날짜 폴더 삭제"""
        if not self.save_dir.exists():
            return

        cutoff_date = datetime.now().date().toordinal() - self.retention_days

        with self.lock:
            for date_dir in self.save_dir.iterdir():
                if not date_dir.is_dir():
                    continue

                try:
                    try:
                        folder_date = datetime.strptime(date_dir.name, "%Y%m%d").date().toordinal()
                    except ValueError:
                        # 날짜 폴더가 아니면 삭제하지 않음
                        continue

                    if folder_date < cutoff_date:
                        for file in date_dir.iterdir():
                            if file.is_file():
                                file.unlink()
                        date_dir.rmdir()
                        print(f"오래된 파일 삭제: {date_dir}")

                except Exception as e:
                    print(f"파일 삭제 중 오류: {e}")

    def get_saved_files(self, date_str: str = None) -> list:
        """
        저장된 파일 목록 반환
        Args:
            date_str: 날짜 문자열 (YYYYMMDD), None이면 오늘
        Returns:
            파일 경로 리스트
        """
        if date_str:
            date_dir = self.save_dir / date_str
        else:
            date_dir = self.get_date_dir()

        if not date_dir.exists():
            return []

        files = []
        for file in date_dir.iterdir():
            if file.is_file() and file.suffix.lower() in ['.bmp', '.jpg', '.jpeg', '.png']:
                files.append(str(file))

        return sorted(files)

    def get_all_date_dirs(self) -> list:
        """모든 날짜 디렉토리 반환"""
        if not self.save_dir.exists():
            return []

        dirs = []
        for item in self.save_dir.iterdir():
            if item.is_dir():
                dirs.append(item.name)

        return sorted(dirs, reverse=True)
