"""
RTSP Stream Module
RTSP 비디오 스트림을 수신하고 관리하는 모듈
"""
import cv2
import threading
import time
from collections import deque
from queue import Queue, Empty
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum


class StreamState(Enum):
    """스트림 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class FrameInfo:
    """프레임 정보"""
    frame: any
    timestamp: float
    frame_number: int


class RTSPStream:
    """RTSP 비디오 스트림 관리 클래스"""

    def __init__(self, rtsp_url: str, reconnect_interval: int = 5):
        self.rtsp_url = rtsp_url
        self.reconnect_interval = reconnect_interval
        self.state = StreamState.DISCONNECTED
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame_queue: Queue = Queue(maxsize=30)  # 최대 30프레임 버퍼
        self.running = False
        self.thread: Optional[threading.Thread] = None
        self.frame_number = 0
        self.last_frame_time = 0
        self.last_frame_timestamp = 0.0
        self.fps = 0.0
        self.fps_timestamps = deque(maxlen=120)
        self.error_callback: Optional[Callable[[str], None]] = None
        self.state_callback: Optional[Callable[[StreamState], None]] = None
        self.restart_requested = threading.Event()
        self.last_error_message = ""
        self.last_error_notified_at = 0.0
        self.error_rate_limit_sec = 5.0

    def set_error_callback(self, callback: Callable[[str], None]):
        """에러 콜백 설정"""
        self.error_callback = callback

    def set_state_callback(self, callback: Callable[[StreamState], None]):
        """상태 변경 콜백 설정"""
        self.state_callback = callback

    def _notify_state(self, state: StreamState):
        """상태 변경 알림"""
        if self.state == state:
            return
        self.state = state
        if self.state_callback:
            self.state_callback(state)

    def _notify_error(self, error_msg: str):
        """에러 알림"""
        now = time.time()
        if (
            error_msg == self.last_error_message
            and now - self.last_error_notified_at < self.error_rate_limit_sec
        ):
            return
        self.last_error_message = error_msg
        self.last_error_notified_at = now
        if self.error_callback:
            self.error_callback(error_msg)

    def connect(self) -> bool:
        """RTSP 스트림 연결"""
        self._notify_state(StreamState.CONNECTING)

        try:
            # OpenCV VideoCapture 생성
            self.cap = cv2.VideoCapture(self.rtsp_url)

            # 연결 확인
            if not self.cap.isOpened():
                self._release_capture()
                self._notify_error(f"RTSP 연결 실패: {self.rtsp_url}")
                self._notify_state(StreamState.ERROR)
                return False

            # 버퍼 크기 설정 (지연 감소)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # 해상도 설정 (선택사항)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

            self.last_frame_time = 0
            self.last_frame_timestamp = 0.0
            self.fps = 0.0
            self.fps_timestamps.clear()
            self._notify_state(StreamState.CONNECTED)
            return True

        except Exception as e:
            self._release_capture()
            self._notify_error(f"RTSP 연결 중 오류 발생: {str(e)}")
            self._notify_state(StreamState.ERROR)
            return False

    def disconnect(self):
        """RTSP 스트림 연결 해제"""
        self.running = False
        if self.thread and threading.current_thread() != self.thread:
            self.thread.join(timeout=2.0)

        if self.cap:
            self.cap.release()
            self.cap = None

        self.fps = 0.0
        self.last_frame_timestamp = 0.0
        self.fps_timestamps.clear()
        self._notify_state(StreamState.DISCONNECTED)

    def start(self):
        """스트림 수신 시작"""
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        """스트림 수신 중지"""
        self.running = False
        self.restart_requested.set()
        if self.thread and threading.current_thread() != self.thread:
            self.thread.join(timeout=2.0)

    def _release_capture(self):
        """캡처 리소스만 해제하고 수신 루프는 유지한다."""
        if self.cap:
            self.cap.release()
            self.cap = None
        self.fps = 0.0
        self.last_frame_timestamp = 0.0
        self.fps_timestamps.clear()

    def request_restart(self):
        """캡처 스레드에 비동기 재시작을 요청한다."""
        self.restart_requested.set()

    def _sleep_reconnect_interval(self):
        """중지/재시작 요청에 빨리 반응하면서 재연결 간격만큼 대기한다."""
        end_time = time.time() + self.reconnect_interval
        while self.running and time.time() < end_time:
            if self.restart_requested.is_set():
                return
            time.sleep(min(0.1, max(0.0, end_time - time.time())))

    def _handle_restart_request(self):
        """요청된 재시작을 캡처 스레드 내부에서 처리한다."""
        if not self.restart_requested.is_set():
            return False
        self.restart_requested.clear()
        self._release_capture()
        self._notify_state(StreamState.DISCONNECTED)
        return True

    def _capture_loop(self):
        """프레임 캡처 루프"""
        while self.running:
            if self._handle_restart_request():
                continue

            if self.state != StreamState.CONNECTED:
                if not self.connect():
                    self._sleep_reconnect_interval()
                    continue

            try:
                if self._handle_restart_request():
                    continue

                # 프레임 읽기
                ret, frame = self.cap.read()

                if not ret:
                    self._notify_error("프레임 읽기 실패")
                    self._release_capture()
                    self._notify_state(StreamState.ERROR)
                    self._sleep_reconnect_interval()
                    continue

                # FPS 계산: 순간값 대신 최근 timestamp 기반 이동평균 사용
                current_time = time.time()
                self.last_frame_timestamp = current_time
                self.fps_timestamps.append(current_time)
                while self.fps_timestamps and current_time - self.fps_timestamps[0] > 3.0:
                    self.fps_timestamps.popleft()
                if len(self.fps_timestamps) >= 2:
                    elapsed = self.fps_timestamps[-1] - self.fps_timestamps[0]
                    self.fps = (len(self.fps_timestamps) - 1) / elapsed if elapsed > 0 else 0.0
                else:
                    self.fps = 0.0
                self.last_frame_time = current_time

                # 프레임 번호 증가
                self.frame_number += 1

                # 큐에 프레임 추가 (오래된 프레임 제거)
                frame_info = FrameInfo(
                    frame=frame,
                    timestamp=current_time,
                    frame_number=self.frame_number
                )

                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except Empty:
                        pass

                self.frame_queue.put(frame_info)

            except Exception as e:
                self._notify_error(f"프레임 캡처 중 오류: {str(e)}")
                self._release_capture()
                self._notify_state(StreamState.ERROR)
                self._sleep_reconnect_interval()

    def get_frame(self) -> Optional[FrameInfo]:
        """최신 프레임 반환"""
        try:
            return self.frame_queue.get_nowait()
        except Empty:
            return None

    def get_latest_frame(self) -> Optional[FrameInfo]:
        """가장 최신 프레임 반환 (큐 비우지 않음)"""
        try:
            # 큐의 모든 프레임을 확인하고 가장 최신 것 반환
            latest_frame = None
            while not self.frame_queue.empty():
                latest_frame = self.frame_queue.get_nowait()
            return latest_frame
        except Empty:
            return None

    def is_connected(self) -> bool:
        """연결 상태 확인"""
        return self.state == StreamState.CONNECTED

    def get_fps(self) -> float:
        """현재 FPS 반환. 최근 5초 이상 프레임이 없으면 0으로 표시."""
        if self.last_frame_timestamp and time.time() - self.last_frame_timestamp > 5.0:
            return 0.0
        return self.fps

    def get_last_frame_timestamp(self) -> float:
        """마지막 프레임 수신 timestamp 반환"""
        return self.last_frame_timestamp

    def restart(self):
        """스트림 재시작"""
        self.request_restart()

    def get_frame_number(self) -> int:
        """현재 프레임 번호 반환"""
        return self.frame_number

    def get_resolution(self) -> tuple:
        """비디오 해상도 반환"""
        if self.cap and self.cap.isOpened():
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (width, height)
        return (0, 0)


class FrameBuffer:
    """프레임 버퍼 클래스 - 이미지 수집용"""

    def __init__(self, max_size: int = 300):
        self.frames: list = []
        self.max_size = max_size
        self.lock = threading.Lock()

    def add_frame(self, frame: any, timestamp: float):
        """프레임 추가"""
        with self.lock:
            if len(self.frames) < self.max_size:
                self.frames.append({
                    'frame': frame,
                    'timestamp': timestamp
                })

    def get_frames(self) -> list:
        """모든 프레임 반환"""
        with self.lock:
            return self.frames.copy()

    def clear(self):
        """버퍼 비우기"""
        with self.lock:
            self.frames.clear()

    def size(self) -> int:
        """버퍼 크기 반환"""
        with self.lock:
            return len(self.frames)

    def is_full(self) -> bool:
        """버퍼가 가득 찼는지 확인"""
        with self.lock:
            return len(self.frames) >= self.max_size
