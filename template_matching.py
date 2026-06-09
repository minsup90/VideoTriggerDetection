"""
Template Matching Engine
Template Matching 및 Tenengrade 선명도 분석 모듈
"""
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from pathlib import Path
import threading


@dataclass
class MatchResult:
    """매칭 결과"""
    pattern_index: int
    matched: bool
    score: float
    location: Tuple[int, int]
    threshold: float


@dataclass
class Template:
    """템플릿 정보"""
    index: int
    image: np.ndarray
    roi: Tuple[int, int, int, int]  # (x, y, width, height)
    threshold: float


class TenengradeAnalyzer:
    """Tenengrade 선명도 분석기"""

    @staticmethod
    def calculate(image: np.ndarray) -> float:
        """
        Tenengrade 알고리즘을 사용한 선명도 계산
        Sobel 필터를 사용하여 엣지 강도 계산
        """
        try:
            # 그레이스케일 변환
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image

            # Sobel 필터 적용
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

            # Tenengrade 계산 (엣지 강도의 제곱합)
            tenengrade = np.mean(sobel_x ** 2 + sobel_y ** 2)

            return float(tenengrade)

        except Exception as e:
            print(f"Tenengrade 계산 오류: {e}")
            return 0.0

    @staticmethod
    def select_sharpest_image(images: List[np.ndarray]) -> Tuple[int, np.ndarray, float]:
        """
        이미지 리스트에서 가장 선명한 이미지 선택
        Returns: (index, image, score)
        """
        if not images:
            return -1, np.array([]), 0.0

        best_index = 0
        best_score = 0.0

        for i, img in enumerate(images):
            score = TenengradeAnalyzer.calculate(img)
            if score > best_score:
                best_score = score
                best_index = i

        return best_index, images[best_index], best_score


class OpticalFlowAnalyzer:
    """Optical Flow 분석기 (pyrLK 알고리즘)"""

    def __init__(self, max_corners: int = 100, quality_level: float = 0.01, min_distance: int = 10):
        self.max_corners = max_corners
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_points: Optional[np.ndarray] = None
        self.lock = threading.Lock()

    def calculate_flow_score(self, current_frame: np.ndarray) -> float:
        """
        Optical Flow를 사용한 움직임 점수 계산
        """
        try:
            # 그레이스케일 변환
            if len(current_frame.shape) == 3:
                gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = current_frame

            with self.lock:
                if self.prev_gray is None:
                    self.prev_gray = gray
                    # 코너 검출
                    self.prev_points = cv2.goodFeaturesToTrack(
                        gray,
                        maxCorners=self.max_corners,
                        qualityLevel=self.quality_level,
                        minDistance=self.min_distance
                    )
                    return 0.0

                if self.prev_points is None or len(self.prev_points) < 5:
                    self.prev_gray = gray
                    self.prev_points = cv2.goodFeaturesToTrack(
                        gray,
                        maxCorners=self.max_corners,
                        qualityLevel=self.quality_level,
                        minDistance=self.min_distance
                    )
                    return 0.0

                # Optical Flow 계산
                next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                    self.prev_gray,
                    gray,
                    self.prev_points,
                    None
                )

                # 유효한 포인트만 선택
                good_next = next_points[status == 1]
                good_prev = self.prev_points[status == 1]

                if len(good_next) < 5:
                    self.prev_gray = gray
                    self.prev_points = cv2.goodFeaturesToTrack(
                        gray,
                        maxCorners=self.max_corners,
                        qualityLevel=self.quality_level,
                        minDistance=self.min_distance
                    )
                    return 0.0

                # 이동 거리 계산
                distances = np.linalg.norm(good_next - good_prev, axis=1)
                avg_distance = np.mean(distances)

                # 상태 업데이트
                self.prev_gray = gray
                self.prev_points = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=self.max_corners,
                    qualityLevel=self.quality_level,
                    minDistance=self.min_distance
                )

                return float(avg_distance)

        except Exception as e:
            print(f"Optical Flow 계산 오류: {e}")
            with self.lock:
                self.prev_gray = None
                self.prev_points = None
            return 0.0

    def reset(self):
        """분석기 초기화"""
        with self.lock:
            self.prev_gray = None
            self.prev_points = None


class TemplateMatcher:
    """Template Matching 엔진"""

    def __init__(self):
        self.templates: Dict[int, Template] = {}
        self.lock = threading.Lock()
        self.match_method = cv2.TM_CCOEFF_NORMED  # 정규화된 상관 계수

    def load_template(self, index: int, template_path: str, roi: Tuple[int, int, int, int], threshold: float) -> bool:
        """
        템플릿 로드
        Args:
            index: 패턴 인덱스
            template_path: 템플릿 이미지 경로
            roi: (x, y, width, height)
            threshold: 매칭 임계값
        """
        try:
            # 템플릿 이미지 로드
            template_img = cv2.imread(template_path)
            if template_img is None:
                print(f"템플릿 이미지 로드 실패: {template_path}")
                return False

            # 그레이스케일 변환 (매칭 성능 향상)
            if len(template_img.shape) == 3:
                template_gray = cv2.cvtColor(template_img, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_img

            with self.lock:
                self.templates[index] = Template(
                    index=index,
                    image=template_gray,
                    roi=roi,
                    threshold=threshold
                )

            return True

        except Exception as e:
            print(f"템플릿 로드 오류: {e}")
            return False

    def load_template_from_image(self, index: int, template_image: np.ndarray, roi: Tuple[int, int, int, int], threshold: float) -> bool:
        """
        이미지에서 직접 템플릿 로드
        """
        try:
            # 그레이스케일 변환
            if len(template_image.shape) == 3:
                template_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
            else:
                template_gray = template_image

            with self.lock:
                self.templates[index] = Template(
                    index=index,
                    image=template_gray,
                    roi=roi,
                    threshold=threshold
                )

            return True

        except Exception as e:
            print(f"템플릿 로드 오류: {e}")
            return False

    def remove_template(self, index: int):
        """템플릿 제거"""
        with self.lock:
            if index in self.templates:
                del self.templates[index]

    def match_all(self, frame: np.ndarray, require_all: bool = True) -> Tuple[bool, List[MatchResult]]:
        """
        모든 템플릿에 대해 매칭 수행
        Args:
            frame: 입력 프레임
            require_all: 모든 템플릿이 매칭되어야 하는지 여부 (AND 조건)
        Returns:
            (overall_matched, results)
        """
        if not self.templates:
            return False, []

        # 그레이스케일 변환
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        results = []

        with self.lock:
            templates_copy = self.templates.copy()

        for index, template in templates_copy.items():
            result = self._match_single(gray, template)
            results.append(result)

        # 전체 매칭 결과 결정
        if require_all:
            overall_matched = all(r.matched for r in results)
        else:
            overall_matched = any(r.matched for r in results)

        return overall_matched, results

    def _match_single(self, gray_frame: np.ndarray, template: Template) -> MatchResult:
        """
        단일 템플릿 매칭
        """
        try:
            # ROI 영역 추출
            x, y, w, h = template.roi
            roi_frame = gray_frame[y:y+h, x:x+w]

            if roi_frame.shape[0] < template.image.shape[0] or roi_frame.shape[1] < template.image.shape[1]:
                # ROI가 템플릿보다 작은 경우
                return MatchResult(
                    pattern_index=template.index,
                    matched=False,
                    score=0.0,
                    location=(0, 0),
                    threshold=template.threshold
                )

            # Template Matching 수행
            result = cv2.matchTemplate(roi_frame, template.image, self.match_method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

            # TM_CCOEFF_NORMED의 경우 최대값이 최적 매칭
            score = max_val
            location = (x + max_loc[0], y + max_loc[1])
            matched = score >= template.threshold

            return MatchResult(
                pattern_index=template.index,
                matched=matched,
                score=score,
                location=location,
                threshold=template.threshold
            )

        except Exception as e:
            print(f"템플릿 매칭 오류: {e}")
            return MatchResult(
                pattern_index=template.index,
                matched=False,
                score=0.0,
                location=(0, 0),
                threshold=template.threshold
            )

    def get_template_count(self) -> int:
        """등록된 템플릿 수 반환"""
        with self.lock:
            return len(self.templates)

    def has_template(self, index: int) -> bool:
        """특정 인덱스의 템플릿 존재 여부 확인"""
        with self.lock:
            return index in self.templates

    def get_template_size(self, index: int) -> Optional[Tuple[int, int]]:
        """
        템플릿 크기 반환
        Returns: (width, height) or None if not found
        """
        with self.lock:
            if index in self.templates:
                template = self.templates[index]
                h, w = template.image.shape
                return (w, h)
        return None

    def clear_all(self):
        """모든 템플릿 제거"""
        with self.lock:
            self.templates.clear()


class TriggerBuffer:
    """트리거 버퍼 - 매칭된 이미지를 저장하는 버퍼"""

    def __init__(self, max_size: int = 10):
        self.buffer: List[np.ndarray] = []
        self.max_size = max_size
        self.lock = threading.Lock()

    def add_frame(self, frame: np.ndarray):
        """프레임 추가"""
        with self.lock:
            if len(self.buffer) < self.max_size:
                self.buffer.append(frame)

    def get_best_frame(self) -> Optional[np.ndarray]:
        """
        버퍼에서 가장 선명한 이미지 반환
        """
        with self.lock:
            if not self.buffer:
                return None

            best_index, best_image, best_score = TenengradeAnalyzer.select_sharpest_image(self.buffer)
            return best_image

    def clear(self):
        """버퍼 비우기"""
        with self.lock:
            self.buffer.clear()

    def size(self) -> int:
        """버퍼 크기 반환"""
        with self.lock:
            return len(self.buffer)

    def is_full(self) -> bool:
        """버퍼가 가득 찼는지 확인"""
        with self.lock:
            return len(self.buffer) >= self.max_size
