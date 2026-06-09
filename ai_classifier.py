"""
AI Classification Module
CPU 기반 AI Classification 모듈 (추후 확장용)
"""
import numpy as np
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import threading


@dataclass
class ClassificationResult:
    """분류 결과"""
    class_name: str
    confidence: float
    success: bool


class AIClassifier:
    """AI 분류기 클래스 (추후 확장용)"""

    def __init__(self, model_path: str = "", threshold: float = 0.8):
        self.model_path = model_path
        self.threshold = threshold
        self.enabled = False
        self.model = None
        self.lock = threading.Lock()

    def load_model(self, model_path: str) -> bool:
        """
        AI 모델 로드
        추후 ONNX, TensorFlow Lite 등 다양한 형식 지원 예정
        """
        try:
            self.model_path = model_path

            # 추후 실제 모델 로드 코드 추가
            # 예: ONNX 모델 로드
            # import onnxruntime as ort
            # self.model = ort.InferenceSession(model_path)

            self.enabled = True
            return True

        except Exception as e:
            print(f"AI 모델 로드 실패: {e}")
            self.enabled = False
            return False

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """
        이미지 분류 수행
        Args:
            image: 입력 이미지 (BGR)
        Returns:
            ClassificationResult
        """
        if not self.enabled or self.model is None:
            return ClassificationResult(
                class_name="",
                confidence=0.0,
                success=False
            )

        try:
            with self.lock:
                # 추후 실제 추론 코드 추가
                # 예: ONNX 추론
                # input_name = self.model.get_inputs()[0].name
                # output_name = self.model.get_outputs()[0].name
                # result = self.model.run([output_name], {input_name: preprocessed_image})

                # 현재는 더미 결과 반환
                confidence = 0.0
                class_name = ""

                success = confidence >= self.threshold

                return ClassificationResult(
                    class_name=class_name,
                    confidence=confidence,
                    success=success
                )

        except Exception as e:
            print(f"AI 분류 오류: {e}")
            return ClassificationResult(
                class_name="",
                confidence=0.0,
                success=False
            )

    def set_threshold(self, threshold: float):
        """분류 임계값 설정"""
        self.threshold = threshold

    def get_threshold(self) -> float:
        """분류 임계값 반환"""
        return self.threshold

    def is_enabled(self) -> bool:
        """활성화 상태 확인"""
        return self.enabled

    def set_enabled(self, enabled: bool):
        """활성화/비활성화 설정"""
        self.enabled = enabled


class DummyAIClassifier(AIClassifier):
    """
    더미 AI 분류기
    테스트용으로 랜덤 결과 반환
    """

    def __init__(self, threshold: float = 0.8):
        super().__init__("", threshold)
        self.enabled = True

    def classify(self, image: np.ndarray) -> ClassificationResult:
        """더미 분류 결과 반환"""
        import random

        # 랜덤 confidence 생성 (0.5 ~ 1.0)
        confidence = random.uniform(0.5, 1.0)
        class_name = "target" if confidence >= self.threshold else "background"

        success = confidence >= self.threshold

        return ClassificationResult(
            class_name=class_name,
            confidence=confidence,
            success=success
        )


class CombinedTrigger:
    """
    Template Matching과 AI Classification을 결합한 트리거
    AND/OR 조건 지원
    """

    def __init__(self, condition_type: str = "and"):
        """
        Args:
            condition_type: 'and' 또는 'or'
        """
        self.condition_type = condition_type.lower()
        self.template_matched = False
        self.ai_matched = False
        self.template_scores: Dict[int, float] = {}
        self.ai_confidence = 0.0

    def update_template_result(self, matched: bool, scores: Dict[int, float]):
        """Template Matching 결과 업데이트"""
        self.template_matched = matched
        self.template_scores = scores

    def update_ai_result(self, result: ClassificationResult):
        """AI Classification 결과 업데이트"""
        self.ai_matched = result.success
        self.ai_confidence = result.confidence

    def check_trigger(self) -> bool:
        """
        트리거 조건 확인
        Returns:
            트리거 발생 여부
        """
        if self.condition_type == "and":
            # AND 조건: 둘 다 True여야 함
            return self.template_matched and self.ai_matched
        elif self.condition_type == "or":
            # OR 조건: 둘 중 하나라도 True면 발생
            return self.template_matched or self.ai_matched
        else:
            # 기본: Template Matching만 사용
            return self.template_matched

    def get_status(self) -> Dict:
        """현재 상태 반환"""
        return {
            'condition_type': self.condition_type,
            'template_matched': self.template_matched,
            'ai_matched': self.ai_matched,
            'template_scores': self.template_scores,
            'ai_confidence': self.ai_confidence,
            'triggered': self.check_trigger()
        }

    def set_condition_type(self, condition_type: str):
        """조건 타입 설정"""
        self.condition_type = condition_type.lower()

    def get_condition_type(self) -> str:
        """조건 타입 반환"""
        return self.condition_type
