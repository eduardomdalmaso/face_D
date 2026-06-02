from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel

class DetectionResult(BaseModel):
    x: int
    y: int
    w: int
    h: int
    confidence: float
    landmarks: Optional[dict[str, list[int]]] = None

class BaseDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> list[DetectionResult]:
        """Detecta rostos na imagem e retorna uma lista de DetectionResult."""
        pass
