"""Signal detection algorithms for TokenCircuit."""

from .composite import CompositeDetector as CompositeDetector
from .composite import DetectionResult as DetectionResult
from .futile_action import FutileActionDetector as FutileActionDetector
from .state_stagnation import (
    StateStagnationDetector as StateStagnationDetector,
)

__all__ = [
    "CompositeDetector",
    "DetectionResult",
    "FutileActionDetector",
    "StateStagnationDetector",
]
