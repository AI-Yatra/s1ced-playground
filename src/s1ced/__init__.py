"""S1-CED: a small encoder that scores typed decisions (yes/no, choice, score) with calibrated probabilities."""
from .hub import DEFAULT_MODEL, resolve_release
from .infer import S1CED

__all__ = ['S1CED', 'DEFAULT_MODEL', 'resolve_release']
__version__ = '1.0.0'
