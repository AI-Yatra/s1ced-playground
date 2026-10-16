"""S1-CED decisions, and one contract (TypeSafe's System One API) for S1-CED, Vega and Jev.

    import s1ced
    m = s1ced.load('s1ced')                 # or 'vega-0.8b', 'vega-4b', 'jev-latest'
    out = m.decide(state, questions)
    out = s1ced.decide({'model': 'vega-0.8b', 'state': state, 'questions': questions})
"""
from .contract import DecisionError, check_request, check_response
from .hub import DEFAULT_MODEL, resolve_release
from .infer import S1CED
from .models import Registry, decide, load

__all__ = ['load', 'decide', 'DecisionError', 'check_request', 'check_response', 'Registry', 'S1CED', 'DEFAULT_MODEL', 'resolve_release']
__version__ = '2.0.0'
