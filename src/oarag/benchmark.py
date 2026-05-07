"""Compatibility wrapper for :mod:`oarag.evaluation.benchmark`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.evaluation.benchmark")
globals().update(_module.__dict__)
