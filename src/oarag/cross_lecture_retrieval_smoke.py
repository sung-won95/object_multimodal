"""Compatibility wrapper for :mod:`oarag.evaluation.cross_lecture_retrieval_smoke`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.evaluation.cross_lecture_retrieval_smoke")
globals().update(_module.__dict__)
