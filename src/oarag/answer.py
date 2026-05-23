"""Compatibility wrapper for :mod:`oarag.retrieval.answer`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.retrieval.answer")
globals().update(_module.__dict__)
