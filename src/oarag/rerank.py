"""Compatibility wrapper for :mod:`oarag.retrieval.rerank`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.retrieval.rerank")
globals().update(_module.__dict__)
