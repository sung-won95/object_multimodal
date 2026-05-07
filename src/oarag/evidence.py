"""Compatibility wrapper for :mod:`oarag.retrieval.evidence`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.retrieval.evidence")
globals().update(_module.__dict__)
