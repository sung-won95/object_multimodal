"""Compatibility wrapper for :mod:`oarag.retrieval.project_query`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.retrieval.project_query")
globals().update(_module.__dict__)
