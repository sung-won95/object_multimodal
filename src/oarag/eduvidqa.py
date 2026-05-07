"""Compatibility wrapper for :mod:`oarag.ingestion.eduvidqa`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.ingestion.eduvidqa")
globals().update(_module.__dict__)
