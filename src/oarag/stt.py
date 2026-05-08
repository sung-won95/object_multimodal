"""Compatibility wrapper for :mod:`oarag.ingestion.stt`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.ingestion.stt")
globals().update(_module.__dict__)
