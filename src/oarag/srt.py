"""Compatibility wrapper for :mod:`oarag.core.srt`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.core.srt")
globals().update(_module.__dict__)
