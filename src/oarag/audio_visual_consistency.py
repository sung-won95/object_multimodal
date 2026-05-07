"""Compatibility wrapper for :mod:`oarag.vision.audio_visual_consistency`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.vision.audio_visual_consistency")
globals().update(_module.__dict__)
