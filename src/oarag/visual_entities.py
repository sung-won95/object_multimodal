"""Compatibility wrapper for :mod:`oarag.vision.visual_entities`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.vision.visual_entities")
globals().update(_module.__dict__)
