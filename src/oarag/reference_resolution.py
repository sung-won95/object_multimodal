"""Compatibility wrapper for :mod:`oarag.vision.reference_resolution`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.vision.reference_resolution")
globals().update(_module.__dict__)
