"""Compatibility wrapper for :mod:`oarag.vision.vlm_alignment_pipeline`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.vision.vlm_alignment_pipeline")
globals().update(_module.__dict__)
