"""Compatibility wrapper for :mod:`oarag.vision.vlm_pipeline_metrics`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.vision.vlm_pipeline_metrics")
globals().update(_module.__dict__)
