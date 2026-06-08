"""Compatibility wrapper for :mod:`oarag.vision.vlm_evidence_validator`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.vision.vlm_evidence_validator")
globals().update(_module.__dict__)
