"""Compatibility wrapper for :mod:`oarag.vision.entity_links`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.vision.entity_links")
globals().update(_module.__dict__)
