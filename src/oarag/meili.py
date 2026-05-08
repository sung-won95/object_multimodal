"""Compatibility wrapper for :mod:`oarag.integrations.meili`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.integrations.meili")
globals().update(_module.__dict__)
