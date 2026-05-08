"""Compatibility wrapper for :mod:`oarag.integrations.neo4j`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.integrations.neo4j")
globals().update(_module.__dict__)
