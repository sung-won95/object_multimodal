"""Compatibility wrapper for :mod:`oarag.graph.graph_ingest`."""

from oarag._compat import alias_module as _alias_module

_module = _alias_module(__name__, "oarag.graph.graph_ingest")
globals().update(_module.__dict__)
