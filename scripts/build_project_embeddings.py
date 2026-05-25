#!/usr/bin/env python
from __future__ import annotations

import sys

from oarag.cli import main as oarag_main


def main(argv: list[str] | None = None) -> None:
    oarag_main(["build-embeddings", *(sys.argv[1:] if argv is None else argv)])


if __name__ == "__main__":
    main()
