from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DOMAIN_LEXICON_FILENAME = "domain_lexicon.json"
KOREAN_PARTICLE_SUFFIXES = (
    "으로",
    "에서",
    "에게",
    "부터",
    "까지",
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "와",
    "과",
    "도",
    "만",
    "의",
)


@dataclass(frozen=True)
class DomainLexicon:
    source_path: Path | None = None
    aliases_by_canonical: dict[str, tuple[str, ...]] = field(default_factory=dict)
    alias_to_canonical: dict[str, str] = field(default_factory=dict)

    @property
    def enabled(self) -> bool:
        return self.source_path is not None

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "canonical_term_count": len(self.aliases_by_canonical),
            "alias_count": sum(
                max(0, len(aliases) - 1) for aliases in self.aliases_by_canonical.values()
            ),
            "term_count": sum(len(aliases) for aliases in self.aliases_by_canonical.values()),
        }

    def canonicalize(self, term: str) -> str:
        normalized = normalize_term(term)
        if normalized in self.alias_to_canonical:
            return self.alias_to_canonical[normalized]
        for suffix in KOREAN_PARTICLE_SUFFIXES:
            if normalized.endswith(suffix) and len(normalized) > len(suffix):
                stem = normalized[: -len(suffix)]
                if stem in self.alias_to_canonical:
                    return self.alias_to_canonical[stem]
        return normalized

    def expand_query(self, query: str) -> str:
        additions = self.query_expansion_terms(query)
        if not additions:
            return query
        return " ".join([query, *additions])

    def query_expansion_terms(self, query: str) -> list[str]:
        if not self.enabled:
            return []
        query_terms = {normalize_term(term) for term in _terms(query)}
        seen = {term for term in query_terms if term}
        additions: list[str] = []
        for term in sorted(query_terms):
            canonical = self.canonicalize(term)
            aliases = self.aliases_by_canonical.get(canonical)
            if aliases is None:
                continue
            for alias in aliases:
                normalized_alias = normalize_term(alias)
                if normalized_alias and normalized_alias not in seen:
                    additions.append(alias)
                    seen.add(normalized_alias)
        return additions

    def query_expansion_metadata(self, query: str) -> dict[str, Any]:
        additions = self.query_expansion_terms(query)
        return {
            "enabled": self.enabled,
            "applied": bool(additions),
            "added_term_count": len(additions),
        }


def load_domain_lexicon(
    *,
    project_dir: Path,
    domain_lexicon_path: Path | None = None,
) -> DomainLexicon:
    source_path = resolve_domain_lexicon_path(
        project_dir=project_dir,
        domain_lexicon_path=domain_lexicon_path,
    )
    if source_path is None:
        return DomainLexicon()
    if not source_path.exists():
        raise FileNotFoundError(f"Domain lexicon not found: {source_path}")

    payload = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in domain lexicon: {source_path}")
    return domain_lexicon_from_payload(payload, source_path=source_path)


def resolve_domain_lexicon_path(
    *,
    project_dir: Path,
    domain_lexicon_path: Path | None = None,
) -> Path | None:
    resolved_project_dir = project_dir.expanduser().resolve()
    if domain_lexicon_path is None:
        default_path = resolved_project_dir / DEFAULT_DOMAIN_LEXICON_FILENAME
        return default_path if default_path.exists() else None

    expanded = domain_lexicon_path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (resolved_project_dir / expanded).resolve()


def domain_lexicon_from_payload(payload: dict[str, Any], *, source_path: Path) -> DomainLexicon:
    raw_aliases = payload.get("aliases", {})
    if not isinstance(raw_aliases, dict):
        raise ValueError("domain_lexicon.json must contain an object field named 'aliases'")

    aliases_by_canonical: dict[str, tuple[str, ...]] = {}
    alias_to_canonical: dict[str, str] = {}
    for raw_canonical, raw_values in raw_aliases.items():
        canonical = normalize_term(str(raw_canonical))
        if not canonical:
            raise ValueError("Domain lexicon canonical terms must be non-empty strings")
        values = _coerce_alias_values(raw_values, canonical=canonical)
        aliases = tuple(sorted({canonical, *values}, key=lambda value: (value.casefold(), value)))
        aliases_by_canonical[canonical] = aliases
        for alias in aliases:
            normalized_alias = normalize_term(alias)
            if normalized_alias:
                alias_to_canonical[normalized_alias] = canonical

    return DomainLexicon(
        source_path=source_path.expanduser().resolve(),
        aliases_by_canonical=aliases_by_canonical,
        alias_to_canonical=alias_to_canonical,
    )


def normalize_term(term: str) -> str:
    return re.sub(r"\s+", " ", term.casefold().strip())


def _coerce_alias_values(raw_values: Any, *, canonical: str) -> list[str]:
    if raw_values is None:
        return []
    if isinstance(raw_values, str):
        values = [raw_values]
    elif isinstance(raw_values, list):
        values = raw_values
    else:
        raise ValueError(f"Aliases for '{canonical}' must be a string or list of strings")

    normalized_values: list[str] = []
    for raw_value in values:
        if not isinstance(raw_value, str):
            raise ValueError(f"Aliases for '{canonical}' must be strings")
        value = normalize_term(raw_value)
        if value:
            normalized_values.append(value)
    return normalized_values


def _terms(text: str) -> set[str]:
    return {match.group(0) for match in re.finditer(r"[0-9a-zA-Z가-힣]+", text.casefold())}
