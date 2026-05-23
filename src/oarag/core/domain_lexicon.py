from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_DOMAIN_LEXICON_FILENAME = "domain_lexicon.json"
DOMAIN_LEXICON_SCHEMA_VERSION = 1
DOMAIN_LEXICON_EXPANSION_SOURCE = "domain_lexicon"
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
TOKEN_CHARS = r"0-9a-zA-Z가-힣"


@dataclass(frozen=True)
class QueryExpansionTerm:
    term: str
    source: str
    canonical: str
    matched_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "term": self.term,
            "source": self.source,
            "canonical": self.canonical,
            "matched_terms": list(self.matched_terms),
        }


@dataclass(frozen=True)
class QueryExpansion:
    enabled: bool
    source_path: Path | None
    original_query: str
    expanded_query: str
    terms: tuple[QueryExpansionTerm, ...] = ()

    @property
    def applied(self) -> bool:
        return bool(self.terms) and self.expanded_query != self.original_query

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "applied": self.applied,
            "added_term_count": len(self.terms),
            "expanded_query": self.expanded_query,
            "source_path": str(self.source_path) if self.source_path is not None else None,
            "terms": [term.to_dict() for term in self.terms],
        }


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
        return self.expand_query_result(query).expanded_query

    def expand_query_result(self, query: str) -> QueryExpansion:
        additions = self.query_expansion_term_details(query)
        expanded_query = (
            query if not additions else " ".join([query, *(term.term for term in additions)])
        )
        return QueryExpansion(
            enabled=self.enabled,
            source_path=self.source_path,
            original_query=query,
            expanded_query=expanded_query,
            terms=tuple(additions),
        )

    def query_expansion_terms(self, query: str) -> list[str]:
        return [term.term for term in self.query_expansion_term_details(query)]

    def query_expansion_term_details(self, query: str) -> list[QueryExpansionTerm]:
        if not self.enabled:
            return []

        canonical_matches = _canonical_matches(query, self.alias_to_canonical)
        seen = {normalize_term(term) for term in _terms(query) if normalize_term(term)}
        additions: list[QueryExpansionTerm] = []
        for canonical, match in sorted(
            canonical_matches.items(),
            key=lambda item: (item[1]["position"], item[0]),
        ):
            aliases = self.aliases_by_canonical.get(canonical, ())
            matched_terms = tuple(sorted(match["matched_terms"]))
            for alias in aliases:
                normalized_alias = normalize_term(alias)
                if (
                    normalized_alias
                    and normalized_alias not in seen
                    and not _contains_surface_alias(query, normalized_alias)
                ):
                    additions.append(
                        QueryExpansionTerm(
                            term=alias,
                            source=DOMAIN_LEXICON_EXPANSION_SOURCE,
                            canonical=canonical,
                            matched_terms=matched_terms,
                        )
                    )
                    seen.add(normalized_alias)
        return additions

    def query_expansion_metadata(self, query: str) -> dict[str, Any]:
        return self.expand_query_result(query).metadata()


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
    _validate_domain_lexicon_payload(payload)
    raw_aliases = payload["aliases"]
    if not isinstance(raw_aliases, dict):
        raise ValueError("domain_lexicon.json must contain an object field named 'aliases'")

    aliases_by_canonical: dict[str, tuple[str, ...]] = {}
    alias_to_canonical: dict[str, str] = {}
    for raw_canonical, raw_values in raw_aliases.items():
        if not isinstance(raw_canonical, str):
            raise ValueError("Domain lexicon canonical terms must be strings")
        canonical = normalize_term(raw_canonical)
        if not canonical:
            raise ValueError("Domain lexicon canonical terms must be non-empty strings")
        if canonical in aliases_by_canonical:
            raise ValueError(f"Duplicate canonical term after normalization: {canonical}")
        values = _coerce_alias_values(raw_values, canonical=canonical)
        aliases = tuple(_unique_preserving_order([canonical, *values]))
        aliases_by_canonical[canonical] = aliases
        for alias in aliases:
            normalized_alias = normalize_term(alias)
            if normalized_alias:
                existing = alias_to_canonical.get(normalized_alias)
                if existing is not None and existing != canonical:
                    raise ValueError(
                        f"Alias '{normalized_alias}' maps to both '{existing}' and '{canonical}'"
                    )
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
    elif isinstance(raw_values, dict):
        unsupported_fields = set(raw_values) - {"aliases"}
        if unsupported_fields:
            fields = ", ".join(sorted(unsupported_fields))
            raise ValueError(f"Aliases for '{canonical}' contain unsupported fields: {fields}")
        if "aliases" not in raw_values:
            raise ValueError(f"Aliases for '{canonical}' must include an 'aliases' field")
        return _coerce_alias_values(raw_values["aliases"], canonical=canonical)
    elif isinstance(raw_values, list):
        values = raw_values
    else:
        raise ValueError(
            f"Aliases for '{canonical}' must be a string, list of strings, or object"
        )

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


def _validate_domain_lexicon_payload(payload: dict[str, Any]) -> None:
    unsupported_fields = set(payload) - {"schema_version", "aliases"}
    if unsupported_fields:
        fields = ", ".join(sorted(unsupported_fields))
        raise ValueError(f"domain_lexicon.json contains unsupported fields: {fields}")
    if "aliases" not in payload:
        raise ValueError("domain_lexicon.json must contain an object field named 'aliases'")
    schema_version = payload.get("schema_version", DOMAIN_LEXICON_SCHEMA_VERSION)
    if schema_version != DOMAIN_LEXICON_SCHEMA_VERSION:
        raise ValueError(
            "domain_lexicon.json schema_version must be "
            f"{DOMAIN_LEXICON_SCHEMA_VERSION}, got {schema_version!r}"
        )


def _canonical_matches(
    query: str,
    alias_to_canonical: dict[str, str],
) -> dict[str, dict[str, Any]]:
    matches: dict[str, dict[str, Any]] = {}
    for alias, canonical in alias_to_canonical.items():
        match = _alias_match(query, alias)
        if match is None:
            continue
        position, matched_term = match
        canonical_match = matches.setdefault(
            canonical,
            {"position": position, "matched_terms": set()},
        )
        canonical_match["position"] = min(canonical_match["position"], position)
        canonical_match["matched_terms"].add(matched_term)
    return matches


def _alias_match(query: str, alias: str) -> tuple[int, str] | None:
    normalized_query = normalize_term(query)
    position = _find_normalized_term(normalized_query, alias)
    if position is not None:
        return position, alias

    for match in re.finditer(rf"[{TOKEN_CHARS}]+", query.casefold()):
        token = normalize_term(match.group(0))
        if token == alias:
            return match.start(), token
        for suffix in KOREAN_PARTICLE_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix):
                stem = token[: -len(suffix)]
                if stem == alias:
                    return match.start(), token

    if _should_try_folded_match(normalized_query, alias):
        folded_query = _fold_separators(normalized_query)
        folded_alias = _fold_separators(alias)
        folded_position = folded_query.find(folded_alias)
        if folded_position >= 0:
            return folded_position, alias
    return None


def _contains_surface_alias(query: str, alias: str) -> bool:
    if _find_normalized_term(normalize_term(query), alias) is not None:
        return True
    for match in re.finditer(rf"[{TOKEN_CHARS}]+", query.casefold()):
        token = normalize_term(match.group(0))
        if token == alias:
            return True
        for suffix in KOREAN_PARTICLE_SUFFIXES:
            if token.endswith(suffix) and len(token) > len(suffix):
                if token[: -len(suffix)] == alias:
                    return True
    return False


def _find_normalized_term(normalized_query: str, term: str) -> int | None:
    if not term:
        return None
    boundary_chars = f"{TOKEN_CHARS}_-"
    if re.match(rf"[{TOKEN_CHARS}]", term) or re.search(rf"[{TOKEN_CHARS}]$", term):
        pattern = re.escape(term)
        if re.match(rf"[{TOKEN_CHARS}]", term):
            pattern = rf"(?<![{boundary_chars}]){pattern}"
        if re.search(rf"[{TOKEN_CHARS}]$", term):
            pattern = rf"{pattern}(?![{boundary_chars}])"
        match = re.search(pattern, normalized_query)
        return match.start() if match is not None else None

    position = normalized_query.find(term)
    return position if position >= 0 else None


def _fold_separators(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value)


def _should_try_folded_match(normalized_query: str, alias: str) -> bool:
    folded_alias = _fold_separators(alias)
    if not folded_alias or folded_alias == alias and not re.search(r"[\s_-]", normalized_query):
        return False
    if re.search(r"[\s_-]", alias):
        return True
    if re.search(r"\d", alias):
        return True
    if re.search(r"[가-힣]", alias) and len(alias) >= 4:
        return True
    return len(alias) >= 8


def _unique_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        normalized = normalize_term(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique_values.append(normalized)
    return unique_values
