"""
Lightweight gene identifier normalization utilities.

This module provides a pluggable mechanism to map between different gene
identifier systems (e.g., Entrez, Ensembl, Symbols) without hard-coding a
database constraint. It is intentionally file-driven to keep policies
configurable by deployment.

Usage:
  - Place a tab- or comma-delimited mapping file on disk with columns such as
    "entrezid", "ensembl", "symbol", and optional "synonyms" (pipe-separated).
  - Set environment variable ISPEC_GENE_MAP_PATH to the file path.
  - The API/CRUD will use the normalizer, when available, to avoid creating
    duplicate E2G rows across different identifier types.

If no mapping file is configured, the normalizer behaves as a no-op.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


_ID_KEYS = ("entrezid", "ensembl", "symbol")


@dataclass
class _GeneGroup:
    entrezid: str | None = None
    ensembl: str | None = None
    symbol: str | None = None
    synonyms: list[str] | None = None


class GeneNormalizer:
    def __init__(self, mapping_path: str | os.PathLike | None = None,
                 preferred_order: tuple[str, ...] = ("entrezid", "ensembl", "symbol")) -> None:
        self.preferred_order = tuple(preferred_order)
        self._groups: list[_GeneGroup] = []
        self._index: dict[tuple[str, str], int] = {}

        if mapping_path:
            self._load_mapping(Path(mapping_path))

    @classmethod
    def from_env(cls) -> "GeneNormalizer | None":
        path = os.getenv("ISPEC_GENE_MAP_PATH")
        if not path:
            return None
        p = Path(path)
        if not p.exists():
            return None
        return cls(p)

    def _load_mapping(self, path: Path) -> None:
        with path.open("r", newline="") as f:
            sample = f.read(1024)
            f.seek(0)
            dialect = csv.Sniffer().sniff(sample, delimiters=",\t")
            reader = csv.DictReader(f, dialect=dialect)
            for row in reader:
                group = _GeneGroup(
                    entrezid=(row.get("entrezid") or row.get("entrez") or "").strip() or None,
                    ensembl=(row.get("ensembl") or "").strip() or None,
                    symbol=(row.get("symbol") or row.get("gene_symbol") or "").strip() or None,
                    synonyms=[s.strip() for s in (row.get("synonyms") or "").split("|") if s.strip()] or None,
                )
                idx = len(self._groups)
                self._groups.append(group)
                for key in _ID_KEYS:
                    val = getattr(group, key)
                    if val:
                        self._index[(key, val)] = idx
                if group.synonyms:
                    for s in group.synonyms:
                        self._index[("symbol", s)] = idx

    def equivalents(self, gene: str, geneidtype: str) -> list[tuple[str, str]]:
        """Return equivalent (type, id) pairs for the supplied identifier.

        If the gene is not in the mapping, returns the original pair.
        """
        gene = (gene or "").strip()
        geneidtype = (geneidtype or "").strip().lower()
        idx = self._index.get((geneidtype, gene))
        if idx is None:
            return [(geneidtype, gene)]

        group = self._groups[idx]
        pairs: list[tuple[str, str]] = []
        for key in self.preferred_order:
            val = getattr(group, key)
            if val:
                pairs.append((key, val))
        # include symbol synonyms to catch different spellings
        if group.synonyms:
            pairs.extend(("symbol", s) for s in group.synonyms)
        # also include the canonical symbol if present
        if group.symbol:
            pairs.append(("symbol", group.symbol))
        # de-duplicate while preserving order
        seen = set()
        uniq: list[tuple[str, str]] = []
        for p in pairs:
            if p not in seen:
                uniq.append(p)
                seen.add(p)
        return uniq or [(geneidtype, gene)]


class TackleGeneNormalizer:
    """Adapter using tackle.containers to derive equivalences.

    Uses HomoloGene-based mapping bundled in ``tackle`` to map Symbols ↔ Entrez
    for human (TaxonID 9606). Ensembl is not handled by this adapter.
    """

    def __init__(self, *, taxon_id: str = "9606") -> None:
        import importlib  # local import to avoid hard dependency

        m = importlib.import_module("tackle.containers")
        self._hm = m.get_hgene_mapper()
        self._taxon = str(taxon_id)
        # build quick indices
        df = self._hm.df
        self._by_symbol = {}
        self._by_entrez = {}
        sub = df[df["TaxonID"].astype(str) == self._taxon]
        for _, row in sub.iterrows():
            sym = str(row.get("Symbol", "")).strip()
            eid = str(row.get("GeneID", "")).strip()
            if sym:
                self._by_symbol.setdefault(sym, set()).add(eid)
            if eid:
                self._by_entrez.setdefault(eid, set()).add(sym)

    @classmethod
    def available(cls) -> bool:
        try:
            import importlib
            importlib.import_module("tackle.containers")
            return True
        except Exception:
            return False

    def equivalents(self, gene: str, geneidtype: str) -> list[tuple[str, str]]:
        gene = (gene or "").strip()
        t = (geneidtype or "").strip().lower()
        pairs: list[tuple[str, str]] = []
        if t == "symbol":
            ents = sorted(self._by_symbol.get(gene, []))
            pairs.extend(("entrezid", e) for e in ents)
            pairs.append(("symbol", gene))
        elif t in ("entrez", "entrezid"):
            syms = sorted(self._by_entrez.get(gene, []))
            pairs.append(("entrezid", gene))
            pairs.extend(("symbol", s) for s in syms)
        else:
            pairs.append((t, gene))
        # de-dup
        seen = set()
        out = []
        for p in pairs:
            if p not in seen:
                out.append(p)
                seen.add(p)
        return out
