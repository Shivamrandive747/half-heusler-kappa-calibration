"""Where the privacy checks get the names they must never let through.

Some inputs to the working copy are not redistributed. The checks that keep them out of the release
need to know their names, but a check that spells those names out would itself publish them. So the
names live in a local manifest, data/external/PRIVATE_MANIFEST.json, which is never committed and
never shipped, and every check asks this module for them.

When the manifest is absent (any public copy of this code) every accessor returns its generic
fallback: no private files, no private UUIDs, no private names, the column check reduced to "uuid",
and each helper replaced by its public, identical definition. A script that only READS through here
therefore behaves the same with or without the manifest; only the scans lose the private names, and
make_release.py refuses to build without them.

Manifest keys: uuid_list (repo-relative path), files (repo-relative paths), modules (script names),
name_fragments (substrings), column_prefixes (CSV column prefixes), helpers ({function: module}).
"""
from __future__ import annotations
import os as _rel_os, sys as _rel_sys; _rel_sys.path[1:1] = [_rel_os.path.join(_rel_os.path.dirname(_rel_os.path.abspath(__file__)), "..", _d) for _d in ("analysis", "corpus", "checks", "paper", "")]  # release layout: see make_release.patch_release_paths

import functools
import importlib
import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parent
# the repository root: this file's folder in the working copy, its parent in the release (checks/)
ROOT = next((d for d in (_HERE, _HERE.parent) if (d / "data").is_dir()), _HERE)
MANIFEST = ROOT / "data" / "external" / "PRIVATE_MANIFEST.json"


@functools.lru_cache(maxsize=1)
def load() -> dict:
    """The manifest as a dict; {} when it is absent."""
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def present() -> bool:
    return bool(load())


def private_files() -> set:
    """Repository-relative paths (posix) that may never ship."""
    return set(load().get("files", []))


def uuids() -> set:
    """Identifiers that may never appear in a shipped file."""
    rel = load().get("uuid_list")
    p = ROOT / rel if rel else None
    if not p or not p.exists():
        return set()
    return {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}


def names() -> list:
    """Every name that may never appear in a shipped path or file, lower-cased: the private files'
    stems, the private scripts, the name fragments and the column prefixes."""
    m = load()
    out = {Path(f).stem for f in m.get("files", [])}
    out |= set(m.get("modules", [])) | set(m.get("name_fragments", [])) | set(m.get("column_prefixes", []))
    return sorted({n.lower() for n in out if n})


def column_pattern() -> re.Pattern:
    """Matches a column (or text) that carries a private identifier or list membership."""
    alts = [re.escape(p) for p in load().get("column_prefixes", [])] + ["uuid"]
    return re.compile("|".join(alts), re.I)


def helper(name: str, fallback):
    """The canonical definition of a small pure helper when its home module is available, else
    `fallback`, which must be identical (the public copy restates it)."""
    mod = load().get("helpers", {}).get(name)
    if mod:
        try:
            return getattr(importlib.import_module(mod), name)
        except (ImportError, AttributeError):
            pass
    return fallback
