#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Remove non-essential single-line comments from Python files.

Policy (conservative):
- Remove lines that are *only* comments (after stripping leading whitespace).
- Keep shebang and encoding cookie.
- Keep tooling directives that can affect lint/type/format/coverage behavior:
  noqa, pylint, pyright, mypy, fmt, isort, pragma, type: ignore.

This intentionally does NOT attempt to remove inline comments (end-of-line),
nor docstrings (triple-quoted strings).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import os
import re
import shutil
from pathlib import Path
from typing import Iterable


_KEEP_LINE_RE = re.compile(
    r"(noqa|pylint:|pyright:|mypy:|fmt:|isort:|pragma:?\s+no cover|type:\s*ignore)",
    re.IGNORECASE,
)
_ENCODING_RE = re.compile(r"coding[:=]\s*([-\w.]+)")


def _iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        parts = set(p.parts)
        if "__pycache__" in parts or ".venv" in parts or "venv" in parts:
            continue
        yield p


def _should_keep_comment_line(line: str, line_idx: int) -> bool:
    s = line.lstrip()
    if not s.startswith("#"):
        return True
    if line_idx == 0 and s.startswith("#!"):
        return True
    if line_idx <= 1 and _ENCODING_RE.search(s):
        return True
    if _KEEP_LINE_RE.search(s):
        return True
    return False


def strip_file(path: Path) -> tuple[str, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines(keepends=True)
    kept: list[str] = []
    removed = 0
    for i, line in enumerate(lines):
        s = line.lstrip()
        if s.startswith("#"):
            if _should_keep_comment_line(line, i):
                kept.append(line)
            else:
                removed += 1
            continue
        kept.append(line)
    return ("".join(kept), removed)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="Root directory to process (recursively).")
    ap.add_argument("--backup_dir", type=str, default="", help="If set, copy original files here before editing.")
    ap.add_argument("--dry_run", action="store_true", help="Only report how many lines would be removed.")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    if not root.exists():
        raise SystemExit(f"[strip_comments] root not found: {root}")

    backup_dir = Path(args.backup_dir).resolve() if args.backup_dir else None
    if backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)

    total_removed = 0
    changed_files = 0
    for p in _iter_py_files(root):
        new_text, removed = strip_file(p)
        if removed <= 0:
            continue
        total_removed += removed
        changed_files += 1
        if args.dry_run:
            continue
        if backup_dir is not None:
            rel = p.relative_to(root)
            dst = backup_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dst)
        p.write_text(new_text, encoding="utf-8")

    stamp = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if args.dry_run:
        print(f"[strip_comments][dry_run] {stamp} changed_files={changed_files} removed_lines={total_removed}")
    else:
        print(f"[strip_comments] {stamp} changed_files={changed_files} removed_lines={total_removed}")
        if backup_dir is not None:
            print(f"[strip_comments] backups saved to: {backup_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

