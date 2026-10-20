# -*- coding: utf-8 -*-
"""Materialise the v10.7 blueprint's code blocks into a runnable tree.

Extracts every code block from the v10.7 master and writes it to the path it
declares, so the gates can be executed against the real modules:

  * backend/<...>.py        -> the Python reference implementations
  * frontend/src/<...>.ts   -> the TypeScript reference implementations
  * the six Part 5 gates    -> __v107_tests/test_gateN.py

Usage:  python __v107_materialize.py
"""
import io
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
MASTER = os.path.join(ROOT, "KHMER-SOVEREIGN-S2ST-BLUEPRINT-v10.7.txt")
TESTS = os.path.join(ROOT, "__v107_tests")

FENCE = "`" * 3


def extract_blocks(path):
    """Return {declared_path: (lang, body)} for every fenced block."""
    text = io.open(path, encoding="utf-8").read()
    lines = text.split("\n")
    blocks = {}
    cur = None
    lang = None
    buf = []
    for line in lines:
        if cur is None:
            if not line.startswith("#"):
                continue
            m = re.search(r"`([^`]+\.(?:py|ts|rs|toml))`", line)
            if m:
                cand = m.group(1)
                if cand.startswith(("backend/", "frontend/", "crates/")):
                    cur = cand
                    lang = None
                    buf = []
            continue
        if line.startswith(FENCE):
            if lang is None:
                lang = line[3:].strip()
                buf = []
            else:
                blocks[cur] = (lang, "\n".join(buf))
                cur = None
                lang = None
            continue
        if cur is not None and lang is not None:
            buf.append(line)
    return blocks


def write(path, body):
    full = os.path.join(ROOT, *path.split("/"))
    os.makedirs(os.path.dirname(full), exist_ok=True)
    io.open(full, "w", encoding="utf-8", newline="\n").write(body.rstrip("\n") + "\n")


def main():
    blocks = extract_blocks(MASTER)
    n = 0
    for key, (lang, body) in blocks.items():
        if lang not in ("python", "typescript"):
            continue
        write(key, body)
        n += 1

    # The gates live in Part 5, keyed by heading rather than by a path in the
    # heading, so they are extracted separately.
    text = io.open(MASTER, encoding="utf-8").read()
    heads = [(m.start(), m.group(0)) for m in re.finditer(r"### 5\.\d Gate \d[^\n]*", text)]
    os.makedirs(TESTS, exist_ok=True)
    for k in range(len(heads)):
        end = heads[k + 1][0] if k + 1 < len(heads) else text.index("## PART 6:", heads[k][0])
        seg = text[heads[k][0]:end]
        fo = seg.index(FENCE + "python")
        fc = seg.index(FENCE, fo + 10)
        # The heading is "### 5.N Gate N: `...`", so the gate number is the
        # first integer after "5.".
        num = re.search(r"### 5\.(\d) Gate", heads[k][1]).group(1)
        name = "test_gate%s.py" % num
        io.open(os.path.join(TESTS, name), "w", encoding="utf-8", newline="\n").write(seg[fo + 10:fc])
        n += 1

    print("materialised %d files (%d modules + %d gates)" % (n, n - len(heads), len(heads)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
