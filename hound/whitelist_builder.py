#!/usr/bin/env python3
"""
whitelist-builder: Generate a file whitelist for Hound from an arbitrary codebase.
Heuristic preselection + optional LLM re-ranking. Supports mixed codebases (C/Go/Rust/Web/etc.).
Usage:
  python whitelist_builder.py \
    --input /path/to/repo \
    --limit-loc 20000 \
    --output whitelist.txt \
    --enable-llm --model gpt-4o-mini
Notes:
- Source API keys first if using LLM: `source API_KEYS.txt`
- Output is a newline-separated whitelist of relative file paths.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess
import time
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Set


# --------- Heuristics ---------

BIN_EXT = {
    '.png', '.jpg', '.jpeg', '.gif', '.webp', '.svg', '.ico',
    '.pdf', '.zip', '.gz', '.tar', '.tgz', '.xz', '.7z', '.rar',
    '.woff', '.woff2', '.ttf', '.otf', '.eot', '.mp3', '.mp4', '.mov', '.webm',
}

DEFAULT_EXCLUDE_DIRS = {
    '.git', '.hg', '.svn', '.idea', '.vscode',
    'node_modules', 'vendor', 'dist', 'build', 'target', 'out', '.next', '.cache',
    '__pycache__', '.venv', 'venv', '.tox', '.mypy_cache', 'coverage', 'site-packages',
}

DEFAULT_EXCLUDE_GLOBS = [
    '**/*~', '**/*.tmp', '**/*.bak', '**/*.log', '**/*.min.*', '**/*.lock',
]

TEST_HINTS = re.compile(r"(^|/)(tests?|__tests__|testdata|spec|mocks?|fixtures?)(/|$)", re.I)
TEST_FILE_HINTS = re.compile(r"(test|spec|mock|fixture)\b", re.I)

LANG_WEIGHTS = {
    # Core languages
    '.c': 1.0, '.h': 0.9, '.cpp': 1.0, '.hpp': 0.9, '.cc': 1.0,
    '.go': 1.2, '.rs': 1.2, '.sol': 1.3,
    '.ts': 1.0, '.tsx': 1.0, '.js': 0.9, '.jsx': 0.9,
    '.py': 1.0, '.java': 0.9, '.cs': 0.8,
    # Config-ish (keep but with lower weight)
    '.toml': 0.5, '.json': 0.4, '.yml': 0.5, '.yaml': 0.5, '.ini': 0.4, '.env': 0.4,
    '.md': 0.1,
}

PATH_BOOSTS = [
    (re.compile(r"(^|/)(src|lib|pkg|internal|core|server|api|cmd)(/|$)", re.I), 0.6),
    (re.compile(r"(^|/)(contracts?)(/|$)", re.I), 0.8),
    (re.compile(r"(^|/)(app|router|handler|controller|service|models?)(/|$)", re.I), 0.4),
]

PATH_PENALTIES = [
    (TEST_HINTS, -1.5),
    (re.compile(r"(^|/)(examples?|samples?|demos?|docs?)(/|$)", re.I), -0.6),
    (re.compile(r"(^|/)(scripts?|ci|\.github|doc)(/|$)", re.I), -0.3),
]

CONFIG_TOP = {
    'Dockerfile', 'docker-compose.yml', 'Makefile',
    'Cargo.toml', 'Cargo.lock', 'go.mod', 'go.sum',
    'package.json', 'pnpm-lock.yaml', 'yarn.lock',
}

ENTRYPOINT_HINTS = [
    re.compile(r"(^|/)cmd/[^/]+/main\.go$", re.I),
    re.compile(r"(^|/)main\.go$", re.I),
    re.compile(r"(^|/)src/main\.rs$", re.I),
    re.compile(r"(^|/)src/main\.(ts|tsx|js|jsx)$", re.I),
]


@dataclass
class FileInfo:
    path: Path
    rel: str
    ext: str
    loc: int
    score: float
    reasons: List[str]


def is_binary(path: Path) -> bool:
    if path.suffix.lower() in BIN_EXT:
        return True
    try:
        with open(path, 'rb') as f:
            chunk = f.read(1024)
        if b'\0' in chunk:
            return True
        # heuristic: too many non-text bytes
        text_chars = sum(c >= 9 and c <= 127 or c in (9, 10, 13) for c in chunk)
        return (text_chars / max(1, len(chunk))) < 0.80
    except Exception:
        return True


def count_loc(path: Path) -> int:
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def score_file(root: Path, path: Path) -> Tuple[float, List[str]]:
    """Return (score, reasons) for a file."""
    reasons: List[str] = []
    rel = str(path.relative_to(root))
    ext = path.suffix.lower()
    score = 0.0

    # Language weight
    lw = LANG_WEIGHTS.get(ext, 0.2)
    score += lw
    reasons.append(f"lang:{ext}:{lw}")

    # Path boosts / penalties
    for rx, w in PATH_BOOSTS:
        if rx.search(rel):
            score += w
            reasons.append(f"boost:{w}")
    for rx, w in PATH_PENALTIES:
        if rx.search(rel):
            score += w
            reasons.append(f"penalty:{w}")

    # Tests or mock-like files get penalized extra
    base = path.name
    if TEST_FILE_HINTS.search(base):
        score -= 0.8
        reasons.append("testname:-0.8")

    # Entrypoints boost
    for rx in ENTRYPOINT_HINTS:
        if rx.search(rel):
            score += 0.7
            reasons.append("entry:+0.7")
            break

    # Config top-level files: moderate boost
    if base in CONFIG_TOP and path.parent == root:
        score += 0.5
        reasons.append("config:+0.5")

    return score, reasons


def iter_candidates(root: Path, only_ext: Optional[Set[str]] = None) -> Iterable[Path]:
    for p in root.rglob('*'):
        if p.is_dir():
            # Skip excluded directories
            if p.name in DEFAULT_EXCLUDE_DIRS:
                # prune by skipping children
                continue
            # rely on rglob to iterate
        elif p.is_file():
            rel = str(p.relative_to(root))
            # Exclude hidden files (mostly)
            if any(seg.startswith('.') and seg not in {'.env'} for seg in p.parts):
                continue
            # Exclude default globs
            if any(Path(rel).match(g) for g in DEFAULT_EXCLUDE_GLOBS):
                continue
            # Ignore obvious binaries
            if is_binary(p):
                continue
            # Only certain extensions if requested
            if only_ext is not None:
                if p.suffix.lower() not in only_ext:
                    continue
            yield p


def _fmt_eta(seconds: float) -> str:
    if seconds <= 0 or seconds != seconds:
        return "?s"
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def build_candidates(root: Path, workers: int = 8, verbose: bool = False, progress: bool = True, only_ext: Optional[Set[str]] = None) -> List[FileInfo]:
    start = time.time()
    paths = list(iter_candidates(root, only_ext=only_ext))
    if verbose:
        print(f"[scan] Found {len(paths)} text-like files; counting LOC and scoring (workers={workers})…")
    items: List[FileInfo] = []
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        futs = {ex.submit(count_loc, p): p for p in paths}
        total = len(futs)
        done = 0
        last_report = 0.0
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                loc = fut.result()
            except Exception:
                loc = 0
            if loc <= 0:
                continue
            score, reasons = score_file(root, p)
            items.append(FileInfo(
                path=p,
                rel=str(p.relative_to(root)),
                ext=p.suffix.lower(),
                loc=loc,
                score=score,
                reasons=reasons,
            ))
            done += 1
            if progress:
                now = time.time()
                if now - last_report > 0.5:
                    elapsed = now - start
                    # Use processed rate to estimate remaining
                    eta = (elapsed / max(1, done)) * max(0, total - done)
                    pct = int(done * 100 / max(1, total))
                    sys.stdout.write(f"\r[scan] {done}/{total} ({pct}%) counted — elapsed {int(elapsed)}s, eta {_fmt_eta(eta)}   ")
                    sys.stdout.flush()
                    last_report = now
        if progress and total:
            # Clear progress line
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
    if verbose:
        print(f"[scan] Built {len(items)} candidates in {time.time()-start:.1f}s")
    return items


def heuristic_rank(items: List[FileInfo]) -> List[FileInfo]:
    # Higher score first; within same score, prefer mid-sized files (avoid single huge file dominating)
    return sorted(items, key=lambda x: (x.score, min(x.loc, 5000)), reverse=True)


def llm_rerank(items: List[FileInfo], model: str = 'gpt-4o-mini', max_items: int = 300, verbose: bool = False) -> List[str]:
    """Use OpenAI to rerank candidate files; returns list of relpaths in prioritized order."""
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        print(f"[LLM] OpenAI library not available: {e}")
        return [it.rel for it in items]

    client = OpenAI()

    # Prepare compact JSON describing candidates
    sample = [
        {
            'rel': it.rel,
            'loc': it.loc,
            'score': round(it.score, 3),
            'ext': it.ext,
            'reasons': it.reasons[:4],
        }
        for it in items[:max_items]
    ]

    system = (
        "You prioritize source files for a security audit whitelist. "
        "Goal: choose the most security-relevant files across the project: core logic, state, auth, interfaces, and important configs. "
        "Avoid tests/mocks/fixtures unless they contain unique logic. Prefer spread across critical components. Respond JSON only."
    )
    user = json.dumps({
        'candidates': sample,
        'instructions': (
            'Return JSON with a single field "prioritized", an array of relpaths, in priority order. '
            'Do not include any other fields.'
        )
    }, indent=2)

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        if verbose:
            print(f"[llm] Rerank completed in {time.time()-t0:.1f}s, model={model}")
        txt = resp.choices[0].message.content or '{}'
        data = json.loads(txt)
        arr = data.get('prioritized') or []
        out = [p for p in arr if isinstance(p, str)]
        if out:
            return out
        return [it.rel for it in items]
    except Exception as e:
        print(f"[LLM] Rerank failed: {e}")
        return [it.rel for it in items]


def run_cloc(root: Path) -> tuple[int, dict[str, int]]:
    """Run `cloc` if available; returns (total, by_language). Raises on errors."""
    try:
        cp = subprocess.run(["cloc", "--json", "--quiet", str(root)], capture_output=True, text=True, check=True)
        data = json.loads(cp.stdout or '{}')
        total = int((data.get('SUM') or {}).get('code') or 0)
        by_lang: dict[str, int] = {}
        for k, v in data.items():
            if k == 'header' or k == 'SUM' or not isinstance(v, dict):
                continue
            code = int(v.get('code') or 0)
            by_lang[str(k)] = code
        return total, by_lang
    except Exception as e:
        raise RuntimeError(f"cloc failed: {e}")


def approx_repo_loc_by_ext(root: Path, workers: int = 8, only_ext: Optional[Set[str]] = None) -> tuple[int, dict[str, int]]:
    total = 0
    by_ext: dict[str, int] = {}
    paths = list(iter_candidates(root, only_ext=only_ext))
    with ThreadPoolExecutor(max_workers=max(2, workers)) as ex:
        futs = {ex.submit(count_loc, p): p for p in paths}
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                loc = fut.result()
            except Exception:
                loc = 0
            if loc <= 0:
                continue
            total += loc
            by_ext[p.suffix.lower()] = by_ext.get(p.suffix.lower(), 0) + loc
    return total, by_ext


def compile_whitelist(prioritized_paths: List[str], by_rel: dict[str, FileInfo], limit_loc: int) -> Tuple[List[str], int]:
    """
    Select files within LOC budget.
    
    Fixed: If all files exceed the limit individually, select at least the first file
    to avoid returning 0 files (which causes Hound to fail).
    """
    selected: List[str] = []
    total = 0
    
    for rel in prioritized_paths:
        info = by_rel.get(rel)
        if not info:
            continue
        
        # If this file would exceed the limit
        if total + info.loc > limit_loc:
            # If we haven't selected anything yet, include this file anyway
            # (better than having 0 files, which causes Hound to fail)
            if len(selected) == 0:
                selected.append(rel)
                total += info.loc
                # Continue to try to add more files if possible
                continue
            # If we've already selected some files, stop here
            else:
                break
        
        # File fits within budget
        selected.append(rel)
        total += info.loc
        
        if total >= limit_loc:
            break
    
    return selected, total


def apply_overrides(cfg_path: str):
    try:
        with open(cfg_path, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
    except Exception as e:
        print(f"[config] Failed to load overrides: {e}")
        return
    # Language weights
    try:
        lw = cfg.get('lang_weights') or {}
        for k, v in lw.items():
            LANG_WEIGHTS[str(k).lower()] = float(v)
    except Exception:
        pass
    # Path boosts / penalties
    def _compile_list(key: str, target_list: list, sign: str = 'boost'):
        arr = cfg.get(key) or []
        for item in arr:
            try:
                rx = re.compile(str(item['pattern']), re.I)
                w = float(item['weight'])
                target_list.append((rx, w))
            except Exception:
                continue
    _compile_list('path_boosts', PATH_BOOSTS, 'boost')
    _compile_list('path_penalties', PATH_PENALTIES, 'penalty')
    # Config top and entrypoints
    try:
        top = cfg.get('config_top') or []
        for name in top:
            CONFIG_TOP.add(str(name))
    except Exception:
        pass
    try:
        eps = cfg.get('entrypoint_hints') or []
        for pat in eps:
            ENTRYPOINT_HINTS.append(re.compile(str(pat), re.I))
    except Exception:
        pass


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description='Build a Hound whitelist for a codebase.')
    ap.add_argument('--input', required=True, help='Path to codebase root')
    ap.add_argument('--limit-loc', type=int, default=20000, help='Total LOC budget for whitelist')
    ap.add_argument('--output', default='whitelist.txt', help='Output whitelist file')
    ap.add_argument('--enable-llm', action='store_true', help='Use LLM to rerank candidates')
    ap.add_argument('--model', default='gpt-4o-mini', help='LLM model to use if enabled')
    ap.add_argument('--max-llm-items', type=int, default=300, help='Max candidates to send to LLM')
    ap.add_argument('--print-summary', action='store_true', help='Print selection summary to stdout')
    ap.add_argument('--use-cloc', action='store_true', help='Use cloc for repo LOC totals if available')
    ap.add_argument('--config', help='JSON file to override weights/patterns')
    ap.add_argument('--verbose', action='store_true', help='Verbose progress output')
    ap.add_argument('--no-progress', action='store_true', help='Disable progress updates')
    ap.add_argument('--only-ext', help='Comma-separated list of extensions to include (e.g., "go,js,ts"). Case-insensitive, dot optional.')
    ap.add_argument('--output-format', choices=['csv','lines'], default='csv', help='Output format for whitelist (default: csv)')
    args = ap.parse_args(argv)

    root = Path(args.input).resolve()
    if not root.exists() or not root.is_dir():
        print(f"Input path not found or not a directory: {root}")
        return 2

    if args.config:
        apply_overrides(args.config)

    only_ext_set: Optional[Set[str]] = None
    if args.only_ext:
        parts = [s.strip().lower() for s in str(args.only_ext).split(',') if s.strip()]
        only_ext_set = { (p if p.startswith('.') else f'.{p}') for p in parts }
        if args.verbose:
            print(f"[filter] Restricting to extensions: {', '.join(sorted(only_ext_set))}")

    items = build_candidates(root, workers=os.cpu_count() or 8, verbose=args.verbose, progress=not args.no_progress, only_ext=only_ext_set)
    if not items:
        print("No candidate files found.")
        return 1

    ranked = heuristic_rank(items)
    by_rel = {it.rel: it for it in ranked}

    if args.enable_llm:
        if args.verbose:
            print(f"[llm] Reranking top {min(len(ranked), args.max_llm_items)} candidates with {args.model}… (1 prompt)")
        prioritized = llm_rerank(ranked, model=args.model, max_items=args.max_llm_items, verbose=args.verbose)
        # Keep only those we have, in that order, then append unseen from heuristic
        seen = set()
        pruned = []
        for rel in prioritized:
            if rel in by_rel and rel not in seen:
                pruned.append(rel); seen.add(rel)
        for it in ranked:
            if it.rel not in seen:
                pruned.append(it.rel); seen.add(it.rel)
        prioritized = pruned
    else:
        prioritized = [it.rel for it in ranked]

    selected, total = compile_whitelist(prioritized, by_rel, args.limit_loc)
    out_path = Path(args.output).resolve()
    if args.output_format == 'csv':
        out_text = ','.join(selected) + ('\n' if selected else '')
    else:
        out_text = '\n'.join(selected) + ('\n' if selected else '')
    out_path.write_text(out_text, encoding='utf-8')

    if args.print_summary:
        # Group by ext
        by_ext: dict[str, int] = {}
        for rel in selected:
            ext = Path(rel).suffix.lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1
        print(f"Wrote {len(selected)} files ({total} LOC) to {out_path}")
        print("Selected files by extension (count):")
        for ext, cnt in sorted(by_ext.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {ext or '(no ext)'}: {cnt}")
        # Repo totals (cloc or approximate)
        repo_total = 0
        repo_by = {}
        if args.use_cloc:
            try:
                repo_total, repo_by_lang = run_cloc(root)
                print(f"Repo total CLOC (cloc): {repo_total}")
                # Map selected ext totals to languages roughly by ext suffix in LANG_WEIGHTS
                # If cloc used, still show per-ext included LOC below
            except Exception as e:
                print(f"[cloc] {e}; falling back to approximate totals")
                repo_total, repo_by = approx_repo_loc_by_ext(root, only_ext=None)
        else:
            repo_total, repo_by = approx_repo_loc_by_ext(root, only_ext=None)
            print(f"Repo total LOC (approx): {repo_total}")
        # Included LOC per ext
        included_by_ext: dict[str, int] = {}
        for rel in selected:
            ext = Path(rel).suffix.lower()
            included_by_ext[ext] = included_by_ext.get(ext, 0) + (by_rel[rel].loc if rel in by_rel else 0)
        print("Included LOC per extension (and coverage vs repo totals when available):")
        for ext, inc in sorted(included_by_ext.items(), key=lambda x: (-x[1], x[0])):
            repo_ext_total = repo_by.get(ext, 0)
            if repo_ext_total > 0:
                pct = int(inc * 100 / max(1, repo_ext_total))
                print(f"  {ext or '(no ext)'}: {inc} / {repo_ext_total} LOC ({pct}%)")
            else:
                print(f"  {ext or '(no ext)'}: {inc} LOC (repo total unknown)")

    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))