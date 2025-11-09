#!/usr/bin/env python3
"""
Whitelist builder for Hound - generates file whitelists within LOC budget.

Based on the approach described in:
https://muellerberndt.medium.com/hunting-for-security-bugs-in-code-with-ai-agents-a-full-walkthrough-a0dc24e1adf0
"""

import argparse
import os
from pathlib import Path
from collections import defaultdict


def count_lines(file_path: Path) -> int:
    """Count lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception:
        return 0


def get_file_extension(file_path: Path) -> str:
    """Get file extension."""
    return file_path.suffix.lower()


def should_include_file(file_path: Path, extensions: set) -> bool:
    """Check if file should be included based on extension."""
    if not file_path.is_file():
        return False
    return get_file_extension(file_path) in extensions


def build_whitelist(
    input_dir: str,
    output_file: str,
    limit_loc: int = 50000,
    extensions: set = None,
    enable_ll: bool = False,
    verbose: bool = False
) -> dict:
    """
    Build a whitelist of files within LOC budget.
    
    Args:
        input_dir: Directory to scan
        output_file: Output file path
        limit_loc: Maximum lines of code
        extensions: Set of file extensions to include (default: common source extensions)
        enable_ll: Enable language-specific logic
        verbose: Print verbose output
    
    Returns:
        Dictionary with summary statistics
    """
    if extensions is None:
        # Default extensions for common languages
        extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.sol', '.rs', '.go',
            '.java', '.cpp', '.c', '.h', '.hpp', '.cs', '.rb', '.php',
            '.swift', '.kt', '.scala', '.ml', '.hs', '.clj', '.ex', '.exs',
            '.vy', '.cairo', '.move'
        }
    
    input_path = Path(input_dir).resolve()
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    # Collect all relevant files with their sizes
    files_with_sizes = []
    total_loc = 0
    
    for file_path in input_path.rglob('*'):
        if not should_include_file(file_path, extensions):
            continue
        
        # Skip common directories
        skip_dirs = {'.git', '.svn', '.hg', 'node_modules', '__pycache__',
                     '.pytest_cache', '.venv', 'venv', 'env', 'dist', 'build',
                     'target', '.idea', '.vscode', '.DS_Store', 'vendor'}
        if any(skip in file_path.parts for skip in skip_dirs):
            continue
        
        loc = count_lines(file_path)
        if loc > 0:
            rel_path = file_path.relative_to(input_path)
            files_with_sizes.append((rel_path, loc))
            total_loc += loc
    
    # Sort by size (largest first) to prioritize important files
    files_with_sizes.sort(key=lambda x: x[1], reverse=True)
    
    # Select files within LOC budget
    selected_files = []
    selected_loc = 0
    
    for rel_path, loc in files_with_sizes:
        if selected_loc + loc <= limit_loc:
            selected_files.append(rel_path)
            selected_loc += loc
        else:
            # Try to fit if close to limit
            if selected_loc + loc <= limit_loc * 1.1:  # 10% tolerance
                selected_files.append(rel_path)
                selected_loc += loc
            break
    
    # Write whitelist (comma-separated format for Hound)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        # Write as comma-separated list (Hound format)
        file_list = ','.join(str(f) for f in selected_files)
        f.write(file_list)
    
    # Summary statistics
    summary = {
        'total_files_found': len(files_with_sizes),
        'total_loc_found': total_loc,
        'selected_files': len(selected_files),
        'selected_loc': selected_loc,
        'limit_loc': limit_loc,
        'coverage': (selected_loc / total_loc * 100) if total_loc > 0 else 0
    }
    
    if verbose:
        print(f"Found {summary['total_files_found']} files ({summary['total_loc_found']:,} LOC)")
        print(f"Selected {summary['selected_files']} files ({summary['selected_loc']:,} LOC)")
        print(f"Coverage: {summary['coverage']:.1f}%")
        print(f"Output: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Build file whitelist for Hound within LOC budget'
    )
    parser.add_argument('--input', '-i', required=True,
                       help='Input directory to scan')
    parser.add_argument('--output', '-o', required=True,
                       help='Output whitelist file path')
    parser.add_argument('--limit-loc', type=int, default=50000,
                       help='Maximum lines of code (default: 50000)')
    parser.add_argument('--extensions', nargs='+',
                       help='File extensions to include (default: common source extensions)')
    parser.add_argument('--enable-ll', action='store_true',
                       help='Enable language-specific logic (placeholder)')
    parser.add_argument('--print-summary', action='store_true',
                       help='Print summary statistics')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    extensions = None
    if args.extensions:
        extensions = {ext if ext.startswith('.') else f'.{ext}' for ext in args.extensions}
    
    summary = build_whitelist(
        input_dir=args.input,
        output_file=args.output,
        limit_loc=args.limit_loc,
        extensions=extensions,
        enable_ll=args.enable_ll,
        verbose=args.verbose or args.print_summary
    )
    
    if args.print_summary:
        print("\n=== Summary ===")
        print(f"Total files found: {summary['total_files_found']}")
        print(f"Total LOC found: {summary['total_loc_found']:,}")
        print(f"Selected files: {summary['selected_files']}")
        print(f"Selected LOC: {summary['selected_loc']:,}")
        print(f"LOC limit: {summary['limit_loc']:,}")
        print(f"Coverage: {summary['coverage']:.1f}%")


if __name__ == '__main__':
    main()

