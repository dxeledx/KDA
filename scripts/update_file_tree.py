#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.file_tree import collect_repo_paths, render_file_tree_markdown


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output", default="FILE_TREE.md")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    repo_root = Path(args.repo_root).resolve()
    output_path = (repo_root / args.output).resolve()

    repo_paths = collect_repo_paths(repo_root)
    output_relpath = output_path.relative_to(repo_root).as_posix()
    if output_relpath not in repo_paths:
        repo_paths.append(output_relpath)

    markdown = render_file_tree_markdown(repo_paths)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Wrote {output_relpath}")


if __name__ == "__main__":
    main()
