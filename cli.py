"""Command-line version of JobFit AI.

Usage:
    python cli.py --resume samples/resume.txt --jd samples/job_description.txt
    python cli.py --resume my_resume.pdf --jd jd.txt --out report.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

from jobfit.analyzer import JobFitAnalyzer
from jobfit.config import settings
from jobfit.embeddings import FastEmbedder
from jobfit.llm import OpenAICompatibleLLM
from jobfit.parsing import extract_text
from jobfit.report import to_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a resume against a job description.")
    parser.add_argument("--resume", required=True, help="Path to resume (PDF, DOCX or TXT)")
    parser.add_argument("--jd", required=True, help="Path to job description text file")
    parser.add_argument("--out", help="Optional path to save the Markdown report")
    args = parser.parse_args()

    analyzer = JobFitAnalyzer(llm=OpenAICompatibleLLM(), embedder=FastEmbedder(settings.embedding_model))
    analyzer.load(extract_text(args.resume), Path(args.jd).read_text(encoding="utf-8"))
    report = analyzer.analyze()
    md = to_markdown(report)

    print(md)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"Saved report to {args.out}")


if __name__ == "__main__":
    main()
