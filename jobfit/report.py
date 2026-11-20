"""Turn an AnalysisReport into a downloadable Markdown report."""

from __future__ import annotations

from datetime import date

from .schemas import AnalysisReport


def score_label(score: float) -> str:
    if score >= 75:
        return "Strong fit"
    if score >= 55:
        return "Good fit with gaps"
    if score >= 35:
        return "Partial fit"
    return "Weak fit"


def to_markdown(report: AnalysisReport) -> str:
    r, a = report.requirements, report.assessment
    lines = [
        f"# JobFit Report: {r.job_title}",
        f"_Generated {date.today().isoformat()} by JobFit AI_",
        "",
        f"**Overall score: {report.overall_score}/100 ({score_label(report.overall_score)})**",
        f"- Skill coverage (semantic match): {report.skill_coverage:.0f}%",
        f"- Recruiter-style LLM assessment: {a.llm_fit_score}/100",
        "",
        "## Summary",
        a.summary,
        "",
        "## Skills",
        "| Skill | Type | Found | Similarity | Evidence |",
        "|---|---|---|---|---|",
    ]
    for m in report.skill_matches:
        ev = m.evidence.replace("|", "/")
        lines.append(
            f"| {m.skill} | {'Required' if m.required else 'Nice to have'} | "
            f"{'Yes' if m.matched else 'No'} | {m.similarity:.2f} | {ev} |"
        )
    lines += ["", "## Strengths", *[f"- {s}" for s in a.strengths]]
    lines += ["", "## Gaps", *[f"- {g}" for g in a.gaps]]
    lines += ["", "## Action plan"]
    for rec in a.recommendations:
        lines.append(f"- **[{rec.priority.upper()}]** {rec.action} - {rec.why}")
    return "\n".join(lines) + "\n"
