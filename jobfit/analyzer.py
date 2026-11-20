"""Core pipeline: requirement extraction -> semantic skill matching -> LLM fit assessment -> RAG chat."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from . import prompts
from .chunking import chunk_text
from .config import settings
from .embeddings import Embedder, VectorStore, cosine_matrix
from .llm import ChatModel, complete_json
from .schemas import (
    AnalysisReport,
    FitAssessment,
    InterviewPrep,
    JobRequirements,
    SkillMatch,
    TailoredBullets,
)

MUST_HAVE_WEIGHT = 1.0
NICE_TO_HAVE_WEIGHT = 0.5


def _resume_units(resume: str) -> list[str]:
    """Short, line-level units: good for finding precise evidence of a single skill."""
    units = []
    for line in resume.splitlines():
        line = line.strip(" \t-•*·")
        if len(line) >= 3:
            units.append(line)
    return units


def _keyword_hit(skill: str, text: str) -> bool:
    """Exact, case-insensitive whole-word match (handles C++, Node.js, etc.)."""
    pattern = r"(?<![\w])" + re.escape(skill.lower()) + r"(?![\w])"
    return re.search(pattern, text.lower()) is not None


@dataclass
class JobFitAnalyzer:
    llm: ChatModel
    embedder: Embedder
    resume: str = ""
    job_description: str = ""
    store: VectorStore | None = None
    requirements: JobRequirements | None = None
    _units: list[str] = field(default_factory=list)
    _unit_vecs: np.ndarray | None = None

    # ---------- setup ----------
    def load(self, resume: str, job_description: str) -> None:
        self.resume, self.job_description = resume.strip(), job_description.strip()
        if not self.resume or not self.job_description:
            raise ValueError("Both resume and job description are required.")
        self.store = VectorStore(self.embedder)
        self.store.add(chunk_text(self.resume, settings.chunk_size, settings.chunk_overlap))
        self._units = _resume_units(self.resume)
        self._unit_vecs = self.embedder.embed(self._units) if self._units else None
        self.requirements = None

    def _require_loaded(self) -> None:
        if self.store is None:
            raise RuntimeError("Call load(resume, job_description) first.")

    # ---------- step 1: structure the JD ----------
    def extract_requirements(self) -> JobRequirements:
        self._require_loaded()
        if self.requirements is None:
            messages = [
                {"role": "system", "content": prompts.SYSTEM_RECRUITER},
                {"role": "user", "content": prompts.EXTRACT_REQUIREMENTS.format(jd=self.job_description[:12000])},
            ]
            self.requirements = complete_json(self.llm, messages, JobRequirements)
        return self.requirements

    # ---------- step 2: semantic skill matching (embeddings, no LLM) ----------
    def match_skills(self, reqs: JobRequirements, threshold: float | None = None) -> list[SkillMatch]:
        self._require_loaded()
        threshold = settings.skill_match_threshold if threshold is None else threshold
        skills = [(s, True) for s in reqs.must_have_skills] + [(s, False) for s in reqs.nice_to_have_skills]
        skills = [(s.strip(), r) for s, r in skills if s.strip()]
        if not skills:
            return []

        sims = None
        if self._unit_vecs is not None:
            sims = cosine_matrix(self.embedder.embed([s for s, _ in skills]), self._unit_vecs)

        results = []
        for i, (skill, required) in enumerate(skills):
            best_sim, evidence = 0.0, ""
            if sims is not None:
                j = int(np.argmax(sims[i]))
                best_sim, evidence = float(sims[i, j]), self._units[j]
            exact = next((u for u in self._units if _keyword_hit(skill, u)), None)
            if exact:  # literal mention beats fuzzy similarity
                best_sim, evidence = max(best_sim, 1.0), exact
            results.append(
                SkillMatch(
                    skill=skill,
                    required=required,
                    matched=best_sim >= threshold,
                    similarity=round(min(best_sim, 1.0), 3),
                    evidence=evidence[:200] if best_sim >= threshold else "",
                )
            )
        return results

    @staticmethod
    def coverage(matches: list[SkillMatch]) -> float:
        """Weighted % of skills covered (must-haves count double)."""
        total = sum(MUST_HAVE_WEIGHT if m.required else NICE_TO_HAVE_WEIGHT for m in matches)
        if total == 0:
            return 0.0
        got = sum((MUST_HAVE_WEIGHT if m.required else NICE_TO_HAVE_WEIGHT) for m in matches if m.matched)
        return round(100 * got / total, 1)

    # ---------- step 3: LLM assessment grounded in retrieved evidence ----------
    def assess(self, reqs: JobRequirements, matches: list[SkillMatch]) -> FitAssessment:
        table = "\n".join(
            f"- {m.skill} [{'required' if m.required else 'nice-to-have'}]: "
            f"{'FOUND' if m.matched else 'NOT FOUND'} (similarity {m.similarity:.2f})"
            for m in matches
        )
        query = " ".join(reqs.must_have_skills + reqs.responsibilities[:5])
        evidence = "\n---\n".join(r.text for r in self.store.search(query, k=settings.top_k))
        messages = [
            {"role": "system", "content": prompts.SYSTEM_RECRUITER},
            {
                "role": "user",
                "content": prompts.ASSESS_FIT.format(
                    title=reqs.job_title,
                    seniority=reqs.seniority,
                    skill_table=table or "(no skills extracted)",
                    evidence=evidence or "(none)",
                    resume=self.resume[:12000],
                ),
            },
        ]
        return complete_json(self.llm, messages, FitAssessment)

    def analyze(self) -> AnalysisReport:
        """Full pipeline. Overall score = 50% embedding skill coverage + 50% LLM judgement."""
        reqs = self.extract_requirements()
        matches = self.match_skills(reqs)
        cov = self.coverage(matches)
        assessment = self.assess(reqs, matches)
        overall = round(0.5 * cov + 0.5 * assessment.llm_fit_score)
        return AnalysisReport(
            requirements=reqs,
            skill_matches=matches,
            skill_coverage=cov,
            assessment=assessment,
            overall_score=int(overall),
        )

    # ---------- extras ----------
    def tailor_bullets(self, n: int = 5) -> TailoredBullets:
        reqs = self.extract_requirements()
        skills = ", ".join(reqs.must_have_skills + reqs.nice_to_have_skills)
        messages = [
            {"role": "system", "content": prompts.SYSTEM_RECRUITER},
            {"role": "user", "content": prompts.TAILOR_BULLETS.format(skills=skills, resume=self.resume[:12000], n=n)},
        ]
        return complete_json(self.llm, messages, TailoredBullets)

    def interview_prep(self, gaps: list[str] | None = None, n: int = 8) -> InterviewPrep:
        reqs = self.extract_requirements()
        messages = [
            {"role": "system", "content": prompts.SYSTEM_RECRUITER},
            {
                "role": "user",
                "content": prompts.INTERVIEW_PREP.format(
                    n=n,
                    title=reqs.job_title,
                    skills=", ".join(reqs.must_have_skills),
                    gaps="; ".join(gaps or []) or "unknown",
                    resume=self.resume[:12000],
                ),
            },
        ]
        return complete_json(self.llm, messages, InterviewPrep)

    def build_chat_messages(self, question: str, history: list[dict] | None = None) -> list[dict]:
        """Retrieval-augmented prompt: top-k resume chunks + JD summary + recent chat history."""
        self._require_loaded()
        hits = self.store.search(question, k=settings.top_k)
        context = "\n---\n".join(h.text for h in hits) or "(no relevant resume text)"
        reqs = self.requirements
        jd_summary = (
            f"{reqs.job_title}; must-have: {', '.join(reqs.must_have_skills)}"
            if reqs
            else self.job_description[:1500]
        )
        messages = [{"role": "system", "content": prompts.RAG_SYSTEM.format(context=context, jd_summary=jd_summary)}]
        messages += (history or [])[-6:]
        messages.append({"role": "user", "content": question})
        return messages

    def chat(self, question: str, history: list[dict] | None = None) -> str:
        return self.llm.complete(self.build_chat_messages(question, history))
