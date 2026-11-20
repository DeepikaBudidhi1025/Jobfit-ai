"""Pydantic models that define the structured outputs we ask the LLM for."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class JobRequirements(BaseModel):
    job_title: str = Field(description="Role title from the job description")
    seniority: str = Field(description="e.g. Fresher, Junior, Mid, Senior")
    must_have_skills: list[str] = Field(description="Required skills, each 1-4 words")
    nice_to_have_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list, description="Key duties, short")


class SkillMatch(BaseModel):
    skill: str
    required: bool
    matched: bool
    similarity: float
    evidence: str = ""


class Recommendation(BaseModel):
    priority: Literal["high", "medium", "low"]
    action: str
    why: str


class FitAssessment(BaseModel):
    summary: str = Field(description="2-3 sentence honest overview of fit")
    strengths: list[str]
    gaps: list[str]
    recommendations: list[Recommendation]
    llm_fit_score: int = Field(ge=0, le=100, description="Holistic fit 0-100")


class AnalysisReport(BaseModel):
    requirements: JobRequirements
    skill_matches: list[SkillMatch]
    skill_coverage: float  # 0-100, computed with embeddings
    assessment: FitAssessment
    overall_score: int  # blend of embedding coverage and LLM judgement


class RewrittenBullet(BaseModel):
    original: str
    improved: str
    keywords_added: list[str] = Field(default_factory=list)


class TailoredBullets(BaseModel):
    bullets: list[RewrittenBullet]


class InterviewQuestion(BaseModel):
    question: str
    type: Literal["technical", "project", "behavioral", "hr"]
    what_they_test: str
    answer_tips: str


class InterviewPrep(BaseModel):
    questions: list[InterviewQuestion]
