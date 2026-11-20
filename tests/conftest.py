"""Test doubles so the test suite runs offline, with no API key and no model download."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SAMPLES = Path(__file__).resolve().parents[1] / "samples"


class HashEmbedder:
    """Deterministic bag-of-words embedder: texts sharing words get similar vectors."""

    dim = 256

    def embed(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().replace(",", " ").split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        return out


class FakeLLM:
    """Returns canned replies in order and records every prompt it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls: list[list[dict]] = []

    def complete(self, messages, temperature=None):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        return reply if isinstance(reply, str) else json.dumps(reply)


REQUIREMENTS = {
    "job_title": "Associate GenAI Engineer",
    "seniority": "Fresher",
    "must_have_skills": ["Python", "SQL", "Docker"],
    "nice_to_have_skills": ["FastAPI"],
    "responsibilities": ["Build RAG pipelines"],
}

ASSESSMENT = {
    "summary": "Solid Python and RAG project, missing deployment skills.",
    "strengths": ["Built and deployed a RAG chatbot"],
    "gaps": ["No Docker experience"],
    "recommendations": [{"priority": "high", "action": "Dockerise the PDF chatbot", "why": "Docker is required"}],
    "llm_fit_score": 70,
}


@pytest.fixture
def resume_text():
    return (SAMPLES / "resume.txt").read_text(encoding="utf-8")


@pytest.fixture
def jd_text():
    return (SAMPLES / "job_description.txt").read_text(encoding="utf-8")


@pytest.fixture
def embedder():
    return HashEmbedder()
