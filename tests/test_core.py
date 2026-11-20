from __future__ import annotations

import pytest

from conftest import ASSESSMENT, REQUIREMENTS, FakeLLM
from jobfit.analyzer import JobFitAnalyzer
from jobfit.chunking import chunk_text
from jobfit.embeddings import VectorStore
from jobfit.llm import LLMError, complete_json, extract_json
from jobfit.parsing import UnsupportedFileError, clean_text, extract_text
from jobfit.report import score_label, to_markdown
from jobfit.schemas import JobRequirements, SkillMatch


# ---------- chunking ----------
def test_chunks_respect_size_and_keep_all_text():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(10))
    chunks = chunk_text(text, chunk_size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    for i in range(10):
        assert any(f"Paragraph {i}." in c for c in chunks)


def test_chunking_edge_cases():
    assert chunk_text("   ") == []
    assert chunk_text("short text") == ["short text"]
    with pytest.raises(ValueError):
        chunk_text("x", chunk_size=100, overlap=100)
    long_word = "a" * 1200
    assert all(len(c) <= 500 for c in chunk_text(long_word, 500, 50))


# ---------- parsing ----------
def test_extract_text_from_txt_bytes():
    assert extract_text(b"Hello   world\r\n\r\n\r\nBye", "cv.txt") == "Hello world\n\nBye"


def test_extract_text_rejects_unknown_and_empty():
    with pytest.raises(UnsupportedFileError):
        extract_text(b"data", "cv.png")
    with pytest.raises(UnsupportedFileError):
        extract_text(b"   ", "cv.txt")


def test_extract_text_from_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "cv.docx"
    doc = docx.Document()
    doc.add_paragraph("Python developer")
    doc.save(path)
    assert "Python developer" in extract_text(path)


def test_clean_text_normalises_whitespace():
    assert clean_text("a \tb\n\n\n\nc") == "a b\n\nc"


# ---------- JSON handling ----------
def test_extract_json_handles_fences_and_prose():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('Here you go: {"a": {"b": 2}} thanks') == {"a": {"b": 2}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_complete_json_retries_after_bad_output():
    llm = FakeLLM(["not json at all", {"job_title": "x"}, REQUIREMENTS])
    result = complete_json(llm, [{"role": "user", "content": "hi"}], JobRequirements)
    assert result.job_title == "Associate GenAI Engineer"
    assert len(llm.calls) == 3
    assert "Error" in llm.calls[-1][-1]["content"]  # error was fed back to the model


def test_complete_json_gives_up():
    llm = FakeLLM(["bad"] * 3)
    with pytest.raises(LLMError):
        complete_json(llm, [], JobRequirements)


# ---------- vector store ----------
def test_vector_store_returns_most_relevant(embedder):
    store = VectorStore(embedder)
    store.add(["python pandas data cleaning", "cooking recipes and food", "docker containers deployment"])
    top = store.search("python data", k=1)[0]
    assert top.text == "python pandas data cleaning"
    assert len(store) == 3
    assert store.search("", k=2) == []


# ---------- analyzer pipeline ----------
def test_full_analysis_pipeline(resume_text, jd_text, embedder):
    llm = FakeLLM([REQUIREMENTS, ASSESSMENT])
    analyzer = JobFitAnalyzer(llm=llm, embedder=embedder)
    analyzer.load(resume_text, jd_text)
    report = analyzer.analyze()

    found = {m.skill: m.matched for m in report.skill_matches}
    assert found["Python"] and found["SQL"]
    assert not found["Docker"] and not found["FastAPI"]
    # 2 of 3 required (weight 1) + 0 of 1 optional (weight 0.5) -> 2 / 3.5
    assert report.skill_coverage == pytest.approx(57.1, abs=0.1)
    assert report.overall_score == round(0.5 * report.skill_coverage + 0.5 * 70)
    # The assessment prompt is grounded in the skill table and resume
    assess_prompt = llm.calls[1][1]["content"]
    assert "Docker [required]: NOT FOUND" in assess_prompt
    assert "PDF Question Answering Chatbot" in assess_prompt


def test_requirements_are_cached(resume_text, jd_text, embedder):
    llm = FakeLLM([REQUIREMENTS])
    analyzer = JobFitAnalyzer(llm=llm, embedder=embedder)
    analyzer.load(resume_text, jd_text)
    analyzer.extract_requirements()
    analyzer.extract_requirements()
    assert len(llm.calls) == 1


def test_load_requires_both_inputs(embedder):
    with pytest.raises(ValueError):
        JobFitAnalyzer(llm=FakeLLM([]), embedder=embedder).load("resume", "  ")


def test_keyword_match_is_whole_word(embedder):
    analyzer = JobFitAnalyzer(llm=FakeLLM([]), embedder=embedder)
    analyzer.load("Skills: JavaScript, C++", "jd")
    reqs = JobRequirements(job_title="x", seniority="x", must_have_skills=["Java", "C++"])
    found = {m.skill: m for m in analyzer.match_skills(reqs, threshold=0.99)}
    assert not found["Java"].matched  # "Java" must not match inside "JavaScript"
    assert found["C++"].matched


def test_rag_chat_injects_relevant_context(resume_text, jd_text, embedder):
    llm = FakeLLM(["You built a FAISS-based chatbot."])
    analyzer = JobFitAnalyzer(llm=llm, embedder=embedder)
    analyzer.load(resume_text, jd_text)
    answer = analyzer.chat("Which project used FAISS vector index?")
    assert "FAISS" in answer
    system_prompt = llm.calls[0][0]["content"]
    assert "FAISS vector index" in system_prompt


def test_coverage_with_no_skills():
    assert JobFitAnalyzer.coverage([]) == 0.0
    m = SkillMatch(skill="x", required=True, matched=True, similarity=1.0)
    assert JobFitAnalyzer.coverage([m]) == 100.0


# ---------- report ----------
def test_markdown_report(resume_text, jd_text, embedder):
    analyzer = JobFitAnalyzer(llm=FakeLLM([REQUIREMENTS, ASSESSMENT]), embedder=embedder)
    analyzer.load(resume_text, jd_text)
    md = to_markdown(analyzer.analyze())
    assert md.startswith("# JobFit Report: Associate GenAI Engineer")
    assert "| Docker | Required | No |" in md
    assert "**[HIGH]** Dockerise the PDF chatbot" in md


@pytest.mark.parametrize("score,label", [(80, "Strong fit"), (60, "Good fit with gaps"), (40, "Partial fit"), (10, "Weak fit")])
def test_score_label(score, label):
    assert score_label(score) == label
