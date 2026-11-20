"""JobFit AI - Streamlit web app.

Run:  streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from jobfit.analyzer import JobFitAnalyzer
from jobfit.config import settings
from jobfit.embeddings import FastEmbedder
from jobfit.llm import LLMError, OpenAICompatibleLLM
from jobfit.parsing import UnsupportedFileError, extract_text
from jobfit.report import score_label, to_markdown

SAMPLES = Path(__file__).parent / "samples"

st.set_page_config(page_title="JobFit AI", page_icon="🎯", layout="wide")


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~130 MB)...")
def get_embedder(model_name: str) -> FastEmbedder:
    return FastEmbedder(model_name)


def init_state() -> None:
    for key, default in {
        "analyzer": None,
        "report": None,
        "bullets": None,
        "interview": None,
        "chat_history": [],
    }.items():
        st.session_state.setdefault(key, default)


init_state()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Settings")
    api_key = st.text_input(
        "LLM API key",
        value=settings.api_key,
        type="password",
        help="Free key from console.groq.com. Never commit keys to GitHub; use .env.",
    )
    base_url = st.text_input("API base URL", value=settings.base_url)
    model = st.text_input("Model", value=settings.model)
    st.caption("Any OpenAI-compatible API works: Groq, OpenAI, OpenRouter, or local Ollama.")
    st.divider()
    st.markdown(
        "**How it works**\n"
        "1. LLM extracts structured requirements from the JD\n"
        "2. Embeddings semantically match each skill to resume lines\n"
        "3. LLM writes a grounded fit assessment\n"
        "4. RAG lets you chat with your resume"
    )

# ---------------- Inputs ----------------
st.title("🎯 JobFit AI")
st.caption("Match your resume to any job description using LLMs, embeddings and RAG.")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Your resume")
    upload = st.file_uploader("Upload PDF, DOCX or TXT", type=["pdf", "docx", "txt", "md"])
    use_sample_resume = st.checkbox("Use sample resume", value=upload is None)
with col2:
    st.subheader("2. Job description")
    sample_jd = (SAMPLES / "job_description.txt").read_text(encoding="utf-8")
    use_sample_jd = st.checkbox("Use sample job description", value=True)
    jd_text = st.text_area(
        "Paste the job description", value=sample_jd if use_sample_jd else "", height=220
    )

if st.button("🔍 Analyze fit", type="primary", use_container_width=True):
    try:
        if upload is not None and not use_sample_resume:
            resume_text = extract_text(upload.getvalue(), upload.name)
        else:
            resume_text = (SAMPLES / "resume.txt").read_text(encoding="utf-8")
        llm = OpenAICompatibleLLM(api_key=api_key, base_url=base_url, model=model)
        analyzer = JobFitAnalyzer(llm=llm, embedder=get_embedder(settings.embedding_model))
        with st.spinner("Reading the job description and your resume..."):
            analyzer.load(resume_text, jd_text)
            report = analyzer.analyze()
        st.session_state.update(
            analyzer=analyzer, report=report, bullets=None, interview=None, chat_history=[]
        )
    except (UnsupportedFileError, ValueError) as err:
        st.error(str(err))
    except LLMError as err:
        st.error(f"LLM error: {err}")

analyzer: JobFitAnalyzer | None = st.session_state.analyzer
report = st.session_state.report

if report is None:
    st.info("Add your resume and a job description, then click **Analyze fit**.")
    st.stop()

# ---------------- Results ----------------
tab_report, tab_tailor, tab_interview, tab_chat = st.tabs(
    ["📊 Fit report", "✍️ Tailor resume", "🎤 Interview prep", "💬 Chat with resume"]
)

with tab_report:
    a = report.assessment
    m1, m2, m3 = st.columns(3)
    m1.metric("Overall score", f"{report.overall_score}/100", score_label(report.overall_score))
    m2.metric("Skill coverage", f"{report.skill_coverage:.0f}%")
    m3.metric("LLM assessment", f"{a.llm_fit_score}/100")
    st.progress(report.overall_score / 100)

    st.markdown(f"**Role:** {report.requirements.job_title} · {report.requirements.seniority}")
    st.write(a.summary)

    df = pd.DataFrame(
        [
            {
                "Skill": m.skill,
                "Type": "Required" if m.required else "Nice to have",
                "Found": "✅" if m.matched else "❌",
                "Similarity": m.similarity,
                "Evidence from resume": m.evidence,
            }
            for m in report.skill_matches
        ]
    )
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={"Similarity": st.column_config.ProgressColumn(min_value=0, max_value=1)},
    )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 💪 Strengths")
        for s in a.strengths:
            st.markdown(f"- {s}")
    with c2:
        st.markdown("#### 🧩 Gaps")
        for g in a.gaps:
            st.markdown(f"- {g}")

    st.markdown("#### 🗺️ Action plan")
    icons = {"high": "🔴", "medium": "🟠", "low": "🟢"}
    for rec in a.recommendations:
        st.markdown(f"{icons[rec.priority]} **{rec.action}**  \n{rec.why}")

    st.download_button(
        "⬇️ Download report (Markdown)",
        to_markdown(report),
        file_name="jobfit_report.md",
        mime="text/markdown",
    )

with tab_tailor:
    st.write("Rewrite your strongest resume bullets to target this job, without inventing anything.")
    n = st.slider("Number of bullets", 3, 8, 5)
    if st.button("Generate tailored bullets"):
        with st.spinner("Rewriting..."):
            try:
                st.session_state.bullets = analyzer.tailor_bullets(n=n)
            except LLMError as err:
                st.error(str(err))
    if st.session_state.bullets:
        for b in st.session_state.bullets.bullets:
            with st.container(border=True):
                st.caption("Before")
                st.write(b.original)
                st.caption("After")
                st.markdown(f"**{b.improved}**")
                if b.keywords_added:
                    st.caption("Keywords: " + ", ".join(b.keywords_added))

with tab_interview:
    st.write("Questions tailored to this role, your projects, and your gaps.")
    if st.button("Generate interview questions"):
        with st.spinner("Thinking like an interviewer..."):
            try:
                st.session_state.interview = analyzer.interview_prep(gaps=report.assessment.gaps)
            except LLMError as err:
                st.error(str(err))
    if st.session_state.interview:
        for i, q in enumerate(st.session_state.interview.questions, 1):
            with st.expander(f"{i}. [{q.type.upper()}] {q.question}"):
                st.markdown(f"**What they're testing:** {q.what_they_test}")
                st.markdown(f"**How to answer:** {q.answer_tips}")

with tab_chat:
    st.write("Ask anything about your resume vs this job. Answers are grounded in your resume (RAG).")
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
    question = st.chat_input("e.g. Which of my projects best shows LLM experience?")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            messages = analyzer.build_chat_messages(question, st.session_state.chat_history)
            try:
                answer = st.write_stream(analyzer.llm.stream(messages))
            except Exception as err:  # streaming errors from the API
                answer = f"Error: {err}"
                st.error(answer)
        st.session_state.chat_history += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
