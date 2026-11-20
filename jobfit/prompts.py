"""All prompt templates in one place, so they are easy to read, tweak and version."""

SYSTEM_RECRUITER = (
    "You are an experienced technical recruiter and career coach in India. "
    "You are honest, specific and practical. You never invent experience the candidate does not have."
)

EXTRACT_REQUIREMENTS = """Extract the hiring requirements from this job description.
Keep each skill short (1-4 words, e.g. "Python", "SQL", "LLM fine-tuning", "Docker").
Split combined items ("Python/SQL") into separate skills. Max 15 must-have skills.

JOB DESCRIPTION:
\"\"\"{jd}\"\"\""""

ASSESS_FIT = """Assess how well this candidate fits the role.

ROLE: {title} ({seniority})

SKILL MATCH RESULTS (computed by semantic search over the resume):
{skill_table}

MOST RELEVANT RESUME EXCERPTS:
{evidence}

FULL RESUME:
\"\"\"{resume}\"\"\"

Instructions:
- Base every strength on something actually in the resume.
- Gaps must be concrete (a missing skill, missing proof, weak metrics), not generic.
- Give 3-5 recommendations the candidate can act on within 2-4 weeks, highest priority first.
- llm_fit_score: your holistic 0-100 judgement for THIS role and seniority."""

TAILOR_BULLETS = """Rewrite the candidate's resume bullet points so they target this job.

JOB REQUIREMENTS: {skills}

RESUME:
\"\"\"{resume}\"\"\"

Rules:
- Pick the {n} bullets/lines (projects or experience) that matter most for this job.
- Use the format: action verb + what you built + tech used + measurable result.
- Weave in job keywords ONLY where the resume genuinely supports them.
- NEVER invent numbers, tools or experience. If no metric exists, describe scope instead.
- Each improved bullet must be under 30 words."""

INTERVIEW_PREP = """Create {n} likely interview questions for this candidate applying to this role.

ROLE: {title}
REQUIRED SKILLS: {skills}
CANDIDATE GAPS: {gaps}

RESUME:
\"\"\"{resume}\"\"\"

Mix: ~40% technical (on required skills), ~30% about the candidate's own projects,
~20% behavioral, ~10% HR. Make project questions reference specific projects in the resume.
answer_tips: 1-2 sentences on how THIS candidate should answer."""

RAG_SYSTEM = """You answer questions about a candidate's resume and a target job.
Use ONLY the context below. If the answer is not in the context, say you don't know.
Be concise. When useful, quote short phrases from the resume.

RESUME CONTEXT:
{context}

JOB DESCRIPTION SUMMARY:
{jd_summary}"""
