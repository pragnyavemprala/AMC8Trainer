import json
import time
from datetime import datetime
from typing import List, Literal, Optional

import pandas as pd
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ==========================================
# 1. CONFIGURATION & PYDANTIC SCHEMAS
# ==========================================
st.set_page_config(page_title="AMC 8 AI Practice & Coach", layout="wide")

# Load secrets safely
if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing `OPENAI_API_KEY` in `.streamlit/secrets.toml`.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4o")

Topics = Literal[
    "Arithmetic & Proportional Reasoning",
    "Elementary Algebra",
    "Elementary Geometry",
    "Number Theory",
    "Counting, Probability & Statistics",
]


class AMC8Question(BaseModel):
    question_number: int = Field(description="Question number from 1 to 25")
    topic: Topics
    subtopic: str = Field(
        description="Specific concept, e.g., 'Pythagorean Theorem', 'Prime Factorization', 'Combinations'"
    )
    difficulty_tier: Literal["Easy (Q1-10)", "Medium (Q11-20)", "Hard (Q21-25)"]
    question_text: str = Field(
        description="Full problem statement. Use LaTeX $...$ for math expressions."
    )
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    option_e: str
    correct_answer: Literal["A", "B", "C", "D", "E"]
    step_by_step_solution: str = Field(
        description="Clear, concise competition-style mathematical solution."
    )


class AMC8Exam(BaseModel):
    questions: List[AMC8Question]


class StudyRecommendation(BaseModel):
    weakest_areas: List[str]
    actionable_study_plan: str
    recommended_focus_subtopics: List[str]


# ==========================================
# 2. SESSION STATE & KNOWLEDGE BASE HELPERS
# ==========================================
def init_state():
    defaults = {
        "kb": {
            "question_bank": [],  # Questions extracted from past PDFs (with AI solutions)
            "attempt_history": [],  # History of completed practice tests
        },
        "current_exam": None,
        "user_answers": {},
        "start_time": None,
        "exam_submitted": False,
        "exam_source": "AI Generated",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_state()


def get_weak_subtopics() -> List[str]:
    """Identify subtopics with < 70% accuracy from past attempts."""
    stats = {}
    for attempt in st.session_state.kb["attempt_history"]:
        for q in attempt["results"]:
            sub = q["subtopic"]
            stats.setdefault(sub, {"correct": 0, "total": 0})
            stats[sub]["total"] += 1
            if q["is_correct"]:
                stats[sub]["correct"] += 1

    weak = [
        sub
        for sub, data in stats.items()
        if data["total"] > 0 and (data["correct"] / data["total"]) < 0.70
    ]
    return weak


# ==========================================
# 3. OPENAI GENERATION & PDF PARSING
# ==========================================
def generate_ai_exam(focus_weak_areas: bool = True) -> AMC8Exam:
    weak_areas = get_weak_subtopics() if focus_weak_areas else []
    focus_instruction = ""
    if weak_areas:
        focus_instruction = (
            f"The student previously struggled with: {', '.join(weak_areas)}. "
            "Include several targeted reinforcement problems covering these subtopics while maintaining the official 25-question distribution."
        )

    prompt = f"""
    Create an authentic 25-question AMC 8 practice exam following official MAA standards:
    - Questions 1-10: Easy (foundational arithmetic, basic geometry, simple word problems)
    - Questions 11-20: Medium (multi-step algebra, number theory, counting/probability, area/angles)
    - Questions 21-25: Hard (creative competition problem-solving, combinatorics, spatial reasoning)
    - All problems must be solvable WITHOUT a calculator.
    - Ensure 5 distinct options (A-E) with one unambiguously correct answer.
    - Use LaTeX ($...$) for all mathematical formulas and numbers in equations.
    {focus_instruction}
    """

    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are an expert AMC 8 math competition writer and coach.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=AMC8Exam,
    )
    return response.choices[0].message.parsed


def extract_and_solve_pdf(uploaded_file) -> List[dict]:
    """Extracts text from an unsolved AMC 8 PDF, parses questions, and solves them via OpenAI."""
    reader = PdfReader(uploaded_file)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    prompt = f"""
    Below is raw text extracted from a past AMC 8 exam PDF (which does not contain solutions).
    1. Parse every multiple-choice math problem found in the text (up to 25 questions).
    2. Clean up any broken formatting and format math using LaTeX ($...$).
    3. Solve each problem rigorously step-by-step and determine the exact correct option (A, B, C, D, or E).
    4. Classify each problem's official AMC 8 topic, specific subtopic, and difficulty tier.

    RAW PDF TEXT:
    {full_text[:28000]}
    """

    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a mathematics competition solver. Solve every AMC 8 problem accurately and verify your final choice matches one of options A-E.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format=AMC8Exam,
    )
    parsed_exam = response.choices[0].message.parsed
    enriched_questions = []
    for q in parsed_exam.questions:
        q_dict = q.model_dump()
        q_dict["source_pdf"] = uploaded_file.name
        enriched_questions.append(q_dict)
    return enriched_questions


def generate_coach_feedback() -> StudyRecommendation:
    history_summary = []
    for attempt in st.session_state.kb["attempt_history"]:
        for r in attempt["results"]:
            history_summary.append(
                {
                    "topic": r["topic"],
                    "subtopic": r["subtopic"],
                    "difficulty": r["difficulty_tier"],
                    "is_correct": r["is_correct"],
                }
            )

    prompt = f"""
    Analyze this student's AMC 8 attempt history and their imported past-paper question bank size ({len(st.session_state.kb['question_bank'])} solved past questions available).
    Attempt Data: {json.dumps(history_summary)}

    Provide:
    1. Their weakest topics/subtopics.
    2. A concrete, actionable study plan for their next practice sessions.
    3. Specific subtopics to prioritize.
    """

    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "You are an encouraging, analytical AMC 8 math coach."},
            {"role": "user", "content": prompt},
        ],
        response_format=StudyRecommendation,
    )
    return response.choices[0].message.parsed


# ==========================================
# 4. EXAM GRADING & SUBMISSION LOGIC
# ==========================================
def submit_exam():
    if st.session_state.exam_submitted or not st.session_state.current_exam:
        return

    elapsed_sec = int(time.time() - st.session_state.start_time)
    results = []
    score = 0

    for q in st.session_state.current_exam:
        q_num = q["question_number"]
        chosen = st.session_state.user_answers.get(q_num, None)
        is_correct = chosen == q["correct_answer"]
        if is_correct:
            score += 1

        results.append(
            {
                **q,
                "user_answer": chosen if chosen else "Unanswered",
                "is_correct": is_correct,
            }
        )

    attempt_record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": st.session_state.exam_source,
        "score": score,
        "total_questions": len(results),
        "time_taken_seconds": min(elapsed_sec, 2400),
        "results": results,
    }

    st.session_state.kb["attempt_history"].append(attempt_record)
    st.session_state.exam_submitted = True


# ==========================================
# 5. STREAMLIT USER INTERFACE
# ==========================================
st.title("🏆 AMC 8 AI Practice & Knowledge Base")

# Sidebar: Knowledge Base Persistence (Save / Load JSON)
with st.sidebar:
    st.header("💾 Knowledge Base Sync")
    st.caption(
        "Download your Knowledge Base to save past PDFs and test attempts across sessions."
    )

    kb_json = json.dumps(st.session_state.kb, indent=2)
    st.download_button(
        label="⬇️ Download Knowledge Base (.json)",
        data=kb_json,
        file_name=f"amc8_knowledge_base_{datetime.now().strftime('%Y%m%d')}.json",
        mime="application/json",
    )

    uploaded_kb = st.file_uploader("⬆️ Restore Knowledge Base (.json)", type=["json"])
    if uploaded_kb is not None:
        if st.button("Load Uploaded Knowledge Base"):
            st.session_state.kb = json.load(uploaded_kb)
            st.success("Knowledge Base restored!")
            st.rerun()

    st.divider()
    st.metric("Past PDF Questions in Bank", len(st.session_state.kb["question_bank"]))
    st.metric("Completed Practice Tests", len(st.session_state.kb["attempt_history"]))

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📝 AMC 8 Practice Test",
        "📄 Import Past PDFs (No Solutions)",
        "🎯 Practice Imported PDFs",
        "📊 Knowledge Base & AI Coach",
    ]
)

# ------------------------------------------
# TAB 1: OFFICIAL FORMAT PRACTICE TEST
# ------------------------------------------
with tab1:
    st.subheader("Official Format: 25 Questions | 40 Minutes | No Calculator")

    col1, col2 = st.columns([2, 1])
    with col1:
        adapt_toggle = st.checkbox(
            "Adapt questions to target my weak areas from Knowledge Base", value=True
        )
    with col2:
        if st.button("🚀 Generate New 25-Question Test", type="primary"):
            with st.spinner("Generating 25 AMC 8 problems..."):
                exam = generate_ai_exam(focus_weak_areas=adapt_toggle)
                st.session_state.current_exam = [q.model_dump() for q in exam.questions]
                st.session_state.user_answers = {}
                st.session_state.start_time = time.time()
                st.session_state.exam_submitted = False
                st.session_state.exam_source = "AI Generated AMC 8"
                st.rerun()

    if st.session_state.current_exam:
        # Timer Logic (2400 seconds = 40 minutes)
        if not st.session_state.exam_submitted:
            elapsed = time.time() - st.session_state.start_time
            remaining = max(0, int(2400 - elapsed))
            mins, secs = divmod(remaining, 60)

            t_col1, t_col2 = st.columns([3, 1])
            with t_col1:
                st.info(f"⏱️ **Time Remaining:** {mins:02d}:{secs:02d} (Updates on interaction)")
            with t_col2:
                if st.button("🔄 Refresh Timer"):
                    st.rerun()

            if remaining == 0:
                st.warning("Time is up! Submitting your exam automatically...")
                submit_exam()
                st.rerun()

        st.divider()

        # Render Questions
        for q in st.session_state.current_exam:
            q_num = q["question_number"]
            st.markdown(f"**Problem {q_num}** — *{q['topic']} ({q['subtopic']})*")
            st.markdown(q["question_text"])

            options_map = {
                "A": f"(A) {q['option_a']}",
                "B": f"(B) {q['option_b']}",
                "C": f"(C) {q['option_c']}",
                "D": f"(D) {q['option_d']}",
                "E": f"(E) {q['option_e']}",
            }

            if not st.session_state.exam_submitted:
                choice = st.radio(
                    f"Select answer for Problem {q_num}:",
                    options=["A", "B", "C", "D", "E"],
                    format_func=lambda x: options_map[x],
                    index=None,
                    key=f"q_radio_{q_num}",
                    label_visibility="collapsed",
                )
                if choice:
                    st.session_state.user_answers[q_num] = choice
            else:
                user_ans = st.session_state.user_answers.get(q_num, "Unanswered")
                correct_ans = q["correct_answer"]
                if user_ans == correct_ans:
                    st.success(f"✅ Your Answer: ({user_ans}) — Correct!")
                else:
                    st.error(
                        f"❌ Your Answer: ({user_ans}) | Correct Answer: ({correct_ans})"
                    )
                with st.expander("View Step-by-Step Solution"):
                    st.markdown(q["step_by_step_solution"])

            st.divider()

        # Submit & Download Controls
        if not st.session_state.exam_submitted:
            if st.button("✅ Submit Exam", type="primary"):
                submit_exam()
                st.rerun()
        else:
            latest_attempt = st.session_state.kb["attempt_history"][-1]
            st.success(
                f"### Final Score: {latest_attempt['score']} / {latest_attempt['total_questions']}"
            )

            # Downloadable File of Questions, User Answers & Solutions
            export_payload = json.dumps(latest_attempt, indent=2)
            st.download_button(
                label="📥 Download Test Questions, Your Answers & Solutions (.json)",
                data=export_payload,
                file_name=f"amc8_attempt_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                type="primary",
            )

# ------------------------------------------
# TAB 2: IMPORT PAST PDFs WITHOUT SOLUTIONS
# ------------------------------------------
with tab2:
    st.subheader("Upload Past AMC 8 Question Papers (PDF)")
    st.write(
        "Upload previous years' AMC 8 PDFs that don't have answer keys. "
        "The AI will extract every question, solve it step-by-step, tag its topic/subtopic, and store it in your Knowledge Base."
    )

    pdf_files = st.file_uploader(
        "Select one or more AMC 8 PDF files", type=["pdf"], accept_multiple_files=True
    )

    if pdf_files and st.button("🧠 Extract & Solve PDFs into Knowledge Base", type="primary"):
        for pdf in pdf_files:
            with st.spinner(f"Reading and solving {pdf.name}..."):
                solved_qs = extract_and_solve_pdf(pdf)
                st.session_state.kb["question_bank"].extend(solved_qs)
                st.success(f"Added {len(solved_qs)} solved questions from **{pdf.name}**!")

    if st.session_state.kb["question_bank"]:
        st.divider()
        st.write("### Preview of Imported & Solved Past Questions")
        bank_df = pd.DataFrame(st.session_state.kb["question_bank"])[
            ["source_pdf", "question_number", "topic", "subtopic", "correct_answer"]
        ]
        st.dataframe(bank_df, use_container_width=True)

# ------------------------------------------
# TAB 3: PRACTICE FROM IMPORTED PDF BANK
# ------------------------------------------
with tab3:
    st.subheader("Take a Custom Test from Your Imported Past PDFs")
    bank = st.session_state.kb["question_bank"]

    if not bank:
        st.info("Upload past AMC 8 PDFs in the **Import Past PDFs** tab first.")
    else:
        sources = sorted(list({q["source_pdf"] for q in bank}))
        selected_source = st.selectbox("Choose a Past Paper to Practice:", sources)
        if st.button("Start Timed Practice with Selected Paper", type="primary"):
            selected_qs = [q for q in bank if q["source_pdf"] == selected_source]
            st.session_state.current_exam = selected_qs
            st.session_state.user_answers = {}
            st.session_state.start_time = time.time()
            st.session_state.exam_submitted = False
            st.session_state.exam_source = f"Past PDF: {selected_source}"
            st.success("Loaded! Switch to the **AMC 8 Practice Test** tab to begin.")

# ------------------------------------------
# TAB 4: KNOWLEDGE BASE ANALYTICS & AI COACH
# ------------------------------------------
with tab4:
    st.subheader("Performance Analytics & Personalized Study Plan")

    attempts = st.session_state.kb["attempt_history"]
    if not attempts:
        st.info("Complete at least one practice test to unlock analytics and AI coaching.")
    else:
        # Flatten all question results
        all_results = [r for att in attempts for r in att["results"]]
        df = pd.DataFrame(all_results)

        # Topic breakdown
        topic_stats = (
            df.groupby("topic")
            .agg(
                Questions_Attempted=("is_correct", "count"),
                Correct=("is_correct", "sum"),
            )
            .reset_index()
        )
        topic_stats["Accuracy (%)"] = (
            topic_stats["Correct"] / topic_stats["Questions_Attempted"] * 100
        ).round(1)

        st.write("### Accuracy by AMC 8 Topic")
        st.dataframe(topic_stats, use_container_width=True)

        # Subtopic breakdown
        subtopic_stats = (
            df.groupby(["topic", "subtopic"])
            .agg(
                Attempted=("is_correct", "count"),
                Correct=("is_correct", "sum"),
            )
            .reset_index()
        )
        subtopic_stats["Accuracy (%)"] = (
            subtopic_stats["Correct"] / subtopic_stats["Attempted"] * 100
        ).round(1)
        st.write("### Granular Subtopic Mastery")
        st.dataframe(
            subtopic_stats.sort_values("Accuracy (%)"), use_container_width=True
        )

        st.divider()
        if st.button("🎓 Generate AI Study Recommendations", type="primary"):
            with st.spinner("Analyzing your Knowledge Base and weak spots..."):
                rec = generate_coach_feedback()
                st.markdown("#### 🎯 Priority Subtopics to Focus On")
                for sub in rec.recommended_focus_subtopics:
                    st.markdown(f"- **{sub}**")
                st.markdown("#### 📋 Recommended Study Plan")
                st.write(rec.actionable_study_plan)
