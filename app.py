import base64
import json
import time
from datetime import datetime
from typing import List, Literal, Optional

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ==========================================
# 1. CONFIGURATION & PYDANTIC SCHEMAS
# ==========================================
st.set_page_config(page_title="AMC 8 AI Practice & Coach", layout="wide")

if "OPENAI_API_KEY" not in st.secrets:
    st.error("Missing `OPENAI_API_KEY` in `.streamlit/secrets.toml`.")
    st.stop()

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
MODEL_NAME = st.secrets.get("OPENAI_MODEL", "gpt-4o")

ALL_TOPICS = [
    "Arithmetic & Proportional Reasoning",
    "Elementary Algebra",
    "Elementary Geometry",
    "Number Theory",
    "Counting, Probability & Statistics",
]

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
# 2. SESSION STATE & GITHUB SYNC HELPERS
# ==========================================
def fetch_kb_from_github() -> Optional[dict]:
    """Attempts to automatically load knowledge_base.json from GitHub on startup."""
    if "GITHUB_TOKEN" not in st.secrets:
        return None
    
    try:
        token = st.secrets["GITHUB_TOKEN"]
        owner = "pragnyavemprala"
        repo = "AMC8Trainer"
        path = "knowledge_base.json"
        
        url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
        }
        
        resp = requests.get(url, headers=headers)
        if resp.status_code == 200:
            file_data = resp.json()
            decoded_bytes = base64.b64decode(file_data["content"])
            return json.loads(decoded_bytes.decode("utf-8"))
    except Exception:
        pass
    return None

def init_state():
    github_kb = fetch_kb_from_github()
    
    default_kb = github_kb if github_kb else {
        "question_bank": [],
        "attempt_history": [],
        "self_confidence": {t: 50 for t in ALL_TOPICS},
        "total_attempts_counter": 0,
    }

    defaults = {
        "kb": default_kb,
        "current_exam": None,
        "user_answers": {},
        "start_time": None,
        "exam_submitted": False,
        "exam_source": "AI Generated",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    st.session_state.kb.setdefault("question_bank", [])
    st.session_state.kb.setdefault("attempt_history", [])
    st.session_state.kb.setdefault("self_confidence", {t: 50 for t in ALL_TOPICS})
    st.session_state.kb.setdefault("total_attempts_counter", len(st.session_state.kb["attempt_history"]))


init_state()


def push_kb_to_github(json_content_str: str) -> bool:
    """Commits and pushes the updated knowledge base JSON directly to GitHub."""
    if "GITHUB_TOKEN" not in st.secrets:
        st.warning("`GITHUB_TOKEN` missing in secrets. Cannot sync to GitHub automatically.")
        return False

    token = st.secrets["GITHUB_TOKEN"]
    owner = "pragnyavemprala"
    repo = "AMC8Trainer"
    path = "knowledge_base.json"
    branch = "main"

    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }

    get_resp = requests.get(url, headers=headers)
    sha = None
    if get_resp.status_code == 200:
        sha = get_resp.json().get("sha")

    encoded_content = base64.b64encode(json_content_str.encode("utf-8")).decode("utf-8")

    payload = {
        "message": f"Auto-update Knowledge Base - Attempt #{st.session_state.kb['total_attempts_counter']}",
        "content": encoded_content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(url, headers=headers, json=payload)
    return put_resp.status_code in [200, 201]


def compute_topic_metrics() -> pd.DataFrame:
    rows = []
    all_results = [
        r for att in st.session_state.kb.get("attempt_history", []) for r in att["results"]
    ]
    bank = st.session_state.kb.get("question_bank", [])

    for topic in ALL_TOPICS:
        topic_attempts = [r for r in all_results if r["topic"] == topic]
        attempted_count = len(topic_attempts)
        correct_count = sum(1 for r in topic_attempts if r["is_correct"])
        bank_count = sum(1 for q in bank if q["topic"] == topic)

        if attempted_count > 0:
            confidence = round((correct_count / attempted_count) * 100, 1)
            source = "Based on Test Accuracy"
        else:
            confidence = st.session_state.kb["self_confidence"].get(topic, 50)
            source = "Self-Rated Baseline"

        rows.append(
            {
                "Topic": topic,
                "Questions Attempted": attempted_count,
                "Correct Answers": correct_count,
                "Past PDF Questions": bank_count,
                "Total Exposure": attempted_count + bank_count,
                "Confidence (%)": confidence,
                "Metric Source": source,
            }
        )
    return pd.DataFrame(rows)


def get_weak_subtopics() -> List[str]:
    stats = {}
    for attempt in st.session_state.kb.get("attempt_history", []):
        for q in attempt["results"]:
            sub = q["subtopic"]
            stats.setdefault(sub, {"correct": 0, "total": 0})
            stats[sub]["total"] += 1
            if q["is_correct"]:
                stats[sub]["correct"] += 1

    return [
        sub
        for sub, data in stats.items()
        if data["total"] > 0 and (data["correct"] / data["total"]) < 0.70
    ]


# ==========================================
# 3. OPENAI GENERATION & FULL-PDF SOLVING
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
    """Extracts all 25 questions from a past AMC 8 PDF, double-checking answers and solutions."""
    reader = PdfReader(uploaded_file)
    full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

    prompt = f"""
    You are an expert AMC 8 math competition solver and grader.
    Below is raw text extracted from a past official AMC 8 exam PDF (which contains 25 questions).
    1. You MUST parse all 25 questions (numbered 1 through 25) found in the text without skipping any.
    2. Clean up any broken formatting and format math using LaTeX ($...$).
    3. Solve each problem rigorously step-by-step. Double-check your arithmetic and logic twice.
    4. Determine the exact correct option letter (A, B, C, D, or E) and verify that the option text corresponds perfectly to it.
    5. Classify each problem's official AMC 8 topic, specific subtopic, and difficulty tier.

    RAW PDF TEXT:
    {full_text[:35000]}
    """

    response = client.beta.chat.completions.parse(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": "You are a precise mathematics competition solver. You must extract and solve all 25 AMC 8 problems accurately, double-checking every answer.",
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
    for attempt in st.session_state.kb.get("attempt_history", []):
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
    Analyze this student's AMC 8 attempt history and their imported past-paper question bank size ({len(st.session_state.kb.get('question_bank', []))} solved past questions available).
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
# 4. EXAM GRADING & ATTEMPT INCREMENT LOGIC
# ==========================================
def submit_exam():
    if st.session_state.exam_submitted or not st.session_state.current_exam:
        return

    st.session_state.kb["total_attempts_counter"] = st.session_state.kb.get("total_attempts_counter", 0) + 1
    current_attempt_num = st.session_state.kb["total_attempts_counter"]

    elapsed_sec = int(time.time() - st.session_state.start_time)
    results = []
    score = 0

    for q in st.session_state.current_exam:
        q_num = q["question_number"]
        chosen = st.session_state.user_answers.get(q_num, None)
        
        correct_ans = str(q["correct_answer"]).strip().upper()
        if len(correct_ans) > 1 and correct_ans[0] in ["A", "B", "C", "D", "E"]:
            correct_ans = correct_ans[0]

        is_correct = False
        if chosen and chosen.strip().upper() == correct_ans:
            is_correct = True
        else:
            options_map = {
                "A": str(q.get("option_a", "")).strip(),
                "B": str(q.get("option_b", "")).strip(),
                "C": str(q.get("option_c", "")).strip(),
                "D": str(q.get("option_d", "")).strip(),
                "E": str(q.get("option_e", "")).strip(),
            }
            chosen_text = options_map.get(str(chosen).strip().upper(), "")
            correct_text = options_map.get(correct_ans, "")
            if chosen_text and chosen_text == correct_text:
                is_correct = True

        if is_correct:
            score += 1

        results.append(
            {
                **q,
                "correct_answer": correct_ans,
                "user_answer": chosen if chosen else "Unanswered",
                "is_correct": is_correct,
            }
        )
