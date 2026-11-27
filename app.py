import base64
import json
import time
from datetime import datetime
from typing import List, Literal, Optional

import boto3
from botocore.exceptions import ClientError
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from pypdf import PdfReader

# ==========================================
# 1. PAGE CONFIG (MUST BE FIRST STREAMLIT COMMAND)
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
# 2. DYNAMODB & GITHUB SYNC HELPERS
# ==========================================
def get_dynamodb_table():
    """Connects to Amazon DynamoDB using Streamlit secrets."""
    try:
        dynamodb = boto3.resource(
            'dynamodb',
            aws_access_key_id=st.secrets.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=st.secrets.get("AWS_SECRET_ACCESS_KEY"),
            region_name=st.secrets.get("AWS_DEFAULT_REGION", "us-east-1")
        )
        table_name = st.secrets.get("DYNAMODB_TABLE", "amc8_users_progress")
        return dynamodb.Table(table_name)
    except Exception as e:
        return None


def load_user_data_from_dynamodb(username: str) -> Optional[dict]:
    """Pulls credentials and AMC8 progress for a specific user from DynamoDB."""
    table = get_dynamodb_table()
    if not table:
        return None
    
    try:
        response = table.get_item(Key={"username": username})
        if "Item" in response:
            return response["Item"]
    except ClientError:
        pass
    return None


def save_user_data_to_dynamodb(username: str, password_hash: str, kb_data: dict):
    """Saves or updates the user's credentials and entire AMC8 progress payload in DynamoDB."""
    table = get_dynamodb_table()
    if not table:
        return False
    
    try:
        table.put_item(
            Item={
                "username": username,
                "password": password_hash,
                "question_bank": kb_data.get("question_bank", []),
                "attempt_history": kb_data.get("attempt_history", []),
                "self_confidence": kb_data.get("self_confidence", {}),
                "total_attempts_counter": kb_data.get("total_attempts_counter", 0)
            }
        )
        return True
    except ClientError:
        return False


def push_kb_to_github(json_content_str: str) -> bool:
    """Commits and pushes backup knowledge base JSON directly to GitHub."""
    if "GITHUB_TOKEN" not in st.secrets:
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
        "message": f"Auto-sync Knowledge Base - Attempt #{st.session_state.kb.get('total_attempts_counter', 0)}",
        "content": encoded_content,
        "branch": branch,
    }
    if sha:
        payload["sha"] = sha

    put_resp = requests.put(url, headers=headers, json=payload)
    return put_resp.status_code in [200, 201]


def init_state():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "current_user" not in st.session_state:
        st.session_state["current_user"] = ""

    defaults = {
        "kb": {
            "question_bank": [],
            "attempt_history": [],
            "self_confidence": {t: 50 for t in ALL_TOPICS},
            "total_attempts_counter": 0,
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

# ==========================================
# 3. RELIABLE LOGIN & REGISTRATION GATE SCREEN
# ==========================================
if not st.session_state["logged_in"]:
    st.title("🔒 AMC 8 Trainer Portal")
    
    auth_mode = st.radio("Choose action:", ["Login to existing account", "Create your account"], horizontal=True)
    st.divider()

    if auth_mode == "Login to existing account":
        st.subheader("Login")
        login_user = st.text_input("Username", key="login_user").strip()
        login_pass = st.text_input("Password", type="password", key="login_pass")
        
        if st.button("Login", type="primary"):
            user_record = load_user_data_from_dynamodb(login_user)
            if user_record and user_record.get("password") == login_pass:
                st.session_state["logged_in"] = True
                st.session_state["current_user"] = login_user
                
                # Load user progress from DynamoDB into session state
                st.session_state.kb = {
                    "question_bank": user_record.get("question_bank", []),
                    "attempt_history": user_record.get("attempt_history", []),
                    "self_confidence": user_record.get("self_confidence", {t: 50 for t in ALL_TOPICS}),
                    "total_attempts_counter": user_record.get("total_attempts_counter", len(user_record.get("attempt_history", [])))
                }
                st.success(f"Welcome back, {login_user}!")
                st.rerun()
            else:
                st.error("Invalid username or password.")
    else:
        st.subheader("Create a New Account")
        new_user = st.text_input("Choose Username", key="signup_user").strip()
        new_pass = st.text_input("Choose Password", type="password", key="signup_pass")
        confirm_pass = st.text_input("Confirm Password", type="password", key="confirm_pass")
        
        if st.button("Register Account", type="primary"):
            if not new_user or not new_pass:
                st.error("Username and password cannot be empty.")
            elif new_pass != confirm_pass:
                st.error("Passwords do not match.")
            else:
                existing_user = load_user_data_from_dynamodb(new_user)
                if existing_user:
                    st.error("Username already exists. Please choose another or log in.")
                else:
                    initial_kb = {
                        "question_bank": [],
                        "attempt_history": [],
                        "self_confidence": {t: 50 for t in ALL_TOPICS},
                        "total_attempts_counter": 0
                    }
                    save_user_data_to_dynamodb(new_user, new_pass, initial_kb)
                    
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = new_user
                    st.session_state.kb = initial_kb
                    
                    push_kb_to_github(json.dumps(initial_kb, indent=2))
                    
                    st.success(f"Account created successfully! Welcome, {new_user}!")
                    st.rerun()

    st.stop()

# ==========================================
# 4. TOPIC METRICS & AI FUNCTIONS
# ==========================================
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
# 5. EXAM GRADING & CLOUD SYNC SUBMISSION
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

    attempt_record = {
        "attempt_number": current_attempt_num,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": st.session_state.exam_source,
        "score": score,
        "total_questions": len(results),
        "time_taken_seconds": min(elapsed_sec, 2400),
        "results": results,
    }

    st.session_state.kb["attempt_history"].append(attempt_record)
    st.session_state.exam_submitted = True

    # Save progress to DynamoDB and GitHub backup
    current_user = st.session_state.get("current_user", "admin")
    user_record = load_user_data_from_dynamodb(current_user)
    current_password = user_record.get("password", "admin123") if user_record else "admin123"
    
    save_user_data_to_dynamodb(current_user, current_password, st.session_state.kb)
    push_kb_to_github(json.dumps(st.session_state.kb, indent=2))


# ==========================================
# 6. MAIN STREAMLIT UI (LOGGED IN)
# ==========================================
st.title("🏆 AMC 8 AI Practice & Knowledge Base")

with st.sidebar:
    st.header("👤 Account & Sync")
    st.write(f"User: **{st.session_state.get('current_user', 'User')}**")
    st.caption(f"Next Attempt: **#{st.session_state.kb.get('total_attempts_counter', 0) + 1}**")

    if st.button("🚪 Logout"):
        st.session_state["logged_in"] = False
        st.session_state["current_user"] = ""
        st.rerun()

    st.divider()

    kb_json = json.dumps(st.session_state.kb, indent=2)
    st.download_button(
        label="⬇️ Download Knowledge Base (.json)",
        data=kb_json,
        file_name=f"amc8_knowledge_base_attempt_{st.session_state.kb.get('total_attempts_counter', 0)}.json",
        mime="application/json",
    )

    if st.button("🚀 Force Sync KB to Cloud"):
        with st.spinner("Syncing to DynamoDB & GitHub..."):
            current_user = st.session_state.get("current_user", "admin")
            user_record = load_user_data_from_dynamodb(current_user)
            curr_pass = user_record.get("password", "admin123") if user_record else "admin123"
            
            save_user_data_to_dynamodb(current_user, curr_pass, st.session_state.kb)
            push_kb_to_github(kb_json)
            st.success("Synced successfully to cloud database!")

    st.divider()
    st.metric("Total Attempts Recorded", len(st.session_state.kb.get("attempt_history", [])))
    st.metric("Past PDF Questions in Bank", len(st.session_state.kb.get("question_bank", [])))

tab_home, tab1, tab2, tab3, tab4 = st.tabs(
    [
        "🏠 Home Dashboard",
        "📝 AMC 8 Practice Test",
        "📄 Import Past PDFs (No Solutions)",
        "🎯 Practice Imported PDFs",
        "📊 Knowledge Base & AI Coach",
    ]
)

# ------------------------------------------
# TAB 0: HOME DASHBOARD
# ------------------------------------------
with tab_home:
    st.subheader("Your AMC 8 Readiness Overview")
    metrics_df = compute_topic_metrics()

    total_attempted = int(metrics_df["Questions Attempted"].sum())
    total_correct = int(metrics_df["Correct Answers"].sum())
    overall_acc = round((total_correct / total_attempted * 100), 1) if total_attempted > 0 else 0.0
    avg_conf = round(metrics_df["Confidence (%)"].mean(), 1)

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Overall Confidence", f"{avg_conf}%")
    kpi2.metric("Practice Accuracy", f"{overall_acc}%" if total_attempted > 0 else "No tests yet")
    kpi3.metric("Questions Solved", total_attempted)
    kpi4.metric("PDF Bank Size", len(st.session_state.kb.get("question_bank", [])))

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### 🥧 AMC 8 Topic Breakdown & Confidence Share")
        fig_pie = px.pie(
            metrics_df,
            names="Topic",
            values="Confidence (%)",
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_pie.update_traces(
            textposition="inside",
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>Confidence Score: %{value}%<br>Share of Total Confidence: %{percent}<extra></extra>",
        )
        fig_pie.update_layout(showlegend=False, margin=dict(t=20, b=20, l=10, r=10))
        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        st.markdown("#### 📈 Confidence Level by Topic (0–100%)")
        fig_bar = px.bar(
            metrics_df.sort_values("Confidence (%)", ascending=True),
            x="Confidence (%)",
            y="Topic",
            orientation="h",
            text="Confidence (%)",
            color="Confidence (%)",
            color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
            range_x=[0, 105],
        )
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        fig_bar.update_layout(
            coloraxis_showscale=False,
            xaxis_title="Confidence Score (%)",
            yaxis_title="",
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with st.expander("🎚️ Set Initial Self-Confidence (Used until you take tests in a topic)"):
        st.caption(
            "Once you answer practice questions in a topic, your confidence automatically updates to your real test accuracy."
        )
        slider_cols = st.columns(len(ALL_TOPICS))
        for idx, topic in enumerate(ALL_TOPICS):
            with slider_cols[idx]:
                val = st.slider(
                    topic.split(" & ")[0],
                    min_value=10,
                    max_value=100,
                    value=int(st.session_state.kb["self_confidence"].get(topic, 50)),
                    key=f"conf_slider_{idx}",
                )
                st.session_state.kb["self_confidence"][topic] = val

# ------------------------------------------
# TAB 1: OFFICIAL FORMAT PRACTICE TEST
# ------------------------------------------
with tab1:
    st.subheader(f"Official Format: 25 Questions | 40 Minutes | Attempt #{st.session_state.kb.get('total_attempts_counter', 0) + 1}")

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
                saved_ans = st.session_state.user_answers.get(q_num, None)
                default_idx = ["A", "B", "C", "D", "E"].index(saved_ans) if saved_ans in ["A", "B", "C", "D", "E"] else None

                choice = st.radio(
                    f"Select answer for Problem {q_num}:",
                    options=["A", "B", "C", "D", "E"],
                    format_func=lambda x: options_map[x],
                    index=default_idx,
                    key=f"q_radio_{q_num}",
                    label_visibility="collapsed",
                )
                if choice:
                    st.session_state.user_answers[q_num] = choice
            else:
                user_ans = st.session_state.user_answers.get(q_num, "Unanswered")
                correct_ans = str(q["correct_answer"]).strip().upper()
                if len(correct_ans) > 1 and correct_ans[0] in ["A", "B", "C", "D", "E"]:
                    correct_ans = correct_ans[0]

                user_full_text = options_map.get(user_ans, "Unanswered")
                correct_full_text = options_map.get(correct_ans, f"({correct_ans})")

                is_correct = False
                if user_ans == correct_ans:
                    is_correct = True
                else:
                    user_val = str(options_map.get(user_ans, "")).strip()
                    correct_val = str(options_map.get(correct_ans, "")).strip()
                    if user_val and user_val == correct_val:
                        is_correct = True

                if is_correct:
                    st.success(f"✅ **Your Answer:** {user_full_text} — Correct!")
                elif user_ans == "Unanswered":
                    st.warning(f"⚠️ **Your Answer:** Unanswered | **Correct Answer:** {correct_full_text}")
                else:
                    st.error(f"❌ **Your Answer:** {user_full_text} | **Correct Answer:** {correct_full_text}")

                with st.expander("View Step-by-Step Solution"):
                    st.markdown(q["step_by_step_solution"])

            st.divider()

        if not st.session_state.exam_submitted:
            if st.button("✅ Submit Exam", type="primary"):
                submit_exam()
                st.rerun()
        else:
            latest_attempt = st.session_state.kb["attempt_history"][-1]
            st.success(
                f"### Final Score: {latest_attempt['score']} / {latest_attempt['total_questions']} (Attempt #{latest_attempt['attempt_number']}) — Synced to Cloud ✅"
            )

            export_payload = json.dumps(latest_attempt, indent=2)
            st.download_button(
                label="📥 Download Test Attempt Report (.json)",
                data=export_payload,
                file_name=f"amc8_attempt_{latest_attempt['attempt_number']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
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
        "The AI will extract all 25 questions, solve them rigorously, double-check answers, and store them in your Knowledge Base."
    )

    pdf_files = st.file_uploader(
        "Select one or more AMC 8 PDF files", type=["pdf"], accept_multiple_files=True
    )

    if pdf_files and st.button("🧠 Extract & Solve All 25 Questions into Knowledge Base", type="primary"):
        for pdf in pdf_files:
            with st.spinner(f"Reading and solving all 25 questions from {pdf.name}..."):
                solved_qs = extract_and_solve_pdf(pdf)
                st.session_state.kb["question_bank"].extend(solved_qs)
                st.success(f"Added {len(solved_qs)} solved questions from **{pdf.name}**!")
        
        # Save updated KB with new questions to DynamoDB
        current_user = st.session_state.get("current_user", "admin")
        user_record = load_user_data_from_dynamodb(current_user)
        curr_pass = user_record.get("password", "admin123") if user_record else "admin123"
        save_user_data_to_dynamodb(current_user, curr_pass, st.session_state.kb)
        push_kb_to_github(json.dumps(st.session_state.kb, indent=2))

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

    attempts = st.session_state.kb.get("attempt_history", [])
    if not attempts:
        st.info("Complete at least one practice test to unlock analytics and AI coaching.")
    else:
        all_results = [r for att in attempts for r in att["results"]]
        df = pd.DataFrame(all_results)

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
