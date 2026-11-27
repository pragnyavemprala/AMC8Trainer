# ==========================================
# 2. SESSION STATE & GITHUB AUTO-SYNC HELPERS
# ==========================================
def fetch_kb_from_github() -> Optional[dict]:
    """Automatically loads knowledge_base.json from GitHub on startup."""
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


def push_kb_to_github(json_content_str: str) -> bool:
    """Commits and pushes the updated knowledge base JSON directly to GitHub."""
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

    github_kb = fetch_kb_from_github()
    
    default_kb = github_kb if github_kb else {
        "users": {"admin": st.secrets.get("APP_PASSWORD", "admin123")}, # Default fallback account
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

    st.session_state.kb.setdefault("users", {"admin": st.secrets.get("APP_PASSWORD", "admin123")})
    st.session_state.kb.setdefault("question_bank", [])
    st.session_state.kb.setdefault("attempt_history", [])
    st.session_state.kb.setdefault("self_confidence", {t: 50 for t in ALL_TOPICS})
    st.session_state.kb.setdefault("total_attempts_counter", len(st.session_state.kb["attempt_history"]))


init_state()

# ==========================================
# 3. LOGIN & REGISTRATION GATE SCREEN
# ==========================================
if not st.session_state["logged_in"]:
    st.title("🔒 AMC 8 Trainer Portal")
    
    auth_mode = st.radio("Choose action:", ["Login to existing account", "Create your account"], horizontal=True)
    st.divider()

    if auth_mode == "Login to existing account":
        st.subheader("Login")
        with st.form("login_form"):
            username = st.text_input("Username").strip()
            password = st.text_input("Password", type="password")
            submit_login = st.form_submit_button("Login", type="primary")
            
            if submit_login:
                users_db = st.session_state.kb.get("users", {})
                if username in users_db and users_db[username] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["current_user"] = username
                    st.success(f"Welcome back, {username}!")
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
    else:
        st.subheader("Create a New Account")
        with st.form("signup_form"):
            new_username = st.text_input("Choose Username").strip()
            new_password = st.text_input("Choose Password", type="password")
            confirm_password = st.text_input("Confirm Password", type="password")
            submit_signup = st.form_submit_button("Register Account", type="primary")
            
            if submit_signup:
                if not new_username or not new_password:
                    st.error("Username and password cannot be empty.")
                elif new_password != confirm_password:
                    st.error("Passwords do not match.")
                else:
                    users_db = st.session_state.kb.setdefault("users", {})
                    if new_username in users_db:
                        st.error("Username already exists. Please choose another or log in.")
                    else:
                        users_db[new_username] = new_password
                        st.session_state["logged_in"] = True
                        st.session_state["current_user"] = new_username
                        
                        # Sync new account creation to GitHub
                        push_kb_to_github(json.dumps(st.session_state.kb, indent=2))
                        
                        st.success(f"Account created successfully! Welcome, {new_username}!")
                        st.rerun()

    st.stop()  # Halt execution of the rest of the app until authenticated
