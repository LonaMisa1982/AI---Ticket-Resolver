"""
streamlit_app.py — Streamlit Frontend for ResolveAI Agent

Features:
  • Sidebar task selector pulled from ResolveAIEnv.TASKS
  • Live workspace dashboard: thought / action / reward per step
  • Custom ticket KB query → Historical Hints injected into agent context
  • Real-time log streaming via subprocess + queue
  • Soft ivory/canary minimal aesthetic
  • ✅ HITL: Approval gate — displays git diff, Approve / Reject buttons
  • ✅ KB Capture: After human approval, triggers knowledge capture via env
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ResolveAI — AI-Powered Code Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inline CSS — ResolveAI dark SaaS aesthetic
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@300;400;600&display=swap');

/* ---- Root ---- */
html, body, [data-testid="stApp"] {
    background: #0F1117;
    color: #E2E8F0;
    font-family: 'Inter', sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #161B27 !important;
    border-right: 1px solid #2D3748 !important;
}
[data-testid="stSidebar"] * { color: #CBD5E0 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F7FAFC !important; }

/* ---- Scrollbar ---- */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #1A202C; border-radius: 3px; }
::-webkit-scrollbar-thumb { background: #4A5568; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #718096; }

/* ---- Headings ---- */
h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    color: #F7FAFC !important;
    letter-spacing: -1px;
    font-size: 2.4rem !important;
    line-height: 1.1 !important;
}
h2 {
    font-family: 'Inter', sans-serif;
    font-weight: 700;
    color: #E2E8F0 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.4px;
}
h3 {
    font-family: 'Inter', sans-serif;
    font-weight: 600;
    color: #CBD5E0 !important;
    font-size: 1.2rem !important;
}

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: #1A202C;
    border: 1px solid #2D3748;
    border-radius: 14px;
    padding: 20px 24px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}
[data-testid="stMetricValue"] {
    color: #F7FAFC !important;
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    font-size: 2rem !important;
}
[data-testid="stMetricLabel"] {
    color: #718096 !important;
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 600;
}
[data-testid="stMetricDelta"] { font-size: 0.8rem !important; }

/* ---- Step card ---- */
.step-card {
    background: #1A202C;
    border: 1px solid #2D3748;
    border-left: 3px solid #4A5568;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    transition: box-shadow 0.2s, border-left-color 0.2s;
}
.step-card:hover { box-shadow: 0 4px 16px rgba(0,0,0,0.35); }
.step-card.done  { border-left-color: #48BB78; }
.step-card.error { border-left-color: #FC8181; }
.step-card.hitl  { border-left-color: #F6E05E; }

.step-label {
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    color: #718096;
    margin-bottom: 4px;
    font-family: 'Inter', sans-serif;
    font-weight: 600;
}
.step-value { color: #E2E8F0; }
.thought-text { color: #A0AEC0; font-style: italic; font-family: 'Inter', sans-serif; }

.action-badge {
    display: inline-block;
    background: #2D3748;
    color: #E2E8F0;
    border-radius: 6px;
    padding: 3px 12px;
    font-size: 0.72rem;
    font-family: 'JetBrains Mono', monospace;
    border: 1px solid #4A5568;
}
.action-badge.submit { background: #1C3A2A; color: #68D391; border-color: #2F6B48; }
.action-badge.error  { background: #3A1C1C; color: #FC8181; border-color: #7B3333; }
.action-badge.hitl   { background: #3A3500; color: #F6E05E; border-color: #7B6E00; }

.reward-good { color: #68D391; font-weight: 700; }
.reward-mid  { color: #F6E05E; font-weight: 700; }
.reward-bad  { color: #FC8181; font-weight: 700; }

/* ---- HITL review panel ---- */
.hitl-panel {
    background: #1E2A1A;
    border: 1.5px solid #4A6741;
    border-radius: 14px;
    padding: 22px 26px;
    margin: 16px 0;
    font-family: 'Inter', sans-serif;
}
.hitl-title {
    color: #9AE6B4;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.diff-view {
    background: #111827;
    border: 1px solid #2D3748;
    border-radius: 10px;
    padding: 16px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.76rem;
    line-height: 1.9;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre;
    color: #CBD5E0;
}
.diff-add  { color: #68D391; background: #1A2E1E; display: block; }
.diff-rem  { color: #FC8181; background: #2E1A1A; display: block; }
.diff-meta { color: #718096; display: block; }

/* ---- Hint card ---- */
.hint-card {
    background: #1A202C;
    border: 1px solid #2D3748;
    border-left: 3px solid #ECC94B;
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* ---- Log terminal ---- */
.log-terminal {
    background: #0D1117;
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 16px 18px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    line-height: 1.9;
    max-height: 280px;
    overflow-y: auto;
    color: #68D391;
    white-space: pre-wrap;
    word-break: break-all;
}

/* ---- Ticket box ---- */
.ticket-box {
    background: #1A202C;
    border: 1px solid #2D3748;
    border-radius: 12px;
    padding: 16px 18px;
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    line-height: 1.7;
    color: #CBD5E0;
    margin-bottom: 14px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.2);
}

/* ---- Buttons ---- */
.stButton > button {
    background: linear-gradient(135deg, #667EEA 0%, #764BA2 100%) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    padding: 10px 24px !important;
    transition: opacity 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 14px rgba(102,126,234,0.35) !important;
    letter-spacing: 0.2px !important;
}
.stButton > button:hover {
    opacity: 0.88 !important;
    box-shadow: 0 6px 20px rgba(102,126,234,0.5) !important;
}
.stButton > button:disabled {
    background: #2D3748 !important;
    color: #718096 !important;
    box-shadow: none !important;
    opacity: 0.7 !important;
}

/* ---- Approve / Reject override ---- */
button[key="hitl_approve"] {
    background: linear-gradient(135deg, #38A169, #276749) !important;
    box-shadow: 0 4px 14px rgba(56,161,105,0.35) !important;
}
button[key="hitl_reject"] {
    background: linear-gradient(135deg, #E53E3E, #9B2C2C) !important;
    box-shadow: 0 4px 14px rgba(229,62,62,0.35) !important;
}

/* ---- Select / text input / toggle — fix dropdown visibility ---- */
[data-baseweb="select"] {
    background: #1A202C !important;
    border-color: #4A5568 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}
[data-baseweb="select"] * {
    color: #E2E8F0 !important;
    background-color: #1A202C !important;
}
[data-baseweb="select"] [role="option"] {
    background-color: #2D3748 !important;
    color: #E2E8F0 !important;
}
[data-baseweb="select"] [role="option"]:hover,
[data-baseweb="select"] [aria-selected="true"] {
    background-color: #4A5568 !important;
}
[data-baseweb="menu"] {
    background: #2D3748 !important;
    border: 1px solid #4A5568 !important;
    border-radius: 10px !important;
}
[data-baseweb="menu"] li {
    color: #E2E8F0 !important;
    background: #2D3748 !important;
}
[data-baseweb="menu"] li:hover {
    background: #4A5568 !important;
}
[data-baseweb="input"] {
    background: #1A202C !important;
    border-color: #4A5568 !important;
    border-radius: 10px !important;
    color: #E2E8F0 !important;
}
[data-baseweb="input"] input {
    color: #E2E8F0 !important;
}
textarea {
    background: #1A202C !important;
    border-color: #4A5568 !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    color: #E2E8F0 !important;
}

/* Toggle accent */
[data-testid="stToggleSwitch"] [role="switch"][aria-checked="true"] {
    background-color: #667EEA !important;
}

/* ---- Progress bar ---- */
.stProgress > div > div { background: linear-gradient(90deg, #667EEA, #764BA2) !important; border-radius: 4px !important; }
.stProgress > div { background: #2D3748 !important; border-radius: 4px !important; }

/* ---- Divider ---- */
hr { border-color: #2D3748 !important; margin: 1.2rem 0 !important; }

/* ---- Section label / caption ---- */
.stCaption, small { color: #718096 !important; font-size: 0.76rem !important; }

/* ---- Radio ---- */
[data-testid="stRadio"] label { font-size: 0.84rem !important; color: #CBD5E0 !important; }
[data-testid="stRadio"] { background: transparent !important; }

/* ---- Expander ---- */
[data-testid="stExpander"] {
    border: 1px solid #2D3748 !important;
    border-radius: 12px !important;
    background: #1A202C !important;
}

/* ---- Section header pill ---- */
.section-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #2D3748;
    border: 1px solid #4A5568;
    border-radius: 20px;
    padding: 5px 16px;
    font-size: 0.76rem;
    font-weight: 700;
    color: #A0AEC0;
    margin-bottom: 14px;
    font-family: 'Inter', sans-serif;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* ---- Form labels ---- */
[data-testid="stSelectbox"] label,
[data-testid="stRadio"] > label,
[data-testid="stTextArea"] label,
[data-testid="stTextInput"] label {
    color: #A0AEC0 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
}

/* ---- Success / info / warning banners ---- */
[data-testid="stAlert"] {
    background: #1A202C !important;
    border-radius: 10px !important;
    border-color: #4A5568 !important;
    color: #E2E8F0 !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Token loading — secure, from file instead of UI input
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_hf_token() -> str:
    """Load HF token securely from ./token file."""
    token_file = Path("./token")
    if token_file.exists():
        try:
            token = token_file.read_text().strip()
            if token:
                return token
        except Exception as exc:
            print(f"[WARNING] Could not read token file: {exc}", flush=True)
    # Fallback to env var if token file doesn't exist
    fallback = os.getenv("HF_TOKEN", "")
    if not fallback:
        print("[WARNING] No HF_TOKEN found. Please create a ./token file or set HF_TOKEN env var.", flush=True)
    return fallback


# ---------------------------------------------------------------------------
# Lazy imports — kept here so Streamlit loads even if env isn't set up
# ---------------------------------------------------------------------------

@st.cache_resource
def _get_tasks_dict():
    try:
        from server.env import JiraToCodeEnv  # type: ignore
        return JiraToCodeEnv.TASKS
    except Exception:
        # Full fallback task list matching JiraToCodeEnv.TASKS
        return {
            "easy":     {"ticket": "TICKET-101: Fix the off-by-one bug in calculator.add() function. It should correctly sum two numbers."},
            "easy_2":   {"ticket": "TICKET-102: Fix the bug in string_utils.count_vowels(). It currently only counts lowercase vowels but should be case-insensitive."},
            "easy_3":   {"ticket": "TICKET-E3: The API endpoint crashes with a KeyError when a user payload doesn't contain an optional 'phone_number' field. Change dictionary indexing to .get() with a fallback."},
            "easy_4":   {"ticket": "TICKET-E4: Off-by-One Pagination. get_page_bounds(page, size) misses the 10th item on page 1. Fix the math index logic."},
            "easy_5":   {"ticket": "TICKET-E5: FastAPI Route Typo. Route signature is id instead of user_id. Fix the parameter mismatch."},
            "medium":   {"ticket": "TICKET-201: Implement format_user_data in formatter.py. It should format dictionary data to 'LAST_NAME, First_name (Age: X)'. Handle missing age by defaulting to 'Unknown'."},
            "medium_2": {"ticket": "TICKET-202: Implement validate_email() and validate_password() in validator.py."},
            "medium_3": {"ticket": "TICKET-M3: Missing Authentication Middleware. A sensitive endpoint (/api/billing) is exposed. Import @require_auth from auth.py and apply it to the route in routes.py."},
            "medium_4": {"ticket": "TICKET-M4: N+1 Database Problem. Rewrite the ORM query to use a JOIN (e.g., select_related)."},
            "medium_5": {"ticket": "TICKET-M5: Flawed Regex Validation. validate_email rejects emails with a plus sign. Update regex to allow user+test@gmail.com."},
            "medium_6": {"ticket": "TICKET-M6: Incomplete Error Handling. fetching currency rates crashes on timeout. Wrap in try/except and return a cached fallback value."},
            "medium_7": {"ticket": "TICKET-M7: Stale Cache Bug. update_user_profile updates DB but forgets to call redis.delete('user:id'). Invalidate the cache."},
            "medium_8": {"ticket": "TICKET-M8: Timezone Naive Conversion. Event scheduling function creates naive datetimes. Make them UTC aware."},
            "medium_9": {"ticket": "TICKET-M9: State Machine Loophole. Cart state machine allows CANCELLED to SHIPPED. Add transition guards."},
            "medium_10":{"ticket": "TICKET-M10: Config Merge Overwrite. YAML merge completely overwrites nested dictionaries. Fix recursion logic."},
            "hard":     {"ticket": "TICKET-301: Implement an LRUCache class in lru_cache.py with put() and get() methods. O(1) time complexity expected. Evict least recently used when capacity is reached."},
            "hard_2":   {"ticket": "TICKET-302: Implement a DirectedGraph class in graph.py with add_edge(), has_path() (BFS/DFS), and topological_sort() methods."},
            "hard_3":   {"ticket": "TICKET-H3: Circular Dependency Resolution. models.py, utils.py, config.py. Extract shared logic into base.py."},
            "hard_4":   {"ticket": "TICKET-H4: Race Condition in Thread Worker. Refactor the architecture to use queue.Queue."},
            "hard_5":   {"ticket": "TICKET-H5: OOM Generator Fix. Readlines causes crash on 5GB file. Rewrite to yield generators."},
            "hard_6":   {"ticket": "TICKET-H6: Implement Abstract Base Class. Implement StripeGateway matching PaymentGateway abstract class."},
            "hard_7":   {"ticket": "TICKET-H7: Deadlock in Asyncio. Route acquires threading.Lock but forgets to release on exception. Use async context managers."},
        }


@st.cache_resource
def _get_kb_collection():
    KB_DIR = "./corporate_memory"
    KB_COLLECTION = "jira_solutions"
    try:
        import chromadb  # type: ignore
        client = chromadb.PersistentClient(path=KB_DIR)
        return client.get_or_create_collection(KB_COLLECTION)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# KB query helper
# ---------------------------------------------------------------------------

def query_kb(ticket_text: str, n: int = 3):
    """Return list of hint dicts from ChromaDB."""
    collection = _get_kb_collection()
    if collection is None:
        return []
    try:
        count = collection.count()
        if count == 0:
            return []
        results = collection.query(query_texts=[ticket_text], n_results=min(n, count))
        hints = []
        for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
            hints.append(
                {
                    "ticket_id": meta.get("ticket_id", "?"),
                    "summary": meta.get("summary", ""),
                    "code": doc,
                    "task_level": meta.get("task_level", ""),
                }
            )
        return hints
    except Exception as exc:
        st.warning(f"KB query failed: {exc}")
        return []


def hints_to_prompt(hints) -> str:
    if not hints:
        return ""
    parts = ["## Historical Hints from KB\n"]
    for i, h in enumerate(hints, 1):
        parts.append(
            f"### Hint {i} — {h['ticket_id']} ({h['task_level']})\n"
            f"**Summary**: {h['summary']}\n"
            f"**Code**:\n```python\n{h['code']}\n```\n"
        )
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# HITL file-based helpers
# ---------------------------------------------------------------------------

HITL_REQUEST_FILE = "./hitl_request.json"
HITL_RESPONSE_FILE = "./hitl_response.json"


def _load_hitl_request() -> Optional[dict]:
    """Check if inference.py has written a pending review request."""
    if not Path(HITL_REQUEST_FILE).exists():
        return None
    try:
        with open(HITL_REQUEST_FILE) as f:
            return json.load(f)
    except Exception:
        return None


def _write_hitl_response(decision: str) -> None:
    """Write human decision ('approved' | 'rejected') so inference.py can continue."""
    with open(HITL_RESPONSE_FILE, "w") as f:
        json.dump({"decision": decision}, f)
    if Path(HITL_REQUEST_FILE).exists():
        os.remove(HITL_REQUEST_FILE)


def _render_diff_html(diff_text: str) -> str:
    """Colour-code a unified diff for HTML display."""
    lines = diff_text.splitlines()
    html_lines = []
    for line in lines:
        escaped = (
            line.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
        )
        if line.startswith("+") and not line.startswith("+++"):
            html_lines.append(f'<span class="diff-add">{escaped}</span>')
        elif line.startswith("-") and not line.startswith("---"):
            html_lines.append(f'<span class="diff-rem">{escaped}</span>')
        elif line.startswith("@@") or line.startswith("---") or line.startswith("+++"):
            html_lines.append(f'<span class="diff-meta">{escaped}</span>')
        else:
            html_lines.append(escaped)
    return "\n".join(html_lines)


# ---------------------------------------------------------------------------
# Log streaming helpers
# ---------------------------------------------------------------------------

_LOG_PATTERNS = {
    "start":        re.compile(r"\[START\]\s+task=(\S+)\s+env=(\S+)\s+model=(\S+)"),
    "step":         re.compile(
        r"\[STEP\]\s+step=(\d+)\s+action=(\S+)\s+reward=([\d.]+)\s+done=(\S+)\s+error=(\S+)"
    ),
    "end":          re.compile(
        r"\[END\]\s+success=(\S+)\s+steps=(\d+)\s+score=([\d.]+)\s+rewards=([\d.,]+)"
    ),
    "thought":      re.compile(r"\[THOUGHT\]\s+step=(\d+)\s+thought=(.+)"),
    "kb":           re.compile(r"\[KB\](.+)"),
    "mapper":       re.compile(r"\[MAPPER\](.+)"),
    "hitl":         re.compile(r"\[HITL\](.+)"),
    "fix_summary":  re.compile(r"\[FIX_SUMMARY\]\s+(.+)"),
    "hitl_pending": re.compile(r"\[HITL_PENDING\]\s+(.+)"),
}


def parse_log_line(line: str) -> Optional[dict]:
    for kind, pat in _LOG_PATTERNS.items():
        m = pat.search(line)
        if m:
            if kind == "start":
                return {"kind": "start", "task": m.group(1), "model": m.group(3)}
            elif kind == "step":
                return {
                    "kind": "step",
                    "step": int(m.group(1)),
                    "action": m.group(2),
                    "reward": float(m.group(3)),
                    "done": m.group(4) == "true",
                    "error": None if m.group(5) == "null" else m.group(5),
                    "thought": "",
                }
            elif kind == "end":
                return {
                    "kind": "end",
                    "success": m.group(1) == "true",
                    "steps": int(m.group(2)),
                    "score": float(m.group(3)),
                    "rewards": [float(r) for r in m.group(4).split(",") if r],
                }
            elif kind == "thought":
                return {"kind": "thought", "step": int(m.group(1)), "thought": m.group(2).strip("'")}
            elif kind == "kb":
                return {"kind": "kb", "msg": m.group(1).strip()}
            elif kind == "mapper":
                return {"kind": "mapper", "msg": m.group(1).strip()}
            elif kind == "hitl":
                return {"kind": "hitl", "msg": m.group(1).strip()}
            elif kind == "fix_summary":
                try:
                    payload = json.loads(m.group(1).strip())
                    return {"kind": "fix_summary", "payload": payload}
                except Exception:
                    return None
            elif kind == "hitl_pending":
                try:
                    payload = json.loads(m.group(1).strip())
                    return {"kind": "hitl_pending", "payload": payload}
                except Exception:
                    return None
    return None


def _stream_subprocess(cmd, log_queue: queue.Queue, env_vars: dict):
    """Run *cmd* in a subprocess and push every output line to *log_queue*."""
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env_vars,
        )
        for line in proc.stdout:
            log_queue.put(line.rstrip())
        proc.wait()
    except Exception as exc:
        log_queue.put(f"[ERROR] Subprocess failed: {exc}")
    finally:
        log_queue.put(None)  # sentinel


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _reward_class(r: float) -> str:
    if r >= 0.4:
        return "reward-good"
    if r >= 0.1:
        return "reward-mid"
    return "reward-bad"


def _action_badge(action: str) -> str:
    if "request_human_review" in action:
        cls = "hitl"
    elif action == "submit":
        cls = "submit"
    elif "ERROR" in action:
        cls = "error"
    else:
        cls = ""
    return f'<span class="action-badge {cls}">{action}</span>'


def render_step_card(step_data: dict):
    thought = step_data.get("thought", "")
    action = step_data.get("action", "")
    reward = step_data.get("reward", 0.0)
    error = step_data.get("error")
    done = step_data.get("done", False)

    is_hitl = "request_human_review" in action
    card_cls = "done" if done else ("hitl" if is_hitl else ("error" if error else ""))
    rc = _reward_class(reward)
    action_html = _action_badge(action[:40])

    thought_html = (
        f'<div class="step-label">Thought</div>'
        f'<div class="thought-text">{thought}</div><br/>'
        if thought
        else ""
    )
    error_html = (
        f'<div class="step-label" style="color:#C0554A">Error</div>'
        f'<div style="color:#C0554A; font-family:JetBrains Mono,monospace">{error}</div><br/>'
        if error
        else ""
    )

    st.markdown(
        f"""
<div class="step-card {card_cls}">
  <div class="step-label">Step {step_data.get('step', '?')}</div>
  {thought_html}
  <div class="step-label">Action</div>
  <div>{action_html}</div><br/>
  {error_html}
  <div class="step-label">Reward</div>
  <div class="{rc}">{reward:.3f}</div>
</div>""",
        unsafe_allow_html=True,
    )


def render_hint_cards(hints):
    for h in hints:
        st.markdown(
            f"""
<div class="hint-card">
  <div class="step-label">📁 {h['ticket_id']} &nbsp;·&nbsp; {h['task_level']}</div>
  <div style="color:#8A7830; margin-bottom:6px; font-family:Inter,sans-serif; font-size:0.82rem">{h['summary']}</div>
  <details>
    <summary style="cursor:pointer; color:#A3A3A3; font-size:0.72rem; font-family:Inter,sans-serif">Show code snippet</summary>
    <pre style="margin-top:8px; color:#5A7A5A; font-size:0.76rem; white-space:pre-wrap; background:#F5F6EC; padding:8px; border-radius:6px">{h['code'][:800]}{'…' if len(h['code']) > 800 else ''}</pre>
  </details>
</div>""",
            unsafe_allow_html=True,
        )


def _escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
    )


def render_fix_summary(summaries: list):
    """Render a before/after side-by-side panel for every file the agent changed."""
    if not summaries:
        return

    st.markdown("---")
    st.markdown(
        '<div class="section-pill">🔬 Bug Fix — Before &amp; After</div>',
        unsafe_allow_html=True,
    )

    for fix in summaries:
        file_path = fix.get("file_path", "unknown")
        original  = fix.get("original", "")
        fixed     = fix.get("fixed", "")
        passed    = fix.get("tests_passed", False)
        n_passed  = fix.get("passed", 0)
        n_total   = fix.get("total", 0)

        status_color = "#3D8B5E" if passed else "#B8960A"
        status_label = (
            f"✅ All {n_total} tests passed"
            if passed
            else f"⚠️ {n_passed}/{n_total} tests passed"
        )

        import difflib
        orig_lines  = original.splitlines()
        fixed_lines = fixed.splitlines()

        diff = list(difflib.unified_diff(orig_lines, fixed_lines, lineterm="", n=0))
        removed = set()
        added   = set()
        orig_line_no  = 0
        fixed_line_no = 0
        for dline in diff:
            if dline.startswith("@@"):
                import re as _re
                m = _re.search(r"-(\d+)(?:,\d+)? \+(\d+)(?:,\d+)?", dline)
                if m:
                    orig_line_no  = int(m.group(1)) - 1
                    fixed_line_no = int(m.group(2)) - 1
            elif dline.startswith("-"):
                removed.add(orig_line_no)
                orig_line_no += 1
            elif dline.startswith("+"):
                added.add(fixed_line_no)
                fixed_line_no += 1
            else:
                orig_line_no  += 1
                fixed_line_no += 1

        def _render_code_with_highlights(lines, highlight_lines, add_color, bg_color):
            html_parts = []
            for i, line in enumerate(lines):
                escaped = _escape_html(line)
                if i in highlight_lines:
                    html_parts.append(
                        f'<div style="background:{bg_color}; color:{add_color}; '
                        f'padding:0 4px; white-space:pre; font-family:JetBrains Mono,monospace; '
                        f'font-size:0.76rem; line-height:1.7;">{escaped}</div>'
                    )
                else:
                    html_parts.append(
                        f'<div style="white-space:pre; font-family:JetBrains Mono,monospace; '
                        f'font-size:0.76rem; line-height:1.7; padding:0 4px; color:#424242;">{escaped}</div>'
                    )
            return "".join(html_parts)

        orig_html  = _render_code_with_highlights(orig_lines,  removed, "#B84A40", "#FDF5F4")
        fixed_html = _render_code_with_highlights(fixed_lines, added,   "#3D8B5E", "#F0FBF4")

        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown(
                f'<div style="font-family:Inter,sans-serif; font-size:0.72rem; '
                f'text-transform:uppercase; letter-spacing:1px; color:#C0554A; '
                f'margin-bottom:6px; font-weight:600;">🐛 Buggy — {_escape_html(file_path)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#FFFFFF; border:1px solid #F0CDCA; '
                f'border-left:3px solid #E8887A; border-radius:8px; padding:12px; '
                f'max-height:480px; overflow-y:auto;">'
                f'{orig_html}</div>',
                unsafe_allow_html=True,
            )

        with col_after:
            st.markdown(
                f'<div style="font-family:Inter,sans-serif; font-size:0.72rem; '
                f'text-transform:uppercase; letter-spacing:1px; color:#3D8B5E; '
                f'margin-bottom:6px; font-weight:600;">✅ Fixed — {_escape_html(file_path)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#FFFFFF; border:1px solid #C8E8D4; '
                f'border-left:3px solid #7EC8A4; border-radius:8px; padding:12px; '
                f'max-height:480px; overflow-y:auto;">'
                f'{fixed_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="text-align:center; margin:8px 0 16px; font-family:Inter,sans-serif; '
            f'font-size:0.85rem; color:{status_color}; font-weight:500;">{status_label}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# HITL Review Panel
# ---------------------------------------------------------------------------

def render_hitl_panel(review: dict, hitl_placeholder):
    """Render the final approval gate for the completed solution."""
    diff_html = _render_diff_html(review.get("diff", "(no diff available)"))
    test_results = review.get("test_results", {})
    passed = test_results.get("passed", 0)
    total = test_results.get("total", 1)
    fixed_files_list = list(review.get("fixed_files", {}).keys())

    with hitl_placeholder.container():
        st.markdown(
            f"""
<div class="hitl-panel">
  <div class="hitl-title">🔍 Final Approval Required</div>
  <p style="color:#6A6A50; font-size:0.88rem; margin:0 0 6px;">
    All tests passed ({passed}/{total}). The agent has completed the task successfully.
    Please review the changes and approve or reject before they are persisted.
  </p>
  <p style="color:#A3A3A3; font-size:0.8rem; margin-top:6px;">
    <strong style="color:#6A6A50">Files changed:</strong> {', '.join(fixed_files_list) if fixed_files_list else 'None'}
  </p>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="step-label" style="margin-bottom:6px;">Code Changes</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="diff-view">{diff_html}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("<br/>", unsafe_allow_html=True)
        col_approve, col_reject, _ = st.columns([1, 1, 4])
        with col_approve:
            if st.button("✅ Approve & Persist", key="hitl_approve", use_container_width=True):
                _write_hitl_response("approved")
                st.session_state.pending_hitl = None
                st.rerun()
        with col_reject:
            if st.button("❌ Reject & Discard", key="hitl_reject", use_container_width=True):
                _write_hitl_response("rejected")
                st.session_state.pending_hitl = None
                st.rerun()


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

def _init_state():
    defaults = {
        "running": False,
        "steps": [],
        "raw_logs": [],
        "end_info": None,
        "start_info": None,
        "log_queue": None,
        "thread": None,
        "thoughts": {},
        "kb_hints": [],
        "custom_ticket": "",
        "pending_hitl": None,
        "fix_summaries": [],
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

TASKS = _get_tasks_dict()
task_keys = list(TASKS.keys())
level_groups = {"easy": [], "medium": [], "hard": []}
for k in task_keys:
    g = "hard" if "hard" in k else ("medium" if "medium" in k else "easy")
    level_groups[g].append(k)

# Load token + config from env (not exposed in UI)
hf_token = _load_hf_token()
api_base   = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")

with st.sidebar:
    st.markdown(
        """
<div style="padding: 10px 0 18px;">
  <div style="display:flex; align-items:center; gap:10px; margin-bottom:4px;">
    <div style="width:32px; height:32px; background:linear-gradient(135deg,#667EEA,#764BA2); border-radius:8px; display:flex; align-items:center; justify-content:center; font-size:16px;">🤖</div>
    <div>
      <div style="font-family:Inter,sans-serif; font-weight:800; font-size:1.15rem; color:#F7FAFC; letter-spacing:-0.4px;">ResolveAI</div>
      <div style="font-size:0.68rem; color:#718096; font-family:Inter,sans-serif; letter-spacing:1px; text-transform:uppercase; font-weight:600;">Agent Dashboard</div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        '<div class="section-pill">🎯 Task Selection</div>',
        unsafe_allow_html=True,
    )
    level_filter = st.radio("Difficulty", ["All", "Easy", "Medium", "Hard"], horizontal=True)
    if level_filter == "All":
        filtered_keys = task_keys
    else:
        filtered_keys = level_groups[level_filter.lower()]

    selected_task = st.selectbox("Select Task", filtered_keys, key="task_select")

    ticket_text = TASKS.get(selected_task, {}).get("ticket", "")
    st.markdown(
        f'<div class="ticket-box"><div class="step-label" style="margin-bottom:6px;">📋 Ticket</div>{ticket_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    # ✅ HITL toggle — only control exposed
    st.markdown(
        '<div class="section-pill">🤝 Human-in-the-Loop</div>',
        unsafe_allow_html=True,
    )
    hitl_enabled = st.toggle("Enable HITL Review", value=True)
    if hitl_enabled:
        st.caption("Agent will pause for your approval before persisting fixes.")
    else:
        st.caption("Fully automated — all proposed changes are auto-approved.")

    st.markdown("---")

    st.markdown(
        '<div class="section-pill">🧠 KB Hints</div>',
        unsafe_allow_html=True,
    )
    custom_ticket = st.text_area(
        "Search knowledge base",
        placeholder="Paste any ticket text to find similar past solutions…",
        key="custom_ticket_input",
        height=90,
    )
    if st.button("🔍 Search KB"):
        query_text = custom_ticket.strip() or ticket_text
        hints = query_kb(query_text)
        st.session_state.kb_hints = hints
        if hints:
            st.success(f"Found {len(hints)} hint(s)")
        else:
            st.info("No matching solutions found in KB yet.")

    kb_collection = _get_kb_collection()
    kb_count = kb_collection.count() if kb_collection else 0
    st.caption(f"KB size: {kb_count} solution(s)  ·  📂 ./corporate_memory")
    st.caption("🔒 API credentials loaded from environment")

    st.markdown("---")
    run_btn = st.button("▶  Run Agent", use_container_width=True, disabled=st.session_state.running)
    if st.session_state.running:
        st.caption("⏳ Agent is running…")

# ---------------------------------------------------------------------------
# Main area layout
# ---------------------------------------------------------------------------

st.markdown(
    """
<div style="padding: 8px 0 20px;">
  <div style="display:flex; align-items:center; gap:14px; margin-bottom:6px;">
    <div style="width:44px; height:44px; background:linear-gradient(135deg,#667EEA,#764BA2); border-radius:12px; display:flex; align-items:center; justify-content:center; font-size:22px; box-shadow:0 4px 16px rgba(102,126,234,0.4);">🤖</div>
    <div>
      <div style="font-family:Inter,sans-serif; font-weight:800; font-size:2.2rem; color:#F7FAFC; letter-spacing:-1px; line-height:1.1;">ResolveAI</div>
      <div style="font-size:0.82rem; color:#718096; font-family:Inter,sans-serif; font-weight:500; margin-top:2px; letter-spacing:0.2px;">AI-Powered Code Agent · Live Dashboard</div>
    </div>
  </div>
  <div style="display:flex; gap:8px; margin-top:10px;">
    <span style="background:#1A2E4A; color:#63B3ED; border:1px solid #2B4A6A; border-radius:6px; padding:3px 10px; font-size:0.72rem; font-family:Inter,sans-serif; font-weight:600;">v1.0</span>
    <span style="background:#1E2A1A; color:#68D391; border:1px solid #2F6B48; border-radius:6px; padding:3px 10px; font-size:0.72rem; font-family:Inter,sans-serif; font-weight:600;">● Live</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("---")

top_col1, top_col2, top_col3, top_col4 = st.columns(4)
metric_step    = top_col1.empty()
metric_reward  = top_col2.empty()
metric_score   = top_col3.empty()
metric_status  = top_col4.empty()


def _refresh_metrics():
    steps = st.session_state.steps
    end = st.session_state.end_info
    total_steps = len(steps)
    last_reward = steps[-1]["reward"] if steps else 0.0
    score = end["score"] if end else (sum(s["reward"] for s in steps))
    if st.session_state.pending_hitl:
        status = "🔍 Awaiting Review"
    elif end and end["success"]:
        status = "✅ Done"
    elif end and not end["success"]:
        status = "❌ Failed"
    else:
        status = "⏳ Running"

    metric_step.metric("Steps", total_steps)
    metric_reward.metric("Last Reward", f"{last_reward:.3f}")
    metric_score.metric("Cumulative Score", f"{score:.3f}")
    metric_status.metric("Status", status)


_refresh_metrics()

# HITL review panel (full-width, shown above step feed when active)
hitl_placeholder = st.empty()

main_col, side_col = st.columns([3, 2])

with main_col:
    st.markdown(
        '<div class="section-pill">📋 Step Feed</div>',
        unsafe_allow_html=True,
    )
    steps_container = st.container()

with side_col:
    st.markdown(
        '<div class="section-pill">📜 Raw Logs</div>',
        unsafe_allow_html=True,
    )
    log_box = st.empty()
    st.markdown(
        '<div class="section-pill" style="margin-top:16px;">💡 KB Hints</div>',
        unsafe_allow_html=True,
    )
    hints_area = st.container()

with hints_area:
    if st.session_state.kb_hints:
        render_hint_cards(st.session_state.kb_hints)
    else:
        st.caption("Search KB in the sidebar to see historical hints.")

progress_bar = st.empty()
end_summary_area = st.empty()

# ---------------------------------------------------------------------------
# Run agent — launch subprocess
# ---------------------------------------------------------------------------

if run_btn and not st.session_state.running:
    st.session_state.steps = []
    st.session_state.raw_logs = []
    st.session_state.end_info = None
    st.session_state.start_info = None
    st.session_state.thoughts = {}
    st.session_state.pending_hitl = None
    st.session_state.fix_summaries = []
    st.session_state.running = True

    env_vars = os.environ.copy()
    env_vars["JIRA_TASK_LEVEL"] = selected_task
    if api_base:
        env_vars["API_BASE_URL"] = api_base
    if model_name:
        env_vars["MODEL_NAME"] = model_name
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    hints_str = hints_to_prompt(st.session_state.kb_hints)
    if hints_str:
        env_vars["KB_HINTS_INJECT"] = hints_str

    lq = queue.Queue()
    st.session_state.log_queue = lq

    cmd = [sys.executable, "inference.py", "--tasks", selected_task]
    if not hitl_enabled:
        cmd.append("--no-hitl")

    t = threading.Thread(
        target=_stream_subprocess, args=(cmd, lq, env_vars), daemon=True
    )
    t.start()
    st.session_state.thread = t

# ---------------------------------------------------------------------------
# HITL approval gate
# ---------------------------------------------------------------------------

if st.session_state.pending_hitl is None and not st.session_state.running:
    review = _load_hitl_request()
    if review:
        st.session_state.pending_hitl = review

if st.session_state.pending_hitl:
    render_hitl_panel(st.session_state.pending_hitl, hitl_placeholder)
else:
    hitl_placeholder.empty()

# ---------------------------------------------------------------------------
# Poll log queue and update UI
# ---------------------------------------------------------------------------

if st.session_state.running:
    lq = st.session_state.log_queue
    if lq is not None:
        changed = False
        max_drain = 60

        for _ in range(max_drain):
            try:
                line = lq.get_nowait()
            except queue.Empty:
                break

            if line is None:
                st.session_state.running = False
                break

            st.session_state.raw_logs.append(line)
            parsed = parse_log_line(line)

            if parsed:
                if parsed["kind"] == "start":
                    st.session_state.start_info = parsed
                elif parsed["kind"] == "thought":
                    st.session_state.thoughts[parsed["step"]] = parsed["thought"]
                    for s in st.session_state.steps:
                        if s["step"] == parsed["step"] and not s.get("thought"):
                            s["thought"] = parsed["thought"]
                elif parsed["kind"] == "step":
                    parsed["thought"] = st.session_state.thoughts.get(parsed["step"], "")
                    st.session_state.steps.append(parsed)
                elif parsed["kind"] == "end":
                    st.session_state.end_info = parsed
                    st.session_state.running = False
                elif parsed["kind"] == "fix_summary":
                    st.session_state.fix_summaries.append(parsed["payload"])
                elif parsed["kind"] == "hitl_pending":
                    if st.session_state.pending_hitl is None:
                        st.session_state.pending_hitl = parsed["payload"]
                elif parsed["kind"] == "hitl":
                    pass

            changed = True

        if changed:
            with steps_container:
                for s in st.session_state.steps:
                    render_step_card(s)

            log_text = "\n".join(st.session_state.raw_logs[-80:])
            log_box.markdown(
                f'<div class="log-terminal">{log_text}</div>',
                unsafe_allow_html=True,
            )

            _refresh_metrics()

            task_max = 15 if "easy" in selected_task else 25
            progress = min(len(st.session_state.steps) / task_max, 1.0)
            progress_bar.progress(progress)

        if st.session_state.running:
            time.sleep(0.5)
            st.rerun()
        elif st.session_state.pending_hitl:
            time.sleep(0.5)
            st.rerun()

# ---------------------------------------------------------------------------
# End summary banner
# ---------------------------------------------------------------------------
if st.session_state.end_info:
    end = st.session_state.end_info
    if end["success"]:
        border_color = "#90D4AA"
        bg_color = "#F4FBF7"
        text_color = "#2A6E4A"
        icon = "✅"
    else:
        border_color = "#EAA09A"
        bg_color = "#FDF5F4"
        text_color = "#8A3530"
        icon = "❌"

    end_summary_area.markdown(
        f"""
<div style="background:{bg_color}; border:1px solid {border_color}; border-radius:12px;
            padding:20px 24px; font-family:Inter,sans-serif; margin-top:16px;
            box-shadow:0 1px 4px rgba(0,0,0,0.04);">
  <div style="font-weight:700; font-size:1rem; color:{text_color}; margin-bottom:8px;">{icon} Episode Complete</div>
  <div style="color:#6A6A50; font-size:0.88rem;">
    <strong>Task:</strong> {selected_task} &nbsp;·&nbsp;
    <strong>Steps:</strong> {end['steps']} &nbsp;·&nbsp;
    <strong>Final Score:</strong> {end['score']:.3f} &nbsp;·&nbsp;
    <strong>Success:</strong> {'Yes' if end['success'] else 'No'}
  </div>
</div>""",
        unsafe_allow_html=True,
    )

# Render before/after for every file the agent changed
if st.session_state.end_info and st.session_state.fix_summaries:
    render_fix_summary(st.session_state.fix_summaries)

# ---------------------------------------------------------------------------
# Render step cards if not currently running (persisted state)
# ---------------------------------------------------------------------------
if not st.session_state.running and st.session_state.steps:
    with steps_container:
        for s in st.session_state.steps:
            render_step_card(s)
    log_text = "\n".join(st.session_state.raw_logs[-80:])
    log_box.markdown(
        f'<div class="log-terminal">{log_text}</div>',
        unsafe_allow_html=True,
    )
    _refresh_metrics()