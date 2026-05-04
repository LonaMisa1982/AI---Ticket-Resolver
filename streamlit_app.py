"""
streamlit_app.py — Streamlit Frontend for Jira-to-Code Agent

Features:
  • Sidebar task selector pulled from JiraToCodeEnv.TASKS
  • Live workspace dashboard: thought / action / reward per step
  • Custom ticket KB query → Historical Hints injected into agent context
  • Real-time log streaming via subprocess + queue
  • Dark industrial-terminal aesthetic
  • ✅ CHANGE 4 (HITL): Approval gate — displays git diff, Approve / Reject buttons
  • ✅ CHANGE 5 (KB Capture): After human approval, triggers knowledge capture via env
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
    page_title="Jira-to-Code Agent",
    page_icon="⚙",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Inline CSS — industrial terminal aesthetic
# ---------------------------------------------------------------------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;600;700&family=Space+Grotesk:wght@300;400;600;700&display=swap');

/* ---- Root ---- */
html, body, [data-testid="stApp"] {
    background: #0a0c0f;
    color: #c9d1d9;
    font-family: 'Space Grotesk', sans-serif;
}

/* ---- Sidebar ---- */
[data-testid="stSidebar"] {
    background: #0d1117 !important;
    border-right: 1px solid #21262d;
}
[data-testid="stSidebar"] * { color: #c9d1d9 !important; }

/* ---- Headings ---- */
h1 { font-family: 'JetBrains Mono', monospace; color: #58a6ff !important; letter-spacing: -1px; }
h2, h3 { font-family: 'Space Grotesk', sans-serif; color: #79c0ff !important; }

/* ---- Metric cards ---- */
[data-testid="stMetric"] {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #58a6ff !important; font-family: 'JetBrains Mono', monospace; font-size: 1.6rem !important; }
[data-testid="stMetricLabel"] { color: #8b949e !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 1px; }

/* ---- Step card ---- */
.step-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
}
.step-card.done { border-left-color: #3fb950; }
.step-card.error { border-left-color: #f85149; }
.step-card.hitl { border-left-color: #d29922; }

.step-label {
    font-size: 0.68rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    color: #8b949e;
    margin-bottom: 3px;
}
.step-value { color: #e6edf3; }
.thought-text { color: #a5d6ff; font-style: italic; }
.action-badge {
    display: inline-block;
    background: #1f6feb;
    color: #e6edf3;
    border-radius: 4px;
    padding: 1px 8px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
}
.action-badge.submit { background: #1a7f37; }
.action-badge.error  { background: #b91c1c; }
.action-badge.hitl   { background: #9e6a03; }

.reward-good { color: #3fb950; font-weight: 600; }
.reward-mid  { color: #d29922; font-weight: 600; }
.reward-bad  { color: #f85149; font-weight: 600; }

/* ---- HITL review panel ---- */
.hitl-panel {
    background: #1c1a10;
    border: 2px solid #d29922;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 16px 0;
    font-family: 'JetBrains Mono', monospace;
}
.hitl-title {
    color: #d29922;
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.diff-view {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 4px;
    padding: 12px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.78rem;
    line-height: 1.7;
    max-height: 400px;
    overflow-y: auto;
    white-space: pre;
    color: #c9d1d9;
}
.diff-add { color: #3fb950; }
.diff-rem { color: #f85149; }
.diff-meta { color: #8b949e; }

/* ---- Hint card ---- */
.hint-card {
    background: #1c2128;
    border: 1px solid #30363d;
    border-left: 3px solid #d29922;
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    line-height: 1.6;
}

/* ---- Log terminal ---- */
.log-terminal {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 6px;
    padding: 14px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    line-height: 1.8;
    max-height: 300px;
    overflow-y: auto;
    color: #7ee787;
    white-space: pre-wrap;
    word-break: break-all;
}

/* ---- Ticket box ---- */
.ticket-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 16px 20px;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #c9d1d9;
    margin-bottom: 16px;
}

/* ---- Buttons ---- */
.stButton > button {
    background: #1f6feb !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-weight: 600 !important;
    padding: 8px 20px !important;
    transition: background 0.2s;
}
.stButton > button:hover { background: #388bfd !important; }

/* ---- Approve / Reject override ---- */
button[data-testid="approve-btn"] { background: #1a7f37 !important; }
button[data-testid="reject-btn"]  { background: #b91c1c !important; }

/* ---- Selectbox / text input ---- */
[data-baseweb="select"] { background: #161b22 !important; border-color: #30363d !important; }
[data-baseweb="input"] { background: #161b22 !important; border-color: #30363d !important; }

/* ---- Progress bar ---- */
.stProgress > div > div { background: #58a6ff !important; }

/* ---- divider ---- */
hr { border-color: #21262d !important; }
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
        return {
            "easy": {"ticket": "TICKET-101: Fix off-by-one bug in calculator.add()"},
            "medium": {"ticket": "TICKET-201: Implement format_user_data in formatter.py"},
            "hard": {"ticket": "TICKET-301: Implement LRUCache class in lru_cache.py"},
        }


@st.cache_resource
def _get_kb_collection():
    # ✅ CHANGE 2: points to ./corporate_memory
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
# ✅ CHANGE 4: HITL file-based helpers
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
    # Remove the request file so _load_hitl_request returns None next poll
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
    "fix_summary":  re.compile(r"\[FIX_SUMMARY\]\s+(.+)"),   # JSON payload
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
        f'<div class="step-label" style="color:#f85149">Error</div>'
        f'<div style="color:#f85149">{error}</div><br/>'
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
  <div style="color:#ffa657; margin-bottom:6px">{h['summary']}</div>
  <details>
    <summary style="cursor:pointer; color:#8b949e; font-size:0.75rem">Show code snippet</summary>
    <pre style="margin-top:8px; color:#7ee787; font-size:0.78rem; white-space:pre-wrap">{h['code'][:800]}{'…' if len(h['code']) > 800 else ''}</pre>
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
    st.markdown("### 🔬 Bug Fix — Before & After")

    for fix in summaries:
        file_path = fix.get("file_path", "unknown")
        original  = fix.get("original", "")
        fixed     = fix.get("fixed", "")
        passed    = fix.get("tests_passed", False)
        n_passed  = fix.get("passed", 0)
        n_total   = fix.get("total", 0)

        status_color = "#3fb950" if passed else "#d29922"
        status_label = (
            f"✅ All {n_total} tests passed"
            if passed
            else f"⚠️ {n_passed}/{n_total} tests passed"
        )

        # Compute a line-level diff to highlight changed lines
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
                # Parse hunk header e.g. @@ -3,4 +3,5 @@
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
                        f'font-size:0.78rem; line-height:1.7;">{escaped}</div>'
                    )
                else:
                    html_parts.append(
                        f'<div style="white-space:pre; font-family:JetBrains Mono,monospace; '
                        f'font-size:0.78rem; line-height:1.7; padding:0 4px;">{escaped}</div>'
                    )
            return "".join(html_parts)

        orig_html  = _render_code_with_highlights(orig_lines,  removed, "#f85149", "#2d1515")
        fixed_html = _render_code_with_highlights(fixed_lines, added,   "#3fb950", "#122612")

        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown(
                f'<div style="font-family:Space Grotesk,sans-serif; font-size:0.75rem; '
                f'text-transform:uppercase; letter-spacing:1px; color:#f85149; '
                f'margin-bottom:6px;">🐛 Buggy — {_escape_html(file_path)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#0d1117; border:1px solid #30363d; '
                f'border-left:3px solid #f85149; border-radius:6px; padding:12px; '
                f'max-height:480px; overflow-y:auto; color:#c9d1d9;">'
                f'{orig_html}</div>',
                unsafe_allow_html=True,
            )

        with col_after:
            st.markdown(
                f'<div style="font-family:Space Grotesk,sans-serif; font-size:0.75rem; '
                f'text-transform:uppercase; letter-spacing:1px; color:#3fb950; '
                f'margin-bottom:6px;">✅ Fixed — {_escape_html(file_path)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div style="background:#0d1117; border:1px solid #30363d; '
                f'border-left:3px solid #3fb950; border-radius:6px; padding:12px; '
                f'max-height:480px; overflow-y:auto; color:#c9d1d9;">'
                f'{fixed_html}</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            f'<div style="text-align:center; margin:8px 0 16px; font-family:Space Grotesk,sans-serif; '
            f'font-size:0.88rem; color:{status_color};">{status_label}</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# ✅ CHANGE 4: HITL Review Panel
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
  <div class="hitl-title">✅ Final Approval Required</div>
  <p style="color:#c9d1d9; font-size:0.9rem">
    All tests passed ({passed}/{total}). The agent has completed the task successfully.
    Please review the changes and approve or reject before they are persisted.
  </p>
  <p style="color:#8b949e; font-size:0.85rem; margin-top:8px;">
    <strong>Files changed:</strong> {', '.join(fixed_files_list) if fixed_files_list else 'None'}
  </p>
</div>""",
            unsafe_allow_html=True,
        )

        st.markdown("**Code Changes:**")
        st.markdown(
            f'<div class="diff-view">{diff_html}</div>',
            unsafe_allow_html=True,
        )

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
        # ✅ CHANGE 4: HITL state
        "pending_hitl": None,   # dict from hitl_request.json while awaiting human
        # Before/after file comparison shown in end summary
        "fix_summaries": [],    # list of {file_path, original, fixed, tests_passed}
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

with st.sidebar:
    st.markdown("## ⚙ Jira-to-Code")
    st.markdown("---")

    st.markdown("### 🎯 Task")
    level_filter = st.radio("Difficulty", ["All", "Easy", "Medium", "Hard"], horizontal=True)
    if level_filter == "All":
        filtered_keys = task_keys
    else:
        filtered_keys = level_groups[level_filter.lower()]

    selected_task = st.selectbox("Select Task", filtered_keys, key="task_select")

    ticket_text = TASKS.get(selected_task, {}).get("ticket", "")
    st.markdown(
        f'<div class="ticket-box"><strong>📋 Ticket</strong><br/><br/>{ticket_text}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("### 🤖 Agent Config")
    api_base = st.text_input("API Base URL", value=os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
    model_name = st.text_input("Model", value=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct"))

    # ✅ SECURE: HF Token loaded from ./token file, not exposed in UI
    hf_token = _load_hf_token()
    st.caption("🔒 API Key loaded securely from ./token file")

    # ✅ CHANGE 4: HITL toggle
    hitl_enabled = st.toggle("Enable HITL (Human-in-the-Loop)", value=True)
    if hitl_enabled:
        st.caption("⚠️ Agent will pause and ask for your approval before writing fixes.")
    else:
        st.caption("Agent runs fully automated — all proposed changes are auto-approved.")

    st.markdown("---")
    st.markdown("### 🧠 KB Hints")
    custom_ticket = st.text_area(
        "Custom ticket to search KB",
        placeholder="Paste any ticket text to find similar past solutions…",
        key="custom_ticket_input",
        height=100,
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
    st.caption(f"KB size: {kb_count} solution(s)  |  📂 ./corporate_memory")

    st.markdown("---")
    run_btn = st.button("▶  Run Agent", use_container_width=True, disabled=st.session_state.running)
    stop_hint = st.empty()
    if st.session_state.running:
        stop_hint.caption("Agent is running…")

# ---------------------------------------------------------------------------
# Main area layout
# ---------------------------------------------------------------------------

st.markdown("# ⚙ Jira-to-Code · Live Dashboard")
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
        status = "🔍 Awaiting Review…"
    elif end and end["success"]:
        status = "✅ Done"
    elif end and not end["success"]:
        status = "❌ Failed"
    else:
        status = "⏳ Running…"

    metric_step.metric("Steps", total_steps)
    metric_reward.metric("Last Reward", f"{last_reward:.3f}")
    metric_score.metric("Cumulative Score", f"{score:.3f}")
    metric_status.metric("Status", status)


_refresh_metrics()

# ✅ CHANGE 4: HITL review panel (full-width, shown above step feed when active)
hitl_placeholder = st.empty()

main_col, side_col = st.columns([3, 2])

with main_col:
    st.markdown("### 📋 Step Feed")
    steps_container = st.container()

with side_col:
    st.markdown("### 📜 Raw Logs")
    log_box = st.empty()
    st.markdown("### 💡 KB Hints")
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

    # ✅ CHANGE 4: Pass --no-hitl flag if the toggle is off
    cmd = [sys.executable, "inference.py", "--tasks", selected_task]
    if not hitl_enabled:
        cmd.append("--no-hitl")

    t = threading.Thread(
        target=_stream_subprocess, args=(cmd, lq, env_vars), daemon=True
    )
    t.start()
    st.session_state.thread = t

# ---------------------------------------------------------------------------
# ✅ RESTRUCTURED: HITL approval is now POST-SUBMIT ONLY
# Only shown after the episode completes with all tests passing
# ---------------------------------------------------------------------------

# Render HITL gate only when episode is done and tests passed
if st.session_state.end_info and st.session_state.end_info.get("success"):
    if st.session_state.pending_hitl is None:
        # Load the HITL request from inference.py
        review = _load_hitl_request()
        if review:
            st.session_state.pending_hitl = review

# Show HITL panel if pending (only after successful episode)
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
                elif parsed["kind"] == "hitl":
                    # Log HITL events into raw logs (already done above)
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

            task_max = 10 if "easy" in selected_task else 20
            progress = min(len(st.session_state.steps) / task_max, 1.0)
            progress_bar.progress(progress)

        if st.session_state.running:
            time.sleep(0.5)
            st.rerun()

# ---------------------------------------------------------------------------
# End summary banner
# ---------------------------------------------------------------------------
if st.session_state.end_info:
    end = st.session_state.end_info
    color = "#3fb950" if end["success"] else "#f85149"
    icon = "✅" if end["success"] else "❌"
    end_summary_area.markdown(
        f"""
<div style="background:#161b22; border:1px solid {color}; border-radius:8px;
            padding:20px 24px; font-family:'Space Grotesk',sans-serif; margin-top:16px;">
  <h3 style="color:{color}; margin:0 0 8px">{icon} Episode Complete</h3>
  <p style="margin:0; color:#c9d1d9; font-size:0.95rem">
    <strong>Task:</strong> {selected_task} &nbsp;|&nbsp;
    <strong>Steps:</strong> {end['steps']} &nbsp;|&nbsp;
    <strong>Final Score:</strong> {end['score']:.3f} &nbsp;|&nbsp;
    <strong>Success:</strong> {'Yes' if end['success'] else 'No'}
  </p>
</div>""",
        unsafe_allow_html=True,
    )

# Render before/after for every file the agent changed (always, once end is known)
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