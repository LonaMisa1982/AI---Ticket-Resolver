---
title: Jira-To-Code Agent Environment
emoji: 🛠️
colorFrom: indigo
colorTo: blue
sdk: docker
pinned: false
license: mit
app_port: 7860
---

# 🚀 Jira-To-Code: The Ultimate RL Coding Environment

[![OpenEnv Validated](https://img.shields.io/badge/OpenEnv-Validated-green.svg)](https://github.com/meta-pytorch/OpenEnv)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Jira-To-Code** is a high-fidelity Reinforcement Learning (RL) environment designed for the **Meta/Hugging Face OpenEnv Hackathon**. It challenges AI agents to act as Senior Software Engineers by resolving real-world Jira tickets within a secure, sandboxed codebase.

---

## 🌟 Key Features

*   **⚡ ReAct Ready**: Built-in support for Thought-Action-Observation loops with mandatory `thought` key enforcing reasoning before every action.
*   **🧠 Episodic Memory**: Maintains full conversational history (`messages[]`) across the episode with a sliding-window trim to respect context limits.
*   **📊 22 Diverse Tasks**: From simple bug fixes to complex architecture, concurrency, and security challenges.
*   **📈 Rich Reward Shaping**: Partial credit for passing tests, step penalties for efficiency, and shaping rewards for active coding — all strictly bounded to `[0.01, 0.99]`.
*   **🛡️ Robust Parsing**: Three-layer JSON extraction with markdown-fence stripping, `json.loads`, and brace-depth scanning — plus self-correction prompt injection on parse failure.
*   **🗺️ Map-Reduce File Discovery**: A fast mapper LLM triages the file tree to identify 3–5 focus files; the main agent then reasons only over those files.
*   **💾 Persistent Knowledge Base**: Human-approved solutions are vectorised into ChromaDB (`./corporate_memory`) and semantically retrieved as historical context for future tasks.
*   **🔍 Human-in-the-Loop (HITL)**: A post-submit approval gate — the agent pauses after all tests pass, a colour-coded diff is shown in the Streamlit UI, and the human approves or rejects before changes are persisted.

---

## 🏗️ System Architecture

### Agent Phases (per episode)

| Phase | What Happens |
| :--- | :--- |
| **0A — Map** | `map_relevant_files()` calls a fast LLM to identify 3–5 focus files from the full file tree + ticket. |
| **0B — Reduce** | `build_reducer_system_prompt()` injects focus files + ChromaDB KB hints into the system prompt. |
| **1 — ReAct Loop** | Agent reasons (`thought`) and acts (`action_type`) step-by-step; observation + reward appended to `messages[]`. |
| **2 — Submit Gate** | On `submit` with all tests passing, a `pending_review` diff is created and the episode pauses for HITL. |
| **3 — HITL Approval** | Human approves or rejects via Streamlit. On approval, fixes are persisted to the source task directory and the solution is captured in ChromaDB. |

### Environment Actions

| Action | Description |
| :--- | :--- |
| `list_files` | Explore the workspace directory structure via `os.walk()`. |
| `read_file` | Read the content of a specific file (path-traversal guarded). |
| `write_file` | Write or overwrite code in the isolated sandbox workspace. |
| `run_tests` | Execute `pytest -v` and receive full traceback output + parsed pass/fail counts. |
| `submit` | Run final tests; if all pass, pause for HITL review; if any fail, end episode immediately. |
| `human_approved_final` | *(Internal)* Write fixes to the source task directory and capture the solution in ChromaDB. |
| `human_rejected_final` | *(Internal)* Discard pending changes; mark episode as failed. |

### 🔍 Human-in-the-Loop (HITL) Flow

HITL is triggered **only after a successful `submit`** (all tests passing). The flow is file-based IPC between `inference.py` and `streamlit_app.py`:

1. `env.py` sets `pending_review = {diff, test_results, fixed_files}` and sets `done=False`.
2. `inference.py` detects `info["awaiting_human_review"]` and writes `hitl_request.json` to disk.
3. The Streamlit UI detects the file and renders a **colour-coded unified diff** with **✅ Approve & Persist** / **❌ Reject & Discard** buttons.
4. The human's decision is written to `hitl_response.json`; `inference.py` polls every 1.5 s (300 s timeout → auto-reject).
5. On **approval**: fixed files are written back to `src/jira_to_code/tasks/<task>/` (persisted permanently) and `_capture_knowledge()` upserts the solution into ChromaDB.
6. On **rejection**: pending changes are discarded and the episode ends as failed.

> **Tip**: Disable HITL with the toggle in the Streamlit sidebar or the `--no-hitl` CLI flag for fully automated batch runs.

### 💾 Knowledge Base (RAG) Pipeline

```
Ticket text  ──►  ChromaDB cosine similarity search  ──►  Top-3 past solutions
                        (./corporate_memory)
                                │
                                ▼
              "## Historical Context" block injected into system prompt
                                │
                                ▼
                  Agent reasons with historical patterns
                                │
                    (after human-approved submit)
                                │
                                ▼
          _capture_knowledge(): LLM generates root-cause summary
                   ──►  ChromaDB upsert (code + metadata)
```

---

## 🎯 Available Tasks (22 Total)

| Task ID | Level | Objective |
| :--- | :--- | :--- |
| `easy` | Easy | Fix off-by-one bug in `calculator.add()`. |
| `easy_2` | Easy | Fix case-sensitivity bug in `string_utils.count_vowels()`. |
| `easy_3` | Easy | API KeyError: use `.get()` with fallback for missing `phone_number`. |
| `easy_4` | Easy | Off-by-One Pagination: Fix math index logic in `get_page_bounds`. |
| `easy_5` | Easy | FastAPI Route Typo: Align `user_id` route param with function arg. |
| `medium` | Medium | Implement `format_user_data()` dictionary mapping specs. |
| `medium_2` | Medium | Implement complex `Email` and `Password` validation logic. |
| `medium_3` | Medium | Missing Auth Middleware: Apply `@require_auth` to `/api/billing`. |
| `medium_4` | Medium | ORM N+1 Problem: Rewrite fetches to use JOINs (`select_related`). |
| `medium_5` | Medium | Regex Validation: Fix email regex to allow plus sign (`+`). |
| `medium_6` | Medium | Error Handling: Add try/except fallback for currency rate timeouts. |
| `medium_7` | Medium | Stale Cache: Add Redis invalidation to `update_user_profile`. |
| `medium_8` | Medium | Timezone Naive: Make naive datetimes UTC aware. |
| `medium_9` | Medium | State Machine: Add transition guards (`CANCELLED` → `SHIPPED`). |
| `medium_10` | Medium | Config Merge: Fix recursion logic for nested dict merges. |
| `hard` | Hard | Implement `LRUCache` with O(1) time complexity. |
| `hard_2` | Hard | Implement `DirectedGraph` with BFS/DFS and Topological Sort. |
| `hard_3` | Hard | Circular Dependency: Refactor `models/utils/config` via `base.py`. |
| `hard_4` | Hard | Race Condition: Refactor threaded worker to use `queue.Queue`. |
| `hard_5` | Hard | OOM Generator: Rewrite `readlines()` loop to use `yield` generators. |
| `hard_6` | Hard | Implementation: Code `StripeGateway` matching `PaymentGateway` ABC. |
| `hard_7` | Hard | Async Deadlock: Fix lock release safety using async context managers. |

---

## 🚀 Getting Started

### 1. Local Setup
```bash
# Clone the repository
git clone https://huggingface.co/spaces/Navigam/jira-to-code
cd jira-to-code

# Create and activate environment
uv venv
source .venv/bin/activate  # Or .venv\Scripts\activate on Windows

# Install dependencies
uv pip install -e .
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token         # Required — used as API key for inference
API_BASE_URL=https://router.huggingface.co/v1  # Default HuggingFace router
MODEL_NAME=Qwen/Qwen2.5-7B-Instruct    # Main reasoning model
MAPPER_MODEL=Qwen/Qwen2.5-7B-Instruct  # Fast model for file triage (can differ)
```

### 3. Run the Streamlit UI
```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501`. Use the sidebar to:
- Select a task and difficulty filter
- Configure your API base URL, model, and HF token
- Toggle HITL on/off
- Search the knowledge base with a custom ticket query

Click **▶ Run Agent** to start. The live dashboard shows each step's thought, action badge, and colour-coded reward in real time. When HITL is enabled, the UI will pause after a successful submit and display the diff for your review.

### 4. Run Inference from CLI
```bash
# Run a random sample (1 easy, 1 medium, 1 hard) — HITL enabled by default
uv run python inference.py

# Run specific tasks
uv run python inference.py --tasks easy_2,medium,hard_4

# Disable HITL for fully automated batch runs
uv run python inference.py --tasks easy --no-hitl
```

### 5. Docker Deployment
```bash
docker build -t jira-to-code .
docker run -p 7860:7860 jira-to-code
```

---

## 🛠️ Deep Dive: Design & Rubric Alignment

### 🎨 Creativity & Novelty

*   **Real-World Software Engineering Domain**: While most RL environments focus on games or simplified logic, **Jira-To-Code** provides a high-stakes, documentation-driven coding domain. Agents interpret edge cases from docstrings (e.g., case-insensitivity in vowel counting) exactly as real developers do.
*   **Non-Sparse Reward Mechanics**: We move away from binary "Pass/Fail" signals. The environment rewards "Progress Toward Solution" by parsing intermediate test results.
*   **Map-Reduce Codebase Discovery**: A fast mapper LLM pre-filters the workspace to the most relevant files, separating the cheap "what to look at" decision from the expensive "how to fix it" reasoning.
*   **HITL as a First-Class Citizen**: Human review is a native, post-submit approval gate with its own reward signal (`+1.0`), diff viewer, and source-directory write-back — not a bolted-on afterthought.
*   **Self-Improving Knowledge Base**: Each human-approved solution is vectorised by ChromaDB and retrieved semantically for future tasks. The environment learns from its own history.

### 📈 Reward Signal Design

The environment provides a dense, informative reward signal. All rewards are strictly bounded to `[0.01, 0.99]` as required by the OpenEnv spec:

| Action / Condition | Reward | Notes |
| :--- | :--- | :--- |
| `list_files` / `read_file` | `0.0` base | No direct reward; encourages moving to action |
| `write_file` | `+0.05` | Shaping reward for active coding |
| `run_tests` — all pass | `0.1 + 0.4 × (p/t)` | Up to `0.5` max |
| `run_tests` — partial pass | `0.1 × (p/t)` | Proportional to tests passed |
| `run_tests` — crash/timeout | `-0.1` | Penalises broken code |
| `submit` — all pass | `0.9` | Held pending HITL approval |
| `human_approved_final` | `1.0` (clipped `0.99`) | Maximum; human-verified fix |
| Steps 1–3 (shaping) | `+0.02` bonus | Rewards early orientation |
| Steps 4+ (efficiency) | `−0.01` penalty | Discourages excessive wandering |

### 🧱 Episode & Workspace Design

*   **Isolation & Reset**: Every `reset()` call generates a **cryptographically unique temp directory** (`tempfile.mkdtemp()`), deep-copies the task source, and snapshots all non-test Python files into `original_files` for diff generation.
*   **Path-Traversal Guard**: Every file operation resolves the absolute path and checks `target_path.is_relative_to(workspace_path)` before proceeding.
*   **Source Write-Back**: On HITL approval, `human_approved_final` writes fixed files back to `src/jira_to_code/tasks/<task>/`, persisting the fix beyond the episode's temp directory.
*   **Atomic Boundaries**: An episode concludes when the agent calls `submit` (and HITL resolves) or reaches `MAX_STEPS` (10 for easy tasks, 20 for medium/hard).
*   **Deterministic Grading**: Graders are `pytest` unit tests that are immutable within the container, ensuring fully reproducible scoring.

### 🖥️ Streamlit Dashboard

The live dashboard (`streamlit_app.py`) provides:

*   **Step Feed**: Each step renders as a card showing the agent's **thought** (blue italic), **action badge** (colour-coded by type), and **reward** (green ≥ 0.4 / amber ≥ 0.1 / red < 0.1).
*   **Raw Log Terminal**: Streams the full subprocess output, parsed for `[STEP]`, `[THOUGHT]`, `[KB]`, `[MAPPER]`, and `[HITL]` structured log lines.
*   **KB Hints Panel**: Query the ChromaDB knowledge base with any ticket text to surface similar past solutions before running the agent.
*   **Before & After Diff**: After the episode, a side-by-side panel highlights changed lines in every file the agent modified.
*   **HITL Approval Gate**: A full-width panel with a colour-coded unified diff and **Approve / Reject** buttons — only shown after a successful submit.

---

## 🏆 Scoring Rubric Alignment


*   **Real-world Utility**: Models a developer's full daily workflow — discovery, coding, testing, review, and knowledge capture.
*   **Task/Grader Quality**: Deterministic `pytest` grading with linear partial credit across 22 well-scoped tasks.
*   **Environment Design**: Gymnasium-style API (`reset` / `step` / `close`) with a comprehensive `JiraCodeObservation` space and Pydantic-typed actions.
*   **Code Quality**: Passes `openenv validate` and follows strict Pydantic typing throughout.
*   **Safety**: HITL gate prevents unreviewed code from being persisted; path-traversal guard prevents sandbox escape.

---

## 📜 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

