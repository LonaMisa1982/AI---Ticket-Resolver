# inference.py — ReAct Agent for Jira-to-Code Environment
#
# Architecture:
#   Phase 0: Map-Reduce Discovery — Fast LLM maps relevant files; Smart LLM reduces with focus
#   Phase 1: Episodic Memory — persistent messages[] across the episode
#   Phase 2: ReAct Pattern — "thought" key forces reasoning before action
#   Phase 3: Robust Parsing — JSON extraction with markdown-fence stripping
#   Phase 4: Self-Correction — negative rewards inject corrective prompts
#   Phase 5: Multi-Task Loop — evaluates all tasks in one run
#   Phase 6: KB Hints — ChromaDB query injects historical solutions as context (Semantic Retrieval)
#   Phase 7: HITL — agent may emit request_human_review; loop pauses for human signal

import argparse
import json
import os
import re
import textwrap
import time
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# Our environment for local/direct testing
from server.env import JiraToCodeEnv
from src.jira_to_code.models import JiraCodeAction

# --- HACKATHON MANDATORY CONFIGURATION ---
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
MAPPER_MODEL = os.getenv("MAPPER_MODEL") or "Qwen/Qwen2.5-7B-Instruct"   # Fast model for discovery
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "jira-to-code"
SUCCESS_SCORE_THRESHOLD = 0.9
ALL_TASKS = list(JiraToCodeEnv.TASKS.keys())
MAX_HISTORY_MESSAGES = 30
MAX_RETRIES = 5
RETRY_BASE_DELAY = 2

# ✅ CHANGE 3 (Semantic Retrieval): KB now uses ./corporate_memory to match env.py
try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

KB_DIR = "./corporate_memory"          # ← matches env.py
KB_COLLECTION = "jira_solutions"
_MAX_KB_HINTS = 3                      # Maximum number of historical hints to inject

# ✅ CHANGE 4: HITL signal file — inference.py writes this when pausing for review,
# streamlit_app.py writes the decision back so the loop can continue.
HITL_REQUEST_FILE = "./hitl_request.json"
HITL_RESPONSE_FILE = "./hitl_response.json"
HITL_POLL_INTERVAL = 1.5   # seconds between polls
HITL_TIMEOUT = 300          # seconds before auto-reject


def _get_kb_collection():
    """Return the ChromaDB collection, or None if unavailable."""
    if not _CHROMA_AVAILABLE:
        return None
    try:
        client = chromadb.PersistentClient(path=KB_DIR)
        return client.get_or_create_collection(KB_COLLECTION)
    except Exception as exc:
        print(f"[KB] Could not connect to ChromaDB: {exc}", flush=True)
        return None


# ✅ CHANGE 3 (Semantic Retrieval): Vectorise incoming ticket → query ChromaDB →
# inject "Historical Context" block into SYSTEM_PROMPT before the ReAct loop starts.
def query_kb_for_hints(ticket: str) -> str:
    """
    Query the ChromaDB KB for solutions similar to *ticket*.

    The ticket text is used as the query vector (ChromaDB embeds it internally).
    Returns a formatted "Historical Context" block (empty string if nothing found).
    """
    collection = _get_kb_collection()
    if collection is None:
        return ""
    try:
        count = collection.count()
        if count == 0:
            return ""
        n = min(_MAX_KB_HINTS, count)
        # Semantic search: ChromaDB vectorises ticket and finds nearest neighbours
        results = collection.query(query_texts=[ticket], n_results=n)
        hints_parts = [
            "## Historical Context (semantically similar past solutions)\n"
            "The following resolved tickets were retrieved from corporate memory.\n"
            "Use them as inspiration — do NOT copy blindly.\n"
        ]
        for i, (doc, meta) in enumerate(
            zip(results["documents"][0], results["metadatas"][0])
        ):
            hints_parts.append(
                f"### Past Fix {i + 1} — {meta.get('ticket_id', 'unknown')}\n"
                f"**Root-cause summary**: {meta.get('summary', 'N/A')}\n"
                f"**Code**:\n```python\n{doc}\n```\n"
            )
        return "\n".join(hints_parts)
    except Exception as exc:
        print(f"[KB] Semantic query failed: {exc}", flush=True)
        return ""


# ---------------------------------------------------------------------------
# PHASE 0 — MAP-REDUCE DISCOVERY
# ---------------------------------------------------------------------------

MAPPER_SYSTEM_PROMPT = textwrap.dedent("""\
You are a senior software engineer doing a quick triage of a Jira ticket.
Given a list of files in a workspace and a Jira ticket description, identify the
3-5 most relevant files the developer should focus on to resolve the ticket.

Respond ONLY with a JSON object in exactly this format (no markdown fences, no extra text):
{
  "thought": "Brief reasoning about which files matter and why",
  "focus_files": ["file1.py", "file2.py", "file3.py"]
}
""").strip()


def map_relevant_files(
    client: OpenAI,
    file_tree: List[str],
    jira_ticket: str,
) -> List[str]:
    """
    Phase A (Mapper): Ask a fast LLM which files are most relevant for the ticket.
    Returns a list of file paths (falls back to full file_tree on any error).
    """
    if not file_tree:
        return []

    user_content = (
        f"Jira Ticket:\n{jira_ticket}\n\n"
        f"Workspace files:\n" + "\n".join(f"  - {f}" for f in file_tree)
    )

    for attempt in range(MAX_RETRIES):
        try:
            completion = client.chat.completions.create(
                model=MAPPER_MODEL,
                messages=[
                    {"role": "system", "content": MAPPER_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.1,
                max_tokens=512,
            )
            raw = (completion.choices[0].message.content or "").strip()
            parsed = extract_json(raw)
            focus = parsed.get("focus_files", [])
            valid = [f for f in focus if f in file_tree]
            if valid:
                print(f"[MAPPER] Focus files identified: {valid}", flush=True)
                return valid
        except Exception as exc:
            exc_str = str(exc)
            is_rate_limit = "429" in exc_str or "rate" in exc_str.lower()
            if is_rate_limit and attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)
                print(f"  [MAPPER RATE LIMIT] Retry {attempt + 1}/{MAX_RETRIES} in {delay}s…", flush=True)
                time.sleep(delay)
                continue
            print(f"[MAPPER] Error during file mapping: {exc}", flush=True)
            break

    print("[MAPPER] Falling back to full file tree.", flush=True)
    return file_tree


def build_reducer_system_prompt(
    focus_files: List[str],
    kb_hints: str = "",
) -> str:
    """
    Phase B (Reducer): Build the system prompt with:
      - Focus Files section from the Mapper
      - ✅ CHANGE 3: Historical Context block from semantic ChromaDB retrieval
    """
    focus_section = ""
    if focus_files:
        file_list = "\n".join(f"  - {f}" for f in focus_files)
        focus_section = textwrap.dedent(f"""
## Focus Files (identified by fast triage)
The following files are most likely relevant to this ticket.
Start by reading these before exploring the rest of the workspace:
{file_list}
""").strip()

    # ✅ CHANGE 3: Historical Context injected before the ReAct loop starts
    kb_section = ""
    if kb_hints:
        kb_section = f"\n\n{kb_hints}"

    base_prompt = textwrap.dedent("""\
You are an expert software engineer resolving Jira tickets.
You operate in a sandboxed workspace. You can read files, write code, list files, run tests, and submit your solution.

## Rules
1. ALWAYS respond with ONLY a valid JSON object. No markdown fences, no explanations outside JSON.
2. You MUST include a "thought" key FIRST to reason about your plan before acting.
3. Work step-by-step: list files, read the code, understand the bug/requirement, write a fix, run tests, then submit.
4. If tests fail, carefully read the traceback and fix your code before re-submitting.
5. Only use "submit" when you are confident all tests will pass.
6. Be efficient — each step has a small penalty. Aim to solve in the fewest steps possible.
7. Read the test file to understand exactly what is expected before writing code.
8. If you want a human to review your proposed change before it is applied, use "request_human_review".
   The change will NOT be written to disk until a human approves it.

## Valid action_types
- "list_files"            — List all files in the workspace
- "read_file"             — Read a file's contents (requires file_path)
- "write_file"            — Write/overwrite a file directly (requires file_path and content)
- "request_human_review"  — Propose a change for human approval before writing (requires file_path and content)
- "run_tests"             — Run pytest on the workspace
- "submit"                — Final submission, runs tests and ends the episode

## Reward Structure
- list_files / read_file: 0.01 (initial exploration)
- write_file: +0.05
- request_human_review: +0.02 (pending approval)
- human_approved / write applied: +0.10
- run_tests (all pass): +0.5 | partial: proportional | crash: 0.01
- submit (all pass): +1.0 | partial: proportional

## JSON Schema
{
  "thought": "Your reasoning about what to do next and why",
  "action_type": "one of the action_types above",
  "file_path": "string or null",
  "content": "string or null"
}

## Strategy Guide
1. list_files → see workspace structure.
2. Read the test file to understand exact expected behaviour.
3. Read the source file to understand the current (buggy/incomplete) code.
4. Use request_human_review with your proposed fix (preferred) OR write_file directly.
5. After human approves (or you used write_file), run_tests to verify.
6. If tests pass → submit. If not → read the error, fix, and retry.
""").strip()

    sections = [base_prompt]
    if focus_section:
        sections.append(focus_section)
    if kb_section:
        sections.append(kb_section)
    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# PHASE 4 & 7: HITL file-based synchronisation helpers
# ---------------------------------------------------------------------------

def write_hitl_request(pending_review: dict) -> None:
    """Write the pending review to disk so Streamlit can display it."""
    import json as _json
    with open(HITL_REQUEST_FILE, "w") as f:
        _json.dump(pending_review, f, indent=2)
    print(f"[HITL] Review request written to '{HITL_REQUEST_FILE}'", flush=True)


def poll_hitl_response(timeout: float = HITL_TIMEOUT) -> Optional[str]:
    """
    Block until streamlit_app.py writes a HITL response file.
    Returns "approved" | "rejected" | None (timeout).
    """
    import json as _json
    # Clean up stale response from a previous round
    if os.path.exists(HITL_RESPONSE_FILE):
        os.remove(HITL_RESPONSE_FILE)

    elapsed = 0.0
    while elapsed < timeout:
        if os.path.exists(HITL_RESPONSE_FILE):
            try:
                with open(HITL_RESPONSE_FILE) as f:
                    data = _json.load(f)
                decision = data.get("decision")
                os.remove(HITL_RESPONSE_FILE)
                # Also clean up the request file
                if os.path.exists(HITL_REQUEST_FILE):
                    os.remove(HITL_REQUEST_FILE)
                print(f"[HITL] Human decision received: {decision}", flush=True)
                return decision
            except Exception as exc:
                print(f"[HITL] Could not read response file: {exc}", flush=True)
        time.sleep(HITL_POLL_INTERVAL)
        elapsed += HITL_POLL_INTERVAL

    print(f"[HITL] Timed out waiting for human review ({timeout}s). Auto-rejecting.", flush=True)
    if os.path.exists(HITL_REQUEST_FILE):
        os.remove(HITL_REQUEST_FILE)
    return None


# --- MANDATORY LOGGING FUNCTIONS ---
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# --- PHASE 3: ROBUST JSON PARSING ---
def extract_json(raw_text: str) -> dict:
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    start = cleaned.find('{')
    if start == -1:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape_next = False
    for i in range(start, len(cleaned)):
        c = cleaned[i]
        if escape_next:
            escape_next = False
            continue
        if c == '\\' and in_string:
            escape_next = True
            continue
        if c == '"' and not escape_next:
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                return json.loads(cleaned[start:i + 1])

    raise ValueError("Unbalanced braces in JSON")


def parse_action(raw_text: str) -> JiraCodeAction:
    """Parse LLM output into a JiraCodeAction, extracting JSON robustly."""
    action_dict = extract_json(raw_text)
    action_dict.pop("thought", None)
    return JiraCodeAction(**action_dict)


# --- PHASE 1 & 2: BUILD OBSERVATION MESSAGE ---
def build_observation_message(step: int, obs, reward: float) -> str:
    parts = [
        f"--- Step {step} Observation ---",
        f"Ticket: {obs.jira_ticket}",
        f"Files in workspace: {', '.join(obs.file_tree) if obs.file_tree else 'None'}",
    ]
    if obs.current_file_content is not None:
        parts.append(f"File Content:\n```\n{obs.current_file_content}\n```")
    if obs.test_output:
        parts.append(f"Test Output:\n```\n{obs.test_output}\n```")
    if obs.error:
        parts.append(f"Error: {obs.error}")
    parts.append(f"Reward: {reward:.2f}")
    parts.append("Respond with your next action as JSON.")
    return "\n".join(parts)


def trim_history(messages: list, max_messages: int = MAX_HISTORY_MESSAGES) -> None:
    while len(messages) > max_messages:
        messages.pop(1)


# --- MAIN AGENT LOOP FOR ONE TASK ---
def run_agent_episode(
    client: OpenAI,
    task_name: str,
    extra_hints: str = "",
    hitl_enabled: bool = True,
) -> tuple:
    """
    Run a full agent episode for one task.

    Args:
        client       : OpenAI-compatible client
        task_name    : Key into JiraToCodeEnv.TASKS
        extra_hints  : Optional pre-fetched KB hints to inject (e.g. from Streamlit UI)
        hitl_enabled : If True, pause for human review when agent uses request_human_review.
                       If False (e.g. CLI / batch mode), auto-approve all reviews.

    Returns:
        (score, steps_taken, rewards, success)
    """
    os.environ["JIRA_TASK_LEVEL"] = task_name
    env = JiraToCodeEnv()

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()

        # ------------------------------------------------------------------
        # Phase 0A — MAP: Fast LLM identifies the most relevant files
        # ------------------------------------------------------------------
        print("[MAP-REDUCE] Running Mapper phase…", flush=True)
        focus_files = map_relevant_files(client, obs.file_tree, obs.jira_ticket)

        # ------------------------------------------------------------------
        # ✅ CHANGE 3 (Semantic Retrieval): Vectorise ticket → query ChromaDB →
        # inject Historical Context block BEFORE the ReAct loop starts.
        # ------------------------------------------------------------------
        kb_hints = extra_hints or query_kb_for_hints(obs.jira_ticket)
        if kb_hints:
            print("[KB] Injecting historical context into system prompt.", flush=True)

        # ------------------------------------------------------------------
        # Phase 0B — REDUCE: Build enriched system prompt for the smart model
        # ------------------------------------------------------------------
        system_prompt = build_reducer_system_prompt(focus_files, kb_hints)

        # ✅ INCREASED STEP LIMITS: Easy tasks need more steps for exploration + fixing + testing
        task_max_steps = 15 if "easy" in task_name else 25

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_observation_message(0, obs, 0.0)},
        ]

        for step in range(1, task_max_steps + 1):
            trim_history(messages)

            raw_text = None
            for attempt in range(MAX_RETRIES):
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=2048,
                    )
                    raw_text = (completion.choices[0].message.content or "").strip()
                    break
                except Exception as exc:
                    exc_str = str(exc)
                    is_rate_limit = "429" in exc_str or "rate" in exc_str.lower()
                    if is_rate_limit and attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        print(f"  [RATE LIMIT] Retry {attempt + 1}/{MAX_RETRIES} in {delay}s…", flush=True)
                        time.sleep(delay)
                        continue
                    messages.append({
                        "role": "user",
                        "content": f"API ERROR: {exc}. Please try again with a valid JSON action.",
                    })
                    log_step(step=step, action=f"API_ERROR: {exc}", reward=0.0, done=False, error=exc_str)
                    rewards.append(0.0)
                    steps_taken = step
                    break

            if raw_text is None:
                continue

            messages.append({"role": "assistant", "content": raw_text})

            try:
                action = parse_action(raw_text)
                action_log = action.action_type

                try:
                    thought_data = extract_json(raw_text)
                    thought = thought_data.get("thought", "")
                    if thought:
                        print(f"[THOUGHT] step={step} thought={thought!r}", flush=True)
                except Exception:
                    pass
            except Exception as exc:
                action = JiraCodeAction(action_type="list_files")
                action_log = "PARSE_ERROR"
                messages.append({
                    "role": "user",
                    "content": (
                        f"ERROR: Your last response was not valid JSON.\n"
                        f"Parse error: {exc}\n"
                        f"You MUST respond with ONLY a valid JSON object. "
                        f"No markdown, no explanations.\nTry again."
                    ),
                })

            obs, reward, done, info = env.step(action)
            error = obs.error

            reward = max(reward, 0.01)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_log, reward=reward, done=done, error=error)

            # ✅ BREAK on completion OR when awaiting HITL approval
            if done or info.get("awaiting_human_review"):
                break

            obs_message = build_observation_message(step, obs, reward)

            if reward <= 0.01 or obs.error:
                obs_message += (
                    f"\n\nLOW/NEGATIVE RESULT (reward={reward:.2f})."
                    f"\nCarefully analyze the error/test output above."
                    f"\nIdentify the root cause and write a fix."
                    f"\nDo NOT repeat the same action that just failed."
                )
            elif reward >= 0.4:
                obs_message += (
                    "\n\nTests are passing! If all tests pass, use 'submit' to finalize."
                )

            messages.append({"role": "user", "content": obs_message})

        # ✅ CHECK: Did the agent complete the task, or hit the step limit?
        if not done:
            print(
                f"[WARNING] Agent reached step limit ({steps_taken}/{task_max_steps}) "
                f"without calling submit(). Episode may be incomplete.",
                flush=True,
            )

        score = min(max(sum(rewards), 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

        # ✅ RESTRUCTURED: HITL Approval Gate — only after successful submit
        # If the agent submitted with all tests passing, pause for final human review
        if env.pending_review is not None and info.get("awaiting_human_review"):
            print("[HITL] Waiting for final human approval of the completed solution…", flush=True)
            
            if hitl_enabled:
                # Write the diff to disk so Streamlit can display it
                write_hitl_request(env.pending_review)
                # ✅ FIX: Emit a structured log line BEFORE blocking so Streamlit
                # can detect the pending review while the subprocess is still alive.
                import json as _json
                pending_payload = _json.dumps(env.pending_review, separators=(",", ":"))
                print(f"[HITL_PENDING] {pending_payload}", flush=True)
                print("[HITL] Waiting for human to approve or reject the final solution…", flush=True)
                decision = poll_hitl_response()
            else:
                # Batch / CLI mode: auto-approve
                decision = "approved"
                print("[HITL] HITL disabled — auto-approving final solution.", flush=True)

            if decision == "approved":
                approval_action = JiraCodeAction(action_type="human_approved_final")
                obs, reward, done, info = env.step(approval_action)
                success = True
                score = min(max(sum(rewards) + reward, 0.01), 0.99)
                print("[HITL] Final approval granted. Solution persisted.", flush=True)
            else:
                rejection_action = JiraCodeAction(action_type="human_rejected_final")
                obs, reward, done, info = env.step(rejection_action)
                success = False
                score = min(max(sum(rewards) + reward, 0.01), 0.99)
                print("[HITL] Final approval denied. Changes not persisted.", flush=True)

    finally:
        env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return score, steps_taken, rewards, success


# --- PHASE 5: MULTI-TASK EVALUATION ---
def main() -> None:
    parser = argparse.ArgumentParser(description="Jira-to-Code ReAct Agent")
    parser.add_argument(
        "--tasks",
        type=str,
        default=None,
        help=(
            "Comma-separated list of tasks to run. "
            f"Available: {', '.join(ALL_TASKS)}. "
            "Default: 1 easy, 1 medium, 1 hard sampled randomly."
        ),
    )
    parser.add_argument(
        "--no-hitl",
        action="store_true",
        default=False,
        help="Disable HITL (auto-approve all agent review requests). Useful for batch runs.",
    )
    args = parser.parse_args()

    import random

    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
        invalid = [t for t in tasks if t not in ALL_TASKS]
        if invalid:
            print(f"ERROR: Unknown tasks: {invalid}", flush=True)
            print(f"Available: {ALL_TASKS}", flush=True)
            return
    else:
        easies = [t for t in ALL_TASKS if "easy" in t]
        mediums = [t for t in ALL_TASKS if "medium" in t]
        hards = [t for t in ALL_TASKS if "hard" in t]

        tasks = []
        if easies:
            tasks.append(random.choice(easies))
        if mediums:
            tasks.append(random.choice(mediums))
        if hards:
            tasks.append(random.choice(hards))

    print(f"Running tasks: {tasks}", flush=True)
    hitl_enabled = not args.no_hitl
    print(f"HITL mode: {'enabled' if hitl_enabled else 'disabled (auto-approve)'}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    total_score = 0.0
    results = []

    for task in tasks:
        score, steps, rewards, success = run_agent_episode(
            client, task, hitl_enabled=hitl_enabled
        )
        results.append({"task": task, "score": score, "steps": steps, "success": success})
        total_score += score

        print("Waiting 20 seconds before next task to respect API limits…", flush=True)
        time.sleep(20)

    print("\n" + "=" * 50, flush=True)
    print("EVALUATION SUMMARY", flush=True)
    print("=" * 50, flush=True)
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        print(
            f"  {r['task']:10s} | score={r['score']:.3f} | "
            f"steps={r['steps']:2d} | {status}",
            flush=True,
        )
    avg_score = total_score / len(tasks)
    print(f"  {'AVERAGE':10s} | score={avg_score:.3f}", flush=True)
    print(f"  {'TOTAL':10s} | score={total_score:.3f} / {len(tasks):.1f}", flush=True)
    print("=" * 50, flush=True)


if __name__ == "__main__":
    main()