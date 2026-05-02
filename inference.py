# inference.py — ReAct Agent for Jira-to-Code Environment
#
# Architecture:
#   Phase 1: Episodic Memory — persistent messages[] across the episode
#   Phase 2: ReAct Pattern — "thought" key forces reasoning before action
#   Phase 3: Robust Parsing — JSON extraction with markdown-fence stripping
#   Phase 4: Self-Correction — negative rewards inject corrective prompts
#   Phase 5: Multi-Task Loop — evaluates all 6 tasks in one run

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
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

BENCHMARK = "jira-to-code"
# MAX_STEPS is now dynamic based on task level
SUCCESS_SCORE_THRESHOLD = 0.9  # Account for step penalties
ALL_TASKS = list(JiraToCodeEnv.TASKS.keys())
MAX_HISTORY_MESSAGES = 30  # Context-window safety: trim if exceeded
MAX_RETRIES = 5            # Rate limit retry attempts
RETRY_BASE_DELAY = 2       # Base delay in seconds for exponential backoff

# --- SYSTEM PROMPT (ReAct + Reward-Aware) ---
SYSTEM_PROMPT = textwrap.dedent("""\
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

## Valid action_types
- "list_files" — List all files in the workspace (file_path and content should be null)
- "read_file" — Read a file's contents (requires file_path, content should be null)
- "write_file" — Write/overwrite a file (requires file_path and content)
- "run_tests" — Run pytest on the workspace (file_path and content should be null)
- "submit" — Final submission, runs tests and ends the episode (file_path and content should be null)

## Reward Structure
- list_files / read_file: 0.01 (initial exploration)
- write_file: +0.05 (reward for taking action)
- run_tests (all pass): +0.5 | run_tests (partial): proportional | run_tests (crash): 0.01
- submit (all pass): +1.0 | submit (partial): proportional
- Every step: 0.01 minimum reward (be efficient!)

## JSON Schema
{
  "thought": "Your reasoning about what to do next and why",
  "action_type": "one of: list_files, read_file, write_file, run_tests, submit",
  "file_path": "string or null",
  "content": "string or null"
}

## Strategy Guide
1. First, list_files to see the workspace structure.
2. Read the test file to understand the exact expected behavior.
3. Read the source file to understand the current (buggy/incomplete) code.
4. Write the fix/implementation.
5. Run tests to verify.
6. If tests pass, submit. If not, read the error, fix, and retry.
""").strip()


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
    """
    Extract a JSON object from LLM output, handling:
    - Markdown code fences (```json ... ```)
    - Leading/trailing whitespace and text
    - Nested braces via brace-counting
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
    cleaned = re.sub(r'\s*```\s*$', '', cleaned)
    cleaned = cleaned.strip()

    # Try direct parse first
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: find the first balanced {...} block via brace counting
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
    # Remove the 'thought' key — it's for reasoning only, not part of the action model
    action_dict.pop("thought", None)
    return JiraCodeAction(**action_dict)


# --- PHASE 1 & 2: BUILD OBSERVATION MESSAGE ---
def build_observation_message(step: int, obs, reward: float) -> str:
    """Format environment observation as a user message for the conversation history."""
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
    """Trim oldest non-system messages if history exceeds max to avoid context overflow."""
    while len(messages) > max_messages:
        # Keep index 0 (system prompt), remove index 1
        messages.pop(1)


# --- MAIN AGENT LOOP FOR ONE TASK ---
def run_agent_episode(client: OpenAI, task_name: str) -> tuple:
    """
    Run a full agent episode for one task.
    Returns: (score, steps_taken, rewards, success)
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

        task_max_steps = 10 if "easy" in task_name else 20

        # Phase 1: Episodic memory — persistent conversation history
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_observation_message(0, obs, 0.0)},
        ]

        for step in range(1, task_max_steps + 1):
            trim_history(messages)

            # Call the LLM with rate-limit retry + exponential backoff
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
                    break  # Success
                except Exception as exc:
                    exc_str = str(exc)
                    is_rate_limit = "429" in exc_str or "rate" in exc_str.lower()
                    if is_rate_limit and attempt < MAX_RETRIES - 1:
                        delay = RETRY_BASE_DELAY * (2 ** attempt)
                        print(f"  [RATE LIMIT] Retry {attempt + 1}/{MAX_RETRIES} in {delay}s...", flush=True)
                        time.sleep(delay)
                        continue
                    # Non-rate-limit error or final attempt — give up
                    messages.append({
                        "role": "user",
                        "content": f"API ERROR: {exc}. Please try again with a valid JSON action.",
                    })
                    log_step(step=step, action=f"API_ERROR: {exc}", reward=0.0, done=False, error=exc_str)
                    rewards.append(0.0)
                    steps_taken = step
                    break

            if raw_text is None:
                continue  # Skip to next step if all retries failed

            # Phase 1: Append assistant response to history
            messages.append({"role": "assistant", "content": raw_text})

            # Phase 3: Robust parsing with safe fallback
            try:
                action = parse_action(raw_text)
                action_log = action.model_dump_json()
            except Exception as exc:
                # Parse failure — No-Op fallback + corrective injection
                action = JiraCodeAction(action_type="list_files")
                action_log = f"PARSE_ERROR: {exc}"

                # Phase 4: Inject corrective message
                messages.append({
                    "role": "user",
                    "content": (
                        f"ERROR: Your last response was not valid JSON.\n"
                        f"Parse error: {exc}\n"
                        f"You MUST respond with ONLY a valid JSON object. "
                        f"No markdown, no explanations.\nTry again."
                    ),
                })

            # Take step in environment
            obs, reward, done, _ = env.step(action)
            error = obs.error

            # Ensure individual step rewards are strictly positive (min 0.01)
            reward = max(reward, 0.01)

            rewards.append(reward)
            steps_taken = step

            # Escape newlines for single-line logging
            safe_action_str = action_log.replace('\n', '\\n').replace('\r', '')
            log_step(step=step, action=safe_action_str, reward=reward, done=done, error=error)

            if done:
                break

            # Phase 1: Append observation to conversation history
            obs_message = build_observation_message(step, obs, reward)

            # Phase 4: Self-correction prompt injection on low/negative reward or error
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

        # Calculate final score (clamp strictly between 0 and 1)
        score = min(max(sum(rewards), 0.01), 0.99)
        success = score >= SUCCESS_SCORE_THRESHOLD

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
            "Default: all tasks."
        ),
    )
    args = parser.parse_args()

    import random

    # Determine which tasks to run
    if args.tasks:
        tasks = [t.strip() for t in args.tasks.split(",")]
        invalid = [t for t in tasks if t not in ALL_TASKS]
        if invalid:
            print(f"ERROR: Unknown tasks: {invalid}", flush=True)
            print(f"Available: {ALL_TASKS}", flush=True)
            return
    else:
        # Baseline inference: 1 easy, 1 medium, 1 hard randomly sampled
        easies = [t for t in ALL_TASKS if "easy" in t]
        mediums = [t for t in ALL_TASKS if "medium" in t]
        hards = [t for t in ALL_TASKS if "hard" in t]
        
        tasks = []
        if easies: tasks.append(random.choice(easies))
        if mediums: tasks.append(random.choice(mediums))
        if hards: tasks.append(random.choice(hards))

    print(f"Running tasks: {tasks}", flush=True)

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    total_score = 0.0
    results = []

    for task in tasks:
        score, steps, rewards, success = run_agent_episode(client, task)
        results.append({
            "task": task,
            "score": score,
            "steps": steps,
            "success": success,
        })
        total_score += score

        print("Waiting 20 seconds before next task to respect API limits...", flush=True)
        time.sleep(20)

    # Summary
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