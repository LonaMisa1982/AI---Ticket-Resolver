# server/env.py
import os
import re
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import sys

from openenv.core.env_server import Environment, State
from src.jira_to_code.models import JiraCodeAction, JiraCodeObservation

# ---------------------------------------------------------------------------
# Knowledge Base — ChromaDB (graceful degradation if not installed)
# ---------------------------------------------------------------------------
try:
    import chromadb
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False

try:
    from openai import OpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# ✅ CHANGE 2: KB now lives in ./corporate_memory (persistent across runs)
KB_DIR = "./corporate_memory"
KB_COLLECTION = "jira_solutions"

# LLM summary generation uses the same router configured via env vars
_KB_LLM_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
_KB_LLM_MODEL = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-7B-Instruct"
_KB_LLM_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "dummy"

_KB_SUMMARY_PROMPT = (
    "You are a senior software engineer. Given a Jira ticket and the code that fixed it, "
    "write a concise (2-4 sentences) summary of:\n"
    "1. The root cause of the bug / what was missing.\n"
    "2. How the fix addresses it.\n"
    "Respond with ONLY the summary — no headers, no bullet points, no extra text."
)


class JiraToCodeEnv(Environment):
    TASKS = {
        "easy": {
            "dir": "src/jira_to_code/tasks/easy",
            "ticket": (
                "TICKET-101: Fix the off-by-one bug in calculator.add() function. "
                "It should correctly sum two numbers."
            ),
        },
        "easy_2": {
            "dir": "src/jira_to_code/tasks/easy_2",
            "ticket": (
                "TICKET-102: Fix the bug in string_utils.count_vowels(). "
                "It currently only counts lowercase vowels but should be case-insensitive."
            ),
        },
        "easy_3": {
            "dir": "src/jira_to_code/tasks/easy_3",
            "ticket": (
                "TICKET-E3: The API endpoint crashes with a KeyError when a user payload "
                "doesn't contain an optional 'phone_number' field. Change dictionary "
                "indexing to .get() with a fallback."
            ),
        },
        "easy_4": {
            "dir": "src/jira_to_code/tasks/easy_4",
            "ticket": (
                "TICKET-E4: Off-by-One Pagination. get_page_bounds(page, size) misses the "
                "10th item on page 1. Fix the math index logic."
            ),
        },
        "easy_5": {
            "dir": "src/jira_to_code/tasks/easy_5",
            "ticket": (
                "TICKET-E5: FastAPI Route Typo. Route signature is id instead of user_id. "
                "Fix the parameter mismatch."
            ),
        },
        "medium": {
            "dir": "src/jira_to_code/tasks/medium",
            "ticket": (
                "TICKET-201: Implement format_user_data in formatter.py. "
                "It should format dictionary data to 'LAST_NAME, First_name (Age: X)'. "
                "Handle missing age by defaulting to 'Unknown'."
            ),
        },
        "medium_2": {
            "dir": "src/jira_to_code/tasks/medium_2",
            "ticket": (
                "TICKET-202: Implement validate_email() and validate_password() in validator.py. "
                "Email: must have exactly one '@', at least 1 char before '@', a '.' after '@' "
                "with chars around it. "
                "Password: at least 8 chars, one uppercase, one lowercase, one digit."
            ),
        },
        "medium_3": {
            "dir": "src/jira_to_code/tasks/medium_3",
            "ticket": (
                "TICKET-M3: Missing Authentication Middleware. A sensitive endpoint "
                "(/api/billing) is exposed. Import @require_auth from auth.py and apply it "
                "to the route in routes.py."
            ),
        },
        "medium_4": {
            "dir": "src/jira_to_code/tasks/medium_4",
            "ticket": (
                "TICKET-M4: N+1 Database Problem. Rewrite the ORM query to use a JOIN "
                "(e.g., select_related)."
            ),
        },
        "medium_5": {
            "dir": "src/jira_to_code/tasks/medium_5",
            "ticket": (
                "TICKET-M5: Flawed Regex Validation. validate_email rejects emails with a "
                "plus sign. Update regex to allow user+test@gmail.com."
            ),
        },
        "medium_6": {
            "dir": "src/jira_to_code/tasks/medium_6",
            "ticket": (
                "TICKET-M6: Incomplete Error Handling. fetching currency rates crashes on "
                "timeout. Wrap in try/except and return a cached fallback value."
            ),
        },
        "medium_7": {
            "dir": "src/jira_to_code/tasks/medium_7",
            "ticket": (
                "TICKET-M7: Stale Cache Bug. update_user_profile updates DB but forgets to "
                "call redis.delete('user:id'). Invalidate the cache."
            ),
        },
        "medium_8": {
            "dir": "src/jira_to_code/tasks/medium_8",
            "ticket": (
                "TICKET-M8: Timezone Naive Conversion. Event scheduling function creates "
                "naive datetimes. Make them UTC aware."
            ),
        },
        "medium_9": {
            "dir": "src/jira_to_code/tasks/medium_9",
            "ticket": (
                "TICKET-M9: State Machine Loophole. Cart state machine allows CANCELLED to "
                "SHIPPED. Add transition guards."
            ),
        },
        "medium_10": {
            "dir": "src/jira_to_code/tasks/medium_10",
            "ticket": (
                "TICKET-M10: Config Merge Overwrite. YAML merge completely overwrites nested "
                "dictionaries. Fix recursion logic."
            ),
        },
        "hard": {
            "dir": "src/jira_to_code/tasks/hard",
            "ticket": (
                "TICKET-301: Implement an LRUCache class in lru_cache.py with put() and get() "
                "methods. O(1) time complexity expected. Evict least recently used when "
                "capacity is reached."
            ),
        },
        "hard_2": {
            "dir": "src/jira_to_code/tasks/hard_2",
            "ticket": (
                "TICKET-302: Implement a DirectedGraph class in graph.py with add_edge(), "
                "has_path() (BFS/DFS), and topological_sort() methods. "
                "topological_sort() must return an empty list if a cycle is detected."
            ),
        },
        "hard_3": {
            "dir": "src/jira_to_code/tasks/hard_3",
            "ticket": (
                "TICKET-H3: Circular Dependency Resolution. models.py, utils.py, config.py. "
                "Extract shared logic into base.py."
            ),
        },
        "hard_4": {
            "dir": "src/jira_to_code/tasks/hard_4",
            "ticket": (
                "TICKET-H4: Race Condition in Thread Worker. Refactor the architecture to "
                "use queue.Queue."
            ),
        },
        "hard_5": {
            "dir": "src/jira_to_code/tasks/hard_5",
            "ticket": (
                "TICKET-H5: OOM Generator Fix. Readlines causes crash on 5GB file. "
                "Rewrite to yield generators."
            ),
        },
        "hard_6": {
            "dir": "src/jira_to_code/tasks/hard_6",
            "ticket": (
                "TICKET-H6: Implement Abstract Base Class. Implement StripeGateway matching "
                "PaymentGateway abstract class."
            ),
        },
        "hard_7": {
            "dir": "src/jira_to_code/tasks/hard_7",
            "ticket": (
                "TICKET-H7: Deadlock in Asyncio. Route acquires threading.Lock but forgets "
                "to release on exception. Use async context managers."
            ),
        },
    }

    STEP_PENALTY = -0.01
    GRACE_STEPS = 3

    def __init__(self):
        super().__init__()
        self.step_count = 0
        self.workspace_dir = None
        self.task_level = "easy"
        self.task_source_dir = None
        self.jira_ticket = ""

        # ✅ CHANGE 2: Persistent ChromaDB client initialized in __init__
        # The client is shared across the lifetime of the environment instance.
        self._chroma_client: Optional[Any] = None
        self._kb_collection: Optional[Any] = None
        if _CHROMA_AVAILABLE:
            try:
                Path(KB_DIR).mkdir(parents=True, exist_ok=True)
                self._chroma_client = chromadb.PersistentClient(path=KB_DIR)
                self._kb_collection = self._chroma_client.get_or_create_collection(KB_COLLECTION)
                print(f"[KB] Persistent ChromaDB initialized at '{KB_DIR}'", flush=True)
            except Exception as exc:
                print(f"[KB] Could not initialize ChromaDB: {exc}", flush=True)

        # ✅ CHANGE 4: HITL state — tracks whether the agent is waiting for human approval
        # Populated by the "request_human_review" action; consumed by streamlit_app.py
        self.pending_review: Optional[Dict[str, Any]] = None  # {file_path, original, proposed, diff}

        # Snapshot of original source files (populated at reset) and fixed files (populated on write)
        self.original_files: Dict[str, str] = {}   # relative_path -> original content
        self.fixed_files: Dict[str, str] = {}      # relative_path -> fixed content

    # ------------------------------------------------------------------
    # Knowledge Base helpers
    # ------------------------------------------------------------------

    def _get_kb_collection(self) -> Optional[Any]:
        """Return the already-initialized ChromaDB collection (no new client per call)."""
        return self._kb_collection

    def _generate_kb_summary(self, ticket: str, code: str) -> str:
        """Call the LLM to produce a brief root-cause + fix summary."""
        if not _OPENAI_AVAILABLE:
            return "No summary available (openai package not installed)."
        try:
            llm = OpenAI(base_url=_KB_LLM_BASE_URL, api_key=_KB_LLM_KEY)
            resp = llm.chat.completions.create(
                model=_KB_LLM_MODEL,
                messages=[
                    {"role": "system", "content": _KB_SUMMARY_PROMPT},
                    {
                        "role": "user",
                        "content": (
                            f"Jira Ticket:\n{ticket}\n\n"
                            f"Fixed Code:\n```python\n{code}\n```"
                        ),
                    },
                ],
                temperature=0.2,
                max_tokens=256,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            print(f"[KB] Summary generation failed: {exc}", flush=True)
            return f"Summary generation failed: {exc}"

    def _capture_knowledge(self, ticket: str, code: str) -> None:
        """
        ✅ CHANGE 5: Persist a successful, human-approved solution to ChromaDB.
        Called from submit logic only after HITL approval (or auto-approve if HITL disabled).

        - document  : the fixed code snippet (used for similarity search)
        - metadata  : ticket_id, ticket text, LLM-generated summary
        - id        : deterministic hash so duplicate runs don't double-insert
        """
        collection = self._get_kb_collection()
        if collection is None:
            print("[KB] ChromaDB not available — skipping knowledge capture.", flush=True)
            return

        try:
            doc_id = f"solution_{self.task_level}"
            # ✅ CHANGE 5: Teacher LLM generates summary before vectorizing
            summary = self._generate_kb_summary(ticket, code)

            ticket_id_match = re.search(r'TICKET-[\w]+', ticket)
            ticket_id = ticket_id_match.group(0) if ticket_id_match else self.task_level

            collection.upsert(
                ids=[doc_id],
                documents=[code],
                metadatas=[
                    {
                        "ticket_id": ticket_id,
                        "ticket_text": ticket,
                        "summary": summary,
                        "task_level": self.task_level,
                    }
                ],
            )
            print(
                f"[KB] Captured solution for '{ticket_id}' (id={doc_id}). "
                f"Summary: {summary[:80]}…",
                flush=True,
            )
        except Exception as exc:
            print(f"[KB] Failed to upsert into ChromaDB: {exc}", flush=True)

    # ------------------------------------------------------------------
    # Standard Environment helpers
    # ------------------------------------------------------------------

    def _get_file_tree(self) -> list[str]:
        if not self.workspace_dir:
            return []
        tree = []
        for root, _, files in os.walk(self.workspace_dir):
            for file in files:
                if "__pycache__" in root or file.endswith(".pyc"):
                    continue
                rel_path = Path(root) / file
                tree.append(str(rel_path.relative_to(self.workspace_dir)))
        return tree

    @staticmethod
    def _parse_pytest_results(output: str) -> tuple[int, int]:
        """Extract (passed, total) from pytest output for partial-credit scoring."""
        match_passed = re.search(r'(\d+) passed', output)
        passed = int(match_passed.group(1)) if match_passed else 0
        match_failed = re.search(r'(\d+) failed', output)
        failed = int(match_failed.group(1)) if match_failed else 0
        match_error = re.search(r'(\d+) error', output)
        errors = int(match_error.group(1)) if match_error else 0
        total = passed + failed + errors
        return passed, max(total, 1)

    def _compute_diff(self, original: str, proposed: str, file_path: str) -> str:
        """Compute a unified diff between original and proposed content."""
        import difflib
        original_lines = original.splitlines(keepends=True)
        proposed_lines = proposed.splitlines(keepends=True)
        diff = difflib.unified_diff(
            original_lines,
            proposed_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm="",
        )
        return "".join(diff) or "(no changes detected)"

    def reset(self) -> JiraCodeObservation:
        self.step_count = 0
        self.pending_review = None  # ✅ CHANGE 4: clear any pending HITL review on reset
        if self.workspace_dir and Path(self.workspace_dir).exists():
            shutil.rmtree(self.workspace_dir)

        self.task_level = os.getenv("JIRA_TASK_LEVEL", "medium").lower()
        if self.task_level not in self.TASKS:
            self.task_level = "easy"

        self.task_source_dir = Path(self.TASKS[self.task_level]["dir"]).resolve()
        self.jira_ticket = self.TASKS[self.task_level]["ticket"]

        self.workspace_dir = tempfile.mkdtemp(prefix=f"jira_env_{self.task_level}_")

        if self.task_source_dir.exists():
            shutil.copytree(self.task_source_dir, self.workspace_dir, dirs_exist_ok=True)
        else:
            print(f"Warning: Task directory {self.task_source_dir} not found!")

        # Snapshot every non-test Python file so we can show before/after later
        self.original_files = {}
        self.fixed_files = {}
        for py_file in Path(self.workspace_dir).rglob("*.py"):
            rel = str(py_file.relative_to(self.workspace_dir))
            if "test_" not in rel and "__pycache__" not in rel:
                try:
                    self.original_files[rel] = py_file.read_text()
                except Exception:
                    pass

        return JiraCodeObservation(
            jira_ticket=self.jira_ticket,
            file_tree=self._get_file_tree(),
        )

    def step(
        self, action: JiraCodeAction
    ) -> Tuple[JiraCodeObservation, float, bool, Dict[str, Any]]:
        self.step_count += 1
        reward = 0.0
        done = False
        current_file_content = None
        test_output = None
        error = None
        info: Dict[str, Any] = {}

        workspace_path = Path(self.workspace_dir).resolve()

        try:
            if action.action_type == "list_files":
                current_file_content = "\n".join(self._get_file_tree())

            elif action.action_type in ["read_file", "write_file"]:
                if not action.file_path:
                    error = "file_path must be provided for read/write actions."
                else:
                    target_path = (workspace_path / action.file_path).resolve()
                    if not target_path.is_relative_to(workspace_path):
                        error = "Access denied: cannot access files outside workspace."
                    elif action.action_type == "read_file":
                        if target_path.exists():
                            current_file_content = target_path.read_text()
                        else:
                            error = f"File not found: {action.file_path}"
                    elif action.action_type == "write_file":
                        if action.content is None:
                            error = "content must be provided for write_file action."
                        else:
                            target_path.parent.mkdir(parents=True, exist_ok=True)
                            target_path.write_text(action.content)
                            current_file_content = action.content
                            # Track this as a fix for before/after display
                            self.fixed_files[action.file_path] = action.content
                            reward = 0.05

            # ✅ RESTRUCTURED: HITL approval is now POST-SUBMIT ONLY (removed mid-episode request_human_review)

            elif action.action_type == "run_tests":
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-v"],
                    cwd=self.workspace_dir,
                    capture_output=True, text=True, timeout=30,
                )
                test_output = result.stdout + "\n" + result.stderr
                passed, total = self._parse_pytest_results(test_output)

                if result.returncode == 0:
                    reward = 0.1 + 0.4 * (passed / total)
                elif result.returncode == 1:
                    reward = 0.1 * (passed / total)
                else:
                    reward = -0.1

            elif action.action_type == "submit":
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", "-v"],
                    cwd=self.workspace_dir,
                    capture_output=True, text=True, timeout=30,
                )
                test_output = result.stdout + "\n" + result.stderr
                passed, total = self._parse_pytest_results(test_output)

                if result.returncode == 0:
                    # ✅ RESTRUCTURED: All tests passed — pause for final HITL approval
                    # Emit before/after summary for every file the agent changed
                    import json as _json
                    for file_path, fixed_content in self.fixed_files.items():
                        original_content = self.original_files.get(file_path, "")
                        summary_payload = _json.dumps({
                            "file_path": file_path,
                            "original": original_content,
                            "fixed": fixed_content,
                            "tests_passed": True,
                            "passed": passed,
                            "total": total,
                        }, separators=(",", ":"))
                        print(f"[FIX_SUMMARY] {summary_payload}", flush=True)

                    # Create pending review for HITL approval
                    if self.fixed_files:
                        # Build a combined diff of all changes
                        combined_diff_parts = []
                        for file_path, fixed_content in self.fixed_files.items():
                            original_content = self.original_files.get(file_path, "")
                            diff_text = self._compute_diff(original_content, fixed_content, file_path)
                            combined_diff_parts.append(diff_text)
                        combined_diff = "\n".join(combined_diff_parts)
                    else:
                        combined_diff = "(no changes detected)"

                    self.pending_review = {
                        "test_results": {"passed": passed, "total": total},
                        "diff": combined_diff,
                        "fixed_files": dict(self.fixed_files),
                        "test_output": test_output,
                    }
                    reward = 0.9
                    info["awaiting_human_review"] = True
                    done = False
                    print(
                        f"[HITL] All tests passed ({passed}/{total}). "
                        f"Pausing for final human approval before persisting changes.",
                        flush=True,
                    )
                else:
                    # Tests failed — end episode immediately without HITL
                    done = True
                    reward = 0.5 * (passed / total)
                    print(
                        f"[HITL] Submit failed: {passed}/{total} tests passed. "
                        f"Episode ended without HITL review.",
                        flush=True,
                    )

            # ✅ RESTRUCTURED: New action — human_approved_final
            elif action.action_type == "human_approved_final":
                if self.pending_review is None:
                    error = "No pending review to approve."
                else:
                    # Write all approved files back to the SOURCE task directory
                    fixed_files_to_persist = self.pending_review.get("fixed_files", {})
                    for file_path, approved_content in fixed_files_to_persist.items():
                        if self.task_source_dir:
                            source_target = (self.task_source_dir / file_path).resolve()
                            try:
                                source_target.relative_to(self.task_source_dir)
                                source_target.parent.mkdir(parents=True, exist_ok=True)
                                source_target.write_text(approved_content)
                                print(
                                    f"[HITL] Fix approved and saved: '{source_target}'",
                                    flush=True,
                                )
                            except ValueError:
                                print(
                                    f"[HITL] Warning: could not resolve source path for '{file_path}'",
                                    flush=True,
                                )

                    # ✅ CHANGE 5: Capture knowledge now that human has approved
                    submitted_code = action.content or ""
                    if not submitted_code and self.workspace_dir:
                        for py_file in Path(self.workspace_dir).rglob("*.py"):
                            rel = str(py_file.relative_to(self.workspace_dir))
                            if "test_" not in rel and not rel.startswith("__"):
                                try:
                                    submitted_code = py_file.read_text()
                                    break
                                except Exception:
                                    pass
                    self._capture_knowledge(self.jira_ticket, submitted_code)

                    self.pending_review = None
                    reward = 1.0
                    done = True
                    info["human_approved_final"] = True
                    print(f"[HITL] Final approval granted. Episode complete.", flush=True)

            # ✅ RESTRUCTURED: New action — human_rejected_final
            elif action.action_type == "human_rejected_final":
                if self.pending_review is None:
                    error = "No pending review to reject."
                else:
                    self.pending_review = None
                    error = "Human rejected the final solution. Tests passed but changes were not approved."
                    reward = 0.01
                    done = True
                    info["human_rejected_final"] = True
                    print(f"[HITL] Final approval denied. Changes discarded.", flush=True)

        except subprocess.TimeoutExpired:
            error = "Tests timed out after 30 seconds."
            test_output = "TIMEOUT"
            reward = -0.1
        except Exception as e:
            error = f"System error: {str(e)}"
            reward = -0.2

        # Shaping rewards
        if self.step_count <= 3:
            reward += 0.02
        else:
            reward -= 0.01

        # Strictly bounded rewards for OpenEnv (0.01 – 0.99)
        reward = max(0.01, min(0.99, reward))

        obs = JiraCodeObservation(
            jira_ticket=self.jira_ticket,
            file_tree=self._get_file_tree(),
            current_file_content=current_file_content,
            test_output=test_output,
            error=error,
        )
        return obs, reward, done, info

    def state(self) -> State:
        return State(
            episode_id=f"jira-{self.task_level}-{self.step_count}",
            step_count=self.step_count,
        )

    def close(self):
        if self.workspace_dir and Path(self.workspace_dir).exists():
            shutil.rmtree(self.workspace_dir)