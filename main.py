import os, re, sys, traceback
from io import StringIO
from typing import List

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS: lets any website call your API (needed for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Data shapes ----------
class CodeRequest(BaseModel):
    code: str

class ErrorAnalysis(BaseModel):
    error_lines: List[int]


# ---------- PART 1: The tool (runs the code) ----------
def execute_python_code(code: str) -> dict:
    old_stdout = sys.stdout
    sys.stdout = StringIO()          # catch anything print() writes
    try:
        exec(code, {"__name__": "__main__"})   # run in a fresh, clean space
        output = sys.stdout.getvalue()
        return {"success": True, "output": output}
    except Exception:
        output = traceback.format_exc()        # the full error message
        return {"success": False, "output": output}
    finally:
        sys.stdout = old_stdout      # always put print() back to normal


# ---------- PART 2: The AI agent (finds the error line) ----------
def analyze_error_with_ai(code: str, tb: str) -> List[int]:
    numbered_code = "\n".join(
        f"{i}: {line}" for i, line in enumerate(code.split("\n"), start=1)
    )
    prompt = f"""Analyze this Python code and its error traceback.
Identify the line number(s) in the USER CODE where the error occurred.
Only use lines from the traceback that say File "<string>".
Ignore lines that mention main.py or other files.

CODE (with line numbers):
{numbered_code}

TRACEBACK:
{tb}

Return the line number(s) where the error is located."""

    resp = requests.post(
        "https://aipipe.org/openrouter/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.environ['AIPIPE_TOKEN']}",
            "Content-Type": "application/json",
        },
        json={
            "model": "google/gemini-2.0-flash-lite-001",
            "messages": [{"role": "user", "content": prompt}],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "error_analysis",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "error_lines": {
                                "type": "array",
                                "items": {"type": "integer"},
                            }
                        },
                        "required": ["error_lines"],
                        "additionalProperties": False,
                    },
                },
            },
        },
        timeout=30,
    )
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return ErrorAnalysis.model_validate_json(content).error_lines


def fallback_lines(tb: str) -> List[int]:
    """Backup plan if the AI call fails: read the line number from the traceback."""
    nums = re.findall(r'File "<string>", line (\d+)', tb)
    return [int(nums[-1])] if nums else []


# ---------- The endpoint ----------
@app.post("/code-interpreter")
def code_interpreter(req: CodeRequest):
    run = execute_python_code(req.code)

    if run["success"]:                       # no error, so no AI needed
        return {"error": [], "result": run["output"]}

    try:
        lines = analyze_error_with_ai(req.code, run["output"])
    except Exception:
        lines = fallback_lines(run["output"])

    return {"error": lines, "result": run["output"]}   # output untouched