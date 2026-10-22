"""Model backends. Gemini via the google-genai SDK, plus two offline baselines.

Baselines (no API needed):
  baseline-textbook  always answers the textbook formula: a pure 'memoriser'.
  baseline-oracle    always answers the true value: checks the grader end to end.
"""
import json
import os
import random
import re
import time


def _extract_json(text):
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.M).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return None


class Baseline:
    def __init__(self, kind):
        self.name = f"baseline-{kind}"
        self.kind = kind

    def ask(self, problem):
        if self.kind == "oracle":
            ans = problem["answer"]
        else:  # textbook
            ans = problem["naive_answer"]
            if ans is None:
                ans = "UNPREDICTABLE" if problem["level"] == 4 else None
        raw = json.dumps({"answer": ans, "confidence": 1.0, "method": self.kind})
        return raw, _extract_json(raw)


class Gemini:
    def __init__(self, model, thinking_budget=None):
        from google import genai  # pip install google-genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("Set GEMINI_API_KEY first (see README).")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.thinking_budget = thinking_budget
        self.name = model if thinking_budget is None else f"{model}@think{thinking_budget}"

    def ask(self, problem, max_retries=6):
        from google.genai import types
        cfg = {"temperature":0.0}
        if not self.model.startswith("gemma"):
            cfg["response_mime_type"]="application/json"
        if self.thinking_budget is not None:
            cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        delay = 10.0
        for attempt in range(max_retries):
            try:
                resp = self.client.models.generate_content(
                    model=self.model, contents=problem["prompt"],
                    config=types.GenerateContentConfig(**cfg))
                raw = resp.text or ""
                return raw, _extract_json(raw)
            except Exception as e:  # rate limits, transient server errors
                msg = str(e)
                if attempt == max_retries - 1:
                    return f"ERROR: {msg}", None
                wait = delay * (2 ** attempt) + random.uniform(0, 3)
                print(f"   [{self.name}] {msg[:90]}... retrying in {wait:.0f}s")
                time.sleep(wait)


def get_model(name, thinking_budget=None):
    if name.startswith("baseline-"):
        return Baseline(name.split("-", 1)[1])
    return Gemini(name, thinking_budget)
