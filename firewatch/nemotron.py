"""
Minimal Nemotron client (OpenAI-compatible chat completions on build.nvidia.com) and a
tool-calling loop with retries, timeouts, tool-error handling, and a JSON repair turn.
No SDK needed; uses urllib. Any OpenAI-compatible endpoint works (e.g. a NIM on Brev).
"""
import json
import re
import time
import urllib.error
import urllib.request

from . import config


class NemotronError(Exception):
    pass


class NemotronClient:
    def __init__(self, model=None, api_key=None, base_url=None, temperature=0.2, max_tokens=4096, timeout=120):
        self.model = model or config.NEMOTRON_MODEL
        self.api_key = api_key if api_key is not None else config.NVIDIA_API_KEY
        self.base_url = (base_url or config.NVIDIA_BASE_URL).rstrip("/")
        self.temperature, self.max_tokens, self.timeout = temperature, max_tokens, timeout
        if not self.api_key and "integrate.api.nvidia.com" in self.base_url:
            raise NemotronError("NVIDIA_API_KEY is not set. Get a free key at build.nvidia.com.")

    def chat(self, messages, tools=None, retries=4):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature,
                "max_tokens": self.max_tokens}
        if tools:
            body["tools"] = tools
            body["tool_choice"] = "auto"
        data = json.dumps(body).encode()
        last = None
        for attempt in range(retries):
            req = urllib.request.Request(f"{self.base_url}/chat/completions", data=data, headers={
                "Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json",
                "Accept": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    resp = json.loads(r.read().decode())
                return resp["choices"][0]["message"], resp.get("usage", {})
            except urllib.error.HTTPError as e:
                msg = e.read().decode("utf-8", "replace")[:300]
                last = f"HTTP {e.code}: {msg}"
                if e.code in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise NemotronError(last)
            except (urllib.error.URLError, TimeoutError) as e:
                last = f"{type(e).__name__}: {e}"
                time.sleep(2 ** attempt)
        raise NemotronError(f"Nemotron request failed after {retries} tries: {last}")


def extract_json(text):
    """Pull the first JSON object out of a model reply (handles ```json fences and <think> blocks)."""
    if not text:
        raise ValueError("empty reply")
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    cand = m.group(1) if m else text[text.find("{"): text.rfind("}") + 1]
    return json.loads(cand)


def run_tool_loop(client, system, user, tools, handlers, max_steps=12, trace=None, max_tool_chars=24000):
    """
    Let Nemotron call tools until it answers. Returns (final_json, messages, usage_totals).
    Tool exceptions are returned to the model as errors so it can recover (retry, pick another date).
    """
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "calls": 0}
    log = trace if trace is not None else []
    for step in range(max_steps):
        t0 = time.time()
        msg, u = client.chat(messages, tools=tools)
        usage["calls"] += 1
        usage["prompt_tokens"] += u.get("prompt_tokens", 0)
        usage["completion_tokens"] += u.get("completion_tokens", 0)
        calls = msg.get("tool_calls") or []
        am = {"role": "assistant", "content": msg.get("content") or ""}
        if calls:
            am["tool_calls"] = calls
        messages.append(am)
        log.append({"step": step, "type": "llm", "latency_s": round(time.time() - t0, 2),
                    "content": (msg.get("content") or "")[:2000],
                    "tool_calls": [{"name": c["function"]["name"], "arguments": c["function"].get("arguments")}
                                   for c in calls]})
        if not calls:
            try:
                return extract_json(msg.get("content")), messages, usage
            except (ValueError, json.JSONDecodeError) as e:
                messages.append({"role": "user", "content":
                                 f"Your reply was not valid JSON ({e}). Reply again with ONLY the JSON object."})
                log.append({"step": step, "type": "repair_request", "error": str(e)})
                continue
        for c in calls:
            name = c["function"]["name"]
            try:
                args = json.loads(c["function"].get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            t1 = time.time()
            try:
                if name not in handlers:
                    raise KeyError(f"unknown tool {name}")
                result = handlers[name](**args)
                ok = True
            except Exception as e:
                result = {"error": f"{type(e).__name__}: {e}"}
                ok = False
            text = json.dumps(result, default=str, separators=(",", ":"))
            if len(text) > max_tool_chars:   # keep it valid JSON for the model
                text = json.dumps({"truncated": True, "partial": text[:max_tool_chars]})
            messages.append({"role": "tool", "tool_call_id": c.get("id", name), "content": text})
            log.append({"step": step, "type": "tool", "name": name, "args": args, "ok": ok,
                        "latency_s": round(time.time() - t1, 2), "result_preview": text[:600]})
    raise NemotronError(f"No final answer after {max_steps} steps")
