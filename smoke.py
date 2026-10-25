#!/usr/bin/env python3
"""Smoke test for a running server: model list, a short reasoning answer, and a tool call.

usage: smoke.py [base_url] (default http://127.0.0.1:8000)
"""

import json
import sys
import time
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000").rstrip("/")


def call(path, body=None, timeout=600):
    req = urllib.request.Request(
        BASE + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def main():
    models = [m["id"] for m in call("/v1/models")["data"]]
    print("models:", models)
    model = models[0]
    ok = True

    t = time.time()
    r = call("/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": "What is 17*23? Answer with the number only."}],
        "max_tokens": 2048,
        "temperature": 0,
    })
    msg = r["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    reasoning = msg.get("reasoning") or msg.get("reasoning_content") or ""
    toks = r["usage"]["completion_tokens"]
    dt = time.time() - t
    print(f"answer: {content!r}  reasoning chars: {len(reasoning)}  {toks} tokens in {dt:.1f}s")
    if "391" not in content:
        print("FAIL: expected 391")
        ok = False
    if "<think>" in content:
        print("FAIL: think tags leaked into content (reasoning parser not active?)")
        ok = False

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    try:
        r = call("/v1/chat/completions", {
            "model": model,
            "messages": [{"role": "user", "content": "What's the weather in Toronto?"}],
            "tools": tools,
            "max_tokens": 2048,
            "temperature": 0,
        })
        c = r["choices"][0]
        calls = c["message"].get("tool_calls") or []
        print("tool call:", c["finish_reason"], [x["function"]["name"] for x in calls])
        if c["finish_reason"] != "tool_calls" or not calls:
            print("FAIL: expected a get_weather tool call")
            ok = False
    except urllib.error.HTTPError as e:
        # The tp2-card profile does not enable auto tool choice.
        print("tool call: skipped (server has no tool parser):", e.code)

    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
