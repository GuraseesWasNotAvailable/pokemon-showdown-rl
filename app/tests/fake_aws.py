#!/usr/bin/env python
"""
A stand-in for the aws CLI, for exercising cloud/teardown.sh without touching AWS.

    python tests/fake_aws.py SCENARIO.json CALLS.jsonl ec2 describe-instances --region us-east-1 ...

It appends its arguments to CALLS.jsonl (one JSON list per call), then answers from SCENARIO.json,
an object that maps "<service> <operation>" to the text the real CLI would print with
--output text, or to {"out": text, "code": exit code}. A call the scenario does not name prints
a complaint and exits 97, so a test notices a call it did not expect. Nothing here opens a
network connection.
"""
from __future__ import annotations

import json
import sys


def main(argv) -> int:
    if len(argv) < 4:
        print("usage: fake_aws.py SCENARIO.json CALLS.jsonl <aws arguments>", file=sys.stderr)
        return 2
    scenario_path, calls_path, args = argv[1], argv[2], argv[3:]
    with open(calls_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(args) + "\n")
    with open(scenario_path, "r", encoding="utf-8") as f:
        scenario = json.load(f)
    words = [a for a in args if not a.startswith("-")]
    key = " ".join(words[:2])
    if key not in scenario:
        print(f"fake_aws: the scenario has no answer for '{key}'", file=sys.stderr)
        return 97
    answer = scenario[key]
    if isinstance(answer, dict):
        text, code = str(answer.get("out", "")), int(answer.get("code", 0))
    else:
        text, code = str(answer), 0
    if text:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv))
