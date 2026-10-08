#!/usr/bin/env python3
"""
Streams /content/training.log from active Colab session into local stage1_live_stdout.log
Handles network hiccups with automatic reconnect and session auto-discovery.
"""
import os
import sys
import time
import subprocess

LOCAL_LOG = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "stage1_live_stdout.log"))


def get_active_session():
    if len(sys.argv) > 1 and sys.argv[1].strip():
        return sys.argv[1].strip()
    try:
        res = subprocess.run("colab sessions", shell=True, capture_output=True, text=True, timeout=10)
        for line in res.stdout.splitlines():
            line = line.strip()
            if line.startswith("[") and "]" in line:
                s_id = line[1:line.index("]")].strip()
                if s_id and s_id not in ("?", "colab"):
                    return s_id
    except Exception:
        pass
    return None


session = get_active_session()
print(f"[*] Starting live log streamer: Colab [{session or 'waiting...'}] -> {LOCAL_LOG}")

last_line_count = 0
last_target = None
current_session = None

while True:
    try:
        if not current_session:
            current_session = get_active_session()
            if current_session:
                print(f"[✓] Connected streamer to Colab session: {current_session}")
                with open(LOCAL_LOG, "a") as f:
                    f.write(f"\n=== [Live Stream Connected to Colab Session {current_session}] ===\n\n")
                    f.flush()
                last_line_count = 0
            else:
                time.sleep(4)
                continue

        cmd = f"""colab exec -s {current_session} --timeout 15 -f /dev/stdin << 'EOF'
import os
target = None
if os.path.exists('/content/training.log'):
    target = '/content/training.log'
elif os.path.exists('/content/stage1_live.log'):
    target = '/content/stage1_live.log'
elif os.path.exists('/content/colab_runner_outer.log'):
    target = '/content/colab_runner_outer.log'

if target:
    lines = open(target).readlines()
    print(f"__LOG_TARGET__={{target}}")
    print(f"__TOTAL_LINES__={{len(lines)}}")
    print("".join(lines[{last_line_count}:]))
EOF"""
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=20)
        stdout = res.stdout

        if "not found" in stdout or "No such session" in stdout or "session_terminated" in stdout:
            print(f"[!] Session {current_session} ended or disconnected. Re-discovering...")
            current_session = None
            time.sleep(3)
            continue

        if "__TOTAL_LINES__=" in stdout:
            parts = stdout.split("\n")
            if "__LOG_TARGET__=" in stdout:
                t_line = [p for p in parts if p.startswith("__LOG_TARGET__=")][0]
                cur_t = t_line.split("=")[1].strip()
                if cur_t != last_target:
                    last_line_count = 0
                    last_target = cur_t

            meta_line = [p for p in parts if p.startswith("__TOTAL_LINES__=")][0]
            new_total = int(meta_line.split("=")[1])
            if new_total < last_line_count:
                last_line_count = 0
            new_content = stdout[stdout.find(meta_line) + len(meta_line) + 1:]
            if new_content:
                with open(LOCAL_LOG, "a") as f:
                    f.write(new_content)
                    f.flush()
                last_line_count = new_total
    except Exception:
        pass
    time.sleep(3)
