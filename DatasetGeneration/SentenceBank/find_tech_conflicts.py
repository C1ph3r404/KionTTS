import re

weird_tech_emotions = {"heartbroken", "affectionate", "overjoyed", "soothing", "playful", "teasing", "dramatic", "sad", "happy", "angry", "disappointed", "excited", "surprised"}

# Using word boundaries to prevent matching substrings like "report" for "port"
tech_regex = re.compile(r'\b(HTTP|port|VRAM|RTX|gateway|ping|subnet|CPU|motherboard|PCIe|server|Wi-Fi|IP|GPU)\b', re.IGNORECASE)

conflicts = []

with open("sentence_bank_cleaned.txt", "r") as f:
    for i, line in enumerate(f, 1):
        if line.strip() and not line.startswith("["):
            if tech_regex.search(line):
                conflicts.append(f"- Line {i}: [MISSING TAG] `{line.strip()}`")
            continue
            
        m = re.match(r"^\[(.*?)\]\s*(.*)", line)
        if m:
            tags_str = m.group(1)
            text = m.group(2)
            
            tags = [t.split("=")[0].strip() for t in tags_str.split(",")]
            
            if tech_regex.search(text):
                has_weird_emotion = any(t in weird_tech_emotions for t in tags)
                
                is_neutral_stmt = "listening on port" in text or "route all traffic" in text or "blocked access from" in text or "gateway is" in text or "completely maxed out" in text or "handles the workload" in text
                
                # Check for HTTP specific mismatches
                is_http_err = "404" in text or "500" in text or "403" in text or "502" in text or "401" in text
                is_http_succ = "200" in text
                
                conflict = False
                if has_weird_emotion:
                    conflict = True
                if is_neutral_stmt and tags[0] != "neutral":
                    conflict = True
                if is_http_err and any(t in ["happy", "overjoyed", "excited", "affectionate"] for t in tags):
                    conflict = True
                if is_http_succ and any(t in ["sad", "angry", "heartbroken", "frustrated", "annoyed", "disappointed"] for t in tags):
                    conflict = True
                    
                if conflict:
                    conflicts.append(f"- Line {i}: `{line.strip()}`")

with open("/home/nate/.gemini/antigravity-ide/brain/b64b997c-b1ad-4035-9c5f-56ec00c975a1/tech_conflicts.md", "w") as f:
    f.write("# Tech Emotion Conflicts\n\n")
    f.write("The following lines have intense or inappropriate emotions assigned to dry, technical statements (or are missing tags entirely).\n\n")
    for c in conflicts:
        f.write(c + "\n")

print(f"Found {len(conflicts)} potential tech conflicts.")
