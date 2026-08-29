import re

pos_emotions = {"happy", "overjoyed", "playful", "affectionate", "excited"}
neg_emotions = {"sad", "disappointed", "angry", "heartbroken", "frustrated", "annoyed"}

pos_words = ["smile", "laugh", "pleased", "happy", "joy", "delight", "wonderful", "great", "glad", "merry", "cheerful", "beautiful", "excellent", "love", "kiss", "haha"]
neg_words = ["sad", "cry", "weep", "tear", "pain", "hurt", "tragedy", "death", "died", "kill", "murder", "miserable", "sorrow", "angry", "furious", "hate", "terrible", "awful", "horrible", "blood", "die", "grief"]

pos_regex = re.compile(r'\b(' + '|'.join(pos_words) + r')s?\b', re.IGNORECASE)
neg_regex = re.compile(r'\b(' + '|'.join(neg_words) + r')s?\b', re.IGNORECASE)

conflicts = []

with open("sentence_bank_cleaned.txt", "r") as f:
    for i, line in enumerate(f, 1):
        m = re.match(r"^\[(.*?)\]\s*(.*)", line)
        if m:
            tags_str = m.group(1)
            text = m.group(2)
            
            tags = []
            for tag_val in tags_str.split(","):
                tag = tag_val.split("=")[0].strip()
                tags.append(tag)
            
            is_pos = any(t in pos_emotions for t in tags)
            is_neg = any(t in neg_emotions for t in tags)
            
            if is_neg and not is_pos:
                if pos_regex.search(text) and not neg_regex.search(text):
                    conflicts.append(f"- Line {i}: `{line.strip()}`")
            elif is_pos and not is_neg:
                if neg_regex.search(text) and not pos_regex.search(text):
                    conflicts.append(f"- Line {i}: `{line.strip()}`")

with open("/home/nate/.gemini/antigravity-ide/brain/b64b997c-b1ad-4035-9c5f-56ec00c975a1/hard_conflicts.md", "w") as f:
    f.write("# Hard Emotion Conflicts\n\n")
    f.write("The following lines have a strong mismatch between the labeled emotion and the sentence content.\n\n")
    for c in conflicts:
        f.write(c + "\n")

print(f"Found {len(conflicts)} potential conflicts.")
