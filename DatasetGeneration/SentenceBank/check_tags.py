import re

file_path = "/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank_fixed.txt"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

current_tag = None
for i, line in enumerate(lines):
    if i < 14800:
        continue
    match = re.match(r"^(\[[^\]]+\])", line)
    if match:
        tag = match.group(1)
        if tag != current_tag:
            print(f"Line {i+1}: {tag}")
            current_tag = tag
