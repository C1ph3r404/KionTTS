import re

file_path = "/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank_fixed.txt"
with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

tags = [
    "happy=0.7, playful=0.6",
    "happy=0.7, teasing=0.6",
    "playful=0.7, teasing=0.7",
    "happy=0.7, affectionate=0.7",
    "calm=0.7, affectionate=0.6",
    "calm=0.8, soothing=0.7",
    "concerned=0.7, soothing=0.7",
    "concerned=0.7, affectionate=0.6",
    "sad=0.6, soothing=0.7",
    "sad=0.7, affectionate=0.6",
    "heartbroken=0.9, affectionate=0.6",
    "disappointed=0.7, sarcasm=0.6",
    "disappointed=0.7, serious=0.7",
    "disappointed=0.6, teasing=0.5",
    "frustrated=0.7, sarcasm=0.7",
    "frustrated=0.7, dramatic=0.8",
    "frustrated=0.6, playful=0.5",
    "angry=0.8, dramatic=0.8",
    "angry=0.7, authoritative=0.8",
    "serious=0.8, authoritative=0.8",
    "serious=0.8, dramatic=0.7",
    "excited=0.8, playful=0.7",
    "excited=0.8, teasing=0.6",
    "excited=0.8, dramatic=0.8",
    "curious=0.7, playful=0.6",
    "curious=0.7, teasing=0.6",
    "confused=0.7, surprised=0.6",
    "surprised=0.8, dramatic=0.7",
    "sarcasm=0.7, playful=0.6",
    "calm=0.8, serious=0.7",
    "calm=0.8, authoritative=0.7",
    "concerned=0.7, serious=0.7",
    "confused=0.6, curious=0.7",
    "disappointed=0.6, affectionate=0.6",
    "serious=0.7, soothing=0.7",
    "sarcasm=0.7, disappointed=0.7, calm=0.8",
    "sarcasm=0.7, frustrated=0.7, playful=0.5",
    "concerned=0.7, affectionate=0.6, soothing=0.8",
    "happy=0.7, affectionate=0.6, teasing=0.5"
]

lengths = [
    100, 99, 99, 100, 100, 99, 100, 100, 100, 100, 
    98, 100, 100, 99, 100, 99, 100, 98, 100, 100, 
    99, 99, 100, 100, 100, 99, 99, 100, 100, 100, 
    100, 99, 100, 96, 100, 100, 100, 100
]
# The remaining lines will take the 39th tag

current_line = 14830 # 0-indexed for 14831
out_lines = lines[:current_line]

for i, length in enumerate(lengths):
    for j in range(length):
        line = lines[current_line]
        # Replace the prefix
        line = re.sub(r"^\[.*?\]\s*", f"[{tags[i]}] ", line)
        out_lines.append(line)
        current_line += 1

# Process the rest of the lines with the last tag
while current_line < len(lines):
    line = lines[current_line]
    if line.strip(): # if it's not just whitespace
        line = re.sub(r"^\[.*?\]\s*", f"[{tags[-1]}] ", line)
    out_lines.append(line)
    current_line += 1

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(out_lines)

print("Done processing.")
