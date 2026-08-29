import re
import sys
import random

file_path = "/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank_fixed.txt"
output_path = "/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank_fixed.txt" # Overwrite in place

# Scaled down targets for exact sum of 14,830
TARGETS = {
    "neutral": 1657,
    "curious": 828,
    "playful": 782,
    "calm": 737,
    "happy": 691,
    "sad": 691,
    "serious": 645,
    "concerned": 599,
    "affectionate": 599,
    "dramatic": 553,
    "annoyed": 553,
    "frustrated": 553,
    "surprised": 553,
    "excited": 553,
    "authoritative": 507,
    "disappointed": 507,
    "bored": 461,
    "confused": 461,
    "angry": 461,
    "soothing": 461,
    "sarcasm": 414,
    "teasing": 414,
    "deadpan": 414,
    "heartbroken": 368,
    "overjoyed": 368
}

def main():
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    re_emotion = re.compile(r'^\[(.*?)\]\s*(.*)')
    
    compound_lines = []
    single_neutral_lines = []
    
    # Track existing tags that we can preserve
    preserved = {k: 0 for k in TARGETS}
    unassigned = []
    
    # 1. Separate compound and single/neutral lines
    for line in lines:
        match = re_emotion.search(line)
        if match:
            tag_content = match.group(1).lower()
            text = match.group(2)
            
            emotions = []
            for part in tag_content.split(','):
                emotion_name = part.split('=')[0].strip()
                emotions.append(emotion_name)
                
            if len(emotions) > 1:
                compound_lines.append(line)
            else:
                emotion_name = emotions[0]
                if emotion_name in TARGETS and preserved[emotion_name] < TARGETS[emotion_name]:
                    preserved[emotion_name] += 1
                    single_neutral_lines.append((emotion_name, line, text))
                else:
                    unassigned.append(text)
        else:
            # neutral line (no tag)
            if preserved["neutral"] < TARGETS["neutral"]:
                preserved["neutral"] += 1
                single_neutral_lines.append(("neutral", line, line.strip()))
            else:
                unassigned.append(line.strip())

    # 2. Reassign unassigned text to missing target quotas
    random.shuffle(unassigned)
    new_assignments = []
    
    for emotion, target in TARGETS.items():
        deficit = target - preserved[emotion]
        for _ in range(deficit):
            text = unassigned.pop()
            new_assignments.append((emotion, text))
            
    # Combine preserved and new assignments
    final_lines = []
    for emotion, original_line, _ in single_neutral_lines:
        final_lines.append(original_line)
        
    for emotion, text in new_assignments:
        if emotion == "neutral":
            final_lines.append(text + "\n")
        else:
            intensity = round(random.uniform(0.3, 0.8), 1)
            final_lines.append(f"[{emotion}={intensity}] {text}\n")
            
    # Combine with compound lines
    all_final_lines = compound_lines + final_lines
    random.shuffle(all_final_lines) # Shuffle everything together for good measure
    
    # Write back
    with open(output_path, 'w', encoding='utf-8') as f:
        f.writelines(all_final_lines)

    print(f"Total compound lines: {len(compound_lines)}")
    print(f"Total single/neutral lines processed: {sum(TARGETS.values())}")
    print(f"Distribution complete. Total lines written: {len(all_final_lines)}")
    
if __name__ == "__main__":
    main()
