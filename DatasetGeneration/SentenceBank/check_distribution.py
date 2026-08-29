import re
import sys
from collections import Counter

def check_distribution(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    total_lines = len(lines)
    distribution = Counter()
    
    single_count = 0
    compound_count = 0
    
    re_emotion = re.compile(r'^\[(.*?)\]')
    
    for line in lines:
        match = re_emotion.search(line)
        if match:
            tag_content = match.group(1).lower()
            # Extract just the emotion names, ignore intensities
            emotions = []
            for part in tag_content.split(','):
                emotion_name = part.split('=')[0].strip()
                emotions.append(emotion_name)
            
            if len(emotions) == 1:
                single_count += 1
            elif len(emotions) > 1:
                compound_count += 1
                
            emotion_key = ", ".join(emotions)
            distribution[emotion_key] += 1
        else:
            # No tag at start means neutral
            distribution["neutral"] += 1
            
    print(f"Total lines: {total_lines}")
    print(f"Neutral lines: {distribution['neutral']}")
    print(f"Single emotion tags: {single_count}")
    print(f"Compound emotion tags: {compound_count}")
    
    print("\nEmotion Distribution:")
    for emotion, count in distribution.most_common():
        print(f"{emotion}: {count} ({(count/total_lines)*100:.2f}%)")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        file_path = "/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank_fixed.txt"
    else:
        file_path = sys.argv[1]
    check_distribution(file_path)
