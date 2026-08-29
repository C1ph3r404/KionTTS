import re
import sys
from collections import defaultdict

def check_duplicates(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    sentence_counts = defaultdict(int)
    re_emotion = re.compile(r'^\[.*?\]\s*')
    
    for line in lines:
        if not line.strip():
            continue
            
        # Strip the emotion tag to get just the sentence text
        raw_sentence = re_emotion.sub('', line).strip()
        
        if raw_sentence:
            # We can lower() it to catch case-insensitive duplicates, but let's do exact match first
            sentence_counts[raw_sentence] += 1
            
    duplicates = {sentence: count for sentence, count in sentence_counts.items() if count > 1}
    
    print(f"Total lines checked: {len(lines)}")
    print(f"Total unique sentences: {len(sentence_counts)}")
    print(f"Number of duplicate sentence strings found: {len(duplicates)}")
    
    # Calculate how many extra lines are caused by duplicates
    extra_lines = sum(count - 1 for sentence, count in duplicates.items())
    print(f"Total redundant lines due to duplication: {extra_lines}")
    
    # Print a few examples
    if duplicates:
        print("\nExamples of duplicates:")
        sorted_dupes = sorted(duplicates.items(), key=lambda x: x[1], reverse=True)
        for s, c in sorted_dupes[:10]:
            print(f"[{c}x] {s}")

if __name__ == "__main__":
    check_duplicates(sys.argv[1])
