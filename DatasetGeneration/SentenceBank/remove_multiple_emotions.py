import re
import sys

def remove_multiple_emotions(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    new_lines = []
    removed_count = 0
    re_emotion = re.compile(r'^\[(.*?)\]')
    
    for line in lines:
        match = re_emotion.search(line)
        if match:
            tag_content = match.group(1)
            # If there's a comma in the tag content, it has multiple emotions
            if ',' in tag_content:
                removed_count += 1
                continue
                
        new_lines.append(line)
        
    with open(file_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)
        
    print(f"Total original lines: {len(lines)}")
    print(f"Lines removed (multiple emotions): {removed_count}")
    print(f"Total new lines: {len(new_lines)}")

if __name__ == "__main__":
    remove_multiple_emotions(sys.argv[1])
