import re
import sys

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    good_lines = []
    dropped_mismatched = 0
    dropped_dash = 0
    dropped_dialogue = 0
    dropped_header = 0

    # Whitelist for very short valid fragments (case insensitive)
    valid_short = {"wait", "really", "why", "okay", "no", "stop", "yes", "what", "with", "how", "when", "who", "where", "come here"}
    
    # Regexes
    re_emotion = re.compile(r'\[(.*?)\]')
    re_dialogue_tag = re.compile(r'(?:,"|,\s*”|\.”|\!”|\?”|,\'|,\s*)\s*(he|she|i|they|we|the man|the woman)\s+(said|asked|replied|cried|whispered|muttered|added|answered|continued|remarked)\b', re.IGNORECASE)
    re_header = re.compile(r'^(CHAPTER|PART|BOOK|THE END|TABLE OF CONTENTS|VOL\.|VOLUME|THE SEVENTEENTH REMOVE)\b', re.IGNORECASE)
    re_dateline = re.compile(r'^[A-Z\s]+,\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', re.IGNORECASE)

    for line in lines:
        original_line = line.strip()
        if not original_line:
            continue
            
        # 1. Extract and process emotion tags
        raw_text = original_line
        tags = []
        for match in re_emotion.finditer(original_line):
            tag_content = match.group(1)
            # Remove this from raw text
            raw_text = raw_text.replace(match.group(0), '')
            # Split tags by comma
            for tag in tag_content.split(','):
                tag = tag.strip()
                if not tag:
                    continue
                if '=' in tag:
                    emo, intensity = tag.split('=', 1)
                else:
                    emo, intensity = tag, '0.5'
                    
                emo = emo.strip()
                intensity = intensity.strip()
                
                # Cleanup emotion names
                # Remove -mild, -strong, dry-
                emo = re.sub(r'(-mild|-strong)$', '', emo)
                emo = re.sub(r'^dry-', '', emo)
                
                tags.append((emo, intensity))
                
        raw_text = raw_text.strip()
        
        # Deduplicate tags, keeping the first occurrence
        unique_tags = {}
        for emo, intensity in tags:
            if emo not in unique_tags:
                unique_tags[emo] = intensity
                
        # 2. Heuristics to drop bad sentences
        # Drop mismatched quotes
        quote_count = raw_text.count('"')
        if quote_count % 2 != 0:
            dropped_mismatched += 1
            continue
            
        # Drop lines ending in hyphen/dash
        if re.search(r'--?[\s"]*$', raw_text):
            dropped_dash += 1
            continue
            
        # Drop dialogue tags
        if re_dialogue_tag.search(raw_text):
            dropped_dialogue += 1
            continue
            
        # Drop headers and artifacts
        if re_header.match(raw_text) or re_dateline.match(raw_text):
            dropped_header += 1
            continue
            
        # Drop all-caps lines longer than 2 words
        words = raw_text.split()
        if len(words) > 2 and raw_text.isupper():
            dropped_header += 1
            continue

        # Format final string
        final_line = raw_text
        if unique_tags:
            tag_str = ', '.join([f"{emo}={val}" for emo, val in unique_tags.items()])
            final_line = f"[{tag_str}] {raw_text}"
            
        good_lines.append(final_line + '\n')

    print(f"Total input lines: {len(lines)}")
    print(f"Total output lines: {len(good_lines)}")
    print(f"Dropped mismatched quotes: {dropped_mismatched}")
    print(f"Dropped dashes: {dropped_dash}")
    print(f"Dropped dialogue tags: {dropped_dialogue}")
    print(f"Dropped headers/artifacts: {dropped_header}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(good_lines)

if __name__ == "__main__":
    process_file(sys.argv[1], sys.argv[2])
