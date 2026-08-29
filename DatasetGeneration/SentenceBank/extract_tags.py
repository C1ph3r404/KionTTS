import re
from collections import defaultdict

pattern = r'\[(.*?)\]'
grouped_tags = defaultdict(set)

file_path = '/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank.txt'

try:
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            matches = re.findall(pattern, line)
            for tag_str in matches:
                pairs = tag_str.split(',')
                num_tags = len(pairs)
                
                # Strip out the intensities
                base_emotions = []
                for pair in pairs:
                    if '=' in pair:
                        base_emotions.append(pair.split('=')[0].strip())
                    elif ':' in pair:
                        base_emotions.append(pair.split(':')[0].strip())
                    else:
                        base_emotions.append(pair.strip())
                        
                clean_tag_str = ', '.join(base_emotions)
                
                if num_tags == 1:
                    grouped_tags['Single'].add(clean_tag_str)
                elif num_tags == 2:
                    grouped_tags['Double'].add(clean_tag_str)
                elif num_tags == 3:
                    grouped_tags['Triple'].add(clean_tag_str)
                else:
                    grouped_tags[f'{num_tags}-Tuple'].add(clean_tag_str)
                    
    print(f"Total unique combinations (without intensities): {sum(len(v) for v in grouped_tags.values())}\n")
    
    for category in ['Single', 'Double', 'Triple']:
        if category in grouped_tags:
            print(f"=== {category} Emotion Tags ({len(grouped_tags[category])}) ===")
            for tag in sorted(grouped_tags[category]):
                print(f" - [{tag}]")
            print()
            
    for category, tags in grouped_tags.items():
        if category not in ['Single', 'Double', 'Triple']:
            print(f"=== {category} Emotion Tags ({len(tags)}) ===")
            for tag in sorted(tags):
                print(f" - [{tag}]")
            print()

except Exception as e:
    print(f"Error: {e}")
