import random
import re
import collections
import os

# Configuration
TARGET_LINES = 15000
NEUTRAL_COUNT = 5500
EMOTIONAL_COUNT = 9500

EMOTIONS = [
    'calm', 'serious', 'authoritative', 'happy', 'excited-medium', 
    'excited-strong', 'playful', 'teasing', 'curious-mild', 'curious-strong', 
    'concerned-mild', 'concerned-strong', 'frustrated-mild', 'frustrated-strong', 
    'disappointed-mild', 'disappointed-medium', 'disappointed-strong', 
    'sad-mild', 'sad-strong', 'heartbroken', 'affectionate', 'soothing', 
    'dry-sarcasm', 'sarcastic-playful', 'sarcastic-deadpan', 'sarcastic-biting', 
    'dramatic-medium', 'dramatic-strong', 'overjoyed', 'surprised-mild', 
    'surprised-strong'
]
# There are 31 emotions here.
INTENSITIES = ['0.3', '0.5', '0.7', '0.8', '0.9']

TECHNICAL_KEYWORDS = [
    'server', 'database', 'production host', 'backup', 'certificate', 
    'deployment', 'ssh', 'vpn', 'websocket', 'logs', 'error rate', 
    'configuration', 'rollback', 'build', 'service', 'gigabytes', 
    'api', 'latency', 'converged'
]

GARBAGE_PATTERNS = [
    r'(?i)all librivox recordings are in the public domain',
    r'(?i)chapter\s+[a-z0-9\.]+',
    r'(?i)section\s+[a-z0-9\.]+',
]

def is_garbage(sentence):
    for p in GARBAGE_PATTERNS:
        if re.search(p, sentence):
            return True
    
    # Roman numeral check or weird formatting like "CHAPTER two.twenty six."
    if "CHAPTER" in sentence or "SECTION" in sentence:
        return True
        
    return False

def is_technical(sentence):
    s_lower = sentence.lower()
    for kw in TECHNICAL_KEYWORDS:
        if kw in s_lower:
            return True
    return False

def normalize_text(text):
    # Fix typos
    text = text.replace('suprised', 'surprised')
    text = text.replace('Suprised', 'Surprised')
    
    # Fix old-fashioned artifacts
    text = re.sub(r'\bmr\b', 'Mr.', text)
    text = re.sub(r'\bmrs\b', 'Mrs.', text)
    text = re.sub(r'\bpeter\b', 'Peter', text)
    text = re.sub(r'\bjames\b', 'James', text)
    text = re.sub(r'\bapril\b', 'April', text)
    text = re.sub(r'\bnovember\b', 'November', text)
    
    # Ensure starts with capital letter (but skip if quotes)
    # This is basic, might need to handle leading quotes
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
        
    # Multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def main():
    print("Loading data...")
    unique_sentences = set()
    sentences_list = []
    
    def process_file(filename):
        if not os.path.exists(filename):
            return
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line: continue
                # Strip emotion tags if any
                sentence = re.sub(r'\[.*?\]', '', line).strip()
                
                if is_garbage(sentence):
                    continue
                    
                sentence = normalize_text(sentence)
                
                if len(sentence) < 3: 
                    continue # too short
                    
                if sentence not in unique_sentences:
                    unique_sentences.add(sentence)
                    sentences_list.append(sentence)

    process_file('sentence_bank.txt')
    process_file('LibriTTS_sentence_bank.txt')
    
    print(f"Total unique clean sentences available: {len(sentences_list)}")
    
    tech_sentences = [s for s in sentences_list if is_technical(s)]
    general_sentences = [s for s in sentences_list if not is_technical(s)]
    
    print(f"Technical sentences: {len(tech_sentences)}")
    print(f"General sentences: {len(general_sentences)}")
    
    pool = []
    
    tech_count = min(3500, len(tech_sentences))
    if tech_count < len(tech_sentences):
        pool.extend(random.sample(tech_sentences, tech_count))
    else:
        pool.extend(tech_sentences)
        
    needed_general = TARGET_LINES - len(pool)
    if needed_general > len(general_sentences):
        pool.extend(random.choices(general_sentences, k=needed_general))
    else:
        pool.extend(random.sample(general_sentences, needed_general))
        
    random.shuffle(pool)
    
    assert len(pool) == TARGET_LINES
    
    neutral_pool = pool[:NEUTRAL_COUNT]
    emotional_pool = pool[NEUTRAL_COUNT:]
    
    # Distribute emotions evenly
    final_lines = []
    
    # Add neutrals
    for s in neutral_pool:
        final_lines.append(s)
        
    # Add emotionals
    emotion_idx = 0
    for s in emotional_pool:
        e = EMOTIONS[emotion_idx % len(EMOTIONS)]
        i = random.choice(INTENSITIES)
        final_lines.append(f'[{e}={i}] {s}')
        emotion_idx += 1
        
    random.shuffle(final_lines)
    
    with open('final_sentence_bank.txt', 'w', encoding='utf-8') as f:
        for line in final_lines:
            f.write(line + '\n')
            
    print(f"Dataset generated! Total lines: {len(final_lines)}")
    print(f"Neutral lines: {NEUTRAL_COUNT}")
    print(f"Emotional lines: {EMOTIONAL_COUNT}")

if __name__ == '__main__':
    main()
