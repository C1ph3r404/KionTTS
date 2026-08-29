import random
import re
import collections
import os

TARGET_LINES = 15000
MULTI_EMOTION_TARGET = 750

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
    r'^[A-Z\s0-9\.,]+$', # ALL CAPS lines
    r'\b(?:thou|thee|thy|thine|do\'st|dost|hast|hath|shalt|wilt)\b', # Archaic
    r'(?i)I mean, that\'s the situation\.',
    r'(?i)so let\'s keep going carefully\.',
    r'(?i)so don\'t change anything else yet\.',
    r'(?i)and I want to verify that before we continue\.',
    r'(?i)and that\'s where things get interesting\.',
    r'(?i)and that\'s the important part\.',
    r'(?i)so let\'s keep moving\.',
    r'(?i)which is actually good news\.',
    r'(?i)at least for now\.',
]

EMOTION_KEYWORDS = {
    'happy': ['happy', 'glad', 'joy', 'smile', 'laugh', 'delighted', 'pleased', 'good', 'great', 'awesome', 'nice', 'fixed', 'works', 'yes'],
    'sad-mild': ['sad', 'unhappy', 'tear', 'cry', 'sorry', 'sigh'],
    'sad-strong': ['grief', 'tragic', 'weep', 'sob'],
    'heartbroken': ['heartbroken', 'devastated', 'shattered', 'lost everything'],
    'frustrated-mild': ['annoying', 'bother', 'ugh', 'stuck', 'damn'],
    'frustrated-strong': ['frustrated', 'angry', 'mad', 'furious', 'rage', 'idiot', 'stupid', 'hell'],
    'curious-mild': ['wonder', 'maybe', 'perhaps', 'why', 'how', 'what'],
    'curious-strong': ['baffled', 'mystery', 'who'],
    'concerned-mild': ['worry', 'worried', 'careful', 'issue', 'problem', 'warning'],
    'concerned-strong': ['danger', 'threat', 'scared', 'fear', 'afraid', 'offline', 'crashed', 'failed', 'fatal', 'error'],
    'excited-medium': ['wow', 'amazing', 'cool', 'exciting', 'eager'],
    'excited-strong': ['incredible', 'fantastic', 'thrilled', 'omg'],
    'affectionate': ['love', 'dear', 'darling', 'sweet', 'hug', 'kiss', 'friend', 'care', 'baby'],
    'playful': ['joke', 'funny', 'fun', 'silly', 'kidding', 'tease'],
    'serious': ['must', 'required', 'critical', 'serious', 'important', 'necessary', 'report', 'status'],
    'authoritative': ['stop', 'do not', 'halt', 'listen', 'command', 'order', 'immediately', 'now'],
    'calm': ['quiet', 'peace', 'relax', 'gentle', 'soft', 'okay', 'fine'],
    'surprised-mild': ['oh', 'huh', 'really', 'sudden'],
    'surprised-strong': ['shocked', 'unbelievable', 'gasp', 'what the'],
    'disappointed-mild': ['shame', 'pity'],
    'disappointed-medium': ['disappointed', 'let down', 'failed me', 'unfortunately'],
    'disappointed-strong': ['betrayed', 'ruined'],
    'dramatic-medium': ['destiny', 'fate', 'forever', 'suddenly'],
    'overjoyed': ['best day', 'miracle', 'blessed']
}

NEGATIONS = ['not', "don't", "didn't", 'never', "can't", "won't", "isn't", "aren't", "wasn't", "weren't", 'no']
INTENSIFIERS = ['very', 'extremely', 'really', 'so', 'absolutely', 'completely', 'totally', 'too']

def is_garbage(sentence):
    for p in GARBAGE_PATTERNS:
        if re.search(p, sentence):
            return True
    if "CHAPTER" in sentence or "SECTION" in sentence:
        return True
    return False

def normalize_text(text):
    text = text.replace('suprised', 'surprised')
    text = text.replace('Suprised', 'Surprised')
    text = re.sub(r'\bmr\b', 'Mr.', text)
    text = re.sub(r'\bmrs\b', 'Mrs.', text)
    text = re.sub(r'\bpeter\b', 'Peter', text)
    text = re.sub(r'\bjames\b', 'James', text)
    text = re.sub(r'\bapril\b', 'April', text)
    text = re.sub(r'\bnovember\b', 'November', text)
    text = text.strip()
    if text and text[0].islower():
        text = text[0].upper() + text[1:]
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def classify_emotion(sentence):
    text_lower = sentence.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    
    active_negation = False
    negation_distance = 0
    scores = {e: 0 for e in EMOTION_KEYWORDS.keys()}
    
    for i, w in enumerate(words):
        if w in NEGATIONS:
            active_negation = True
            negation_distance = 3
            continue
            
        if active_negation:
            negation_distance -= 1
            if negation_distance <= 0:
                active_negation = False
                
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if w in keywords:
                points = 1.0
                if active_negation:
                    if 'happy' in emotion or 'excited' in emotion:
                        scores['disappointed-mild'] += 1.0
                    elif 'sad' in emotion or 'frustrated' in emotion:
                        scores['calm'] += 1.0
                    points = -1.0
                
                if i > 0 and words[i-1] in INTENSIFIERS:
                    points *= 1.5
                    
                scores[emotion] += max(0, points)

    if '?' in sentence:
        if scores['concerned-strong'] <= 0:
            scores['curious-mild'] += 1.0
            if '!' in sentence:
                scores['surprised-strong'] += 1.0
    if '!' in sentence:
        if sum(scores.values()) == 0:
            scores['excited-medium'] += 0.5

    best_emotion = max(scores, key=scores.get)
    max_score = scores[best_emotion]
    
    if max_score >= 1.0:
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) > 1 and (sorted_scores[0] - sorted_scores[1] < 0.5) and sorted_scores[1] > 0:
            if sorted_scores[0] < 1.5:
                return None, 0
        
        intensity = 0.5
        if max_score > 1.5:
            intensity = 0.7
        if max_score >= 2.0 or '!' in sentence:
            intensity = 0.8
        if max_score >= 3.0:
            intensity = 0.9
        if max_score == 1.0 and len(words) > 10 and '!' not in sentence:
            intensity = random.choice([0.3, 0.4, 0.5])
            
        return best_emotion, intensity
    return None, 0

def split_sentence(sentence):
    # Try to split by common punctuation that divides clauses safely
    if ', but ' in sentence:
        return sentence.split(', but ', 1)
    if '. ' in sentence:
        parts = sentence.split('. ')
        if len(parts) >= 2:
            return parts[0] + '.', ' '.join(parts[1:])
    return None

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
                sentence = re.sub(r'\[.*?\]', '', line).strip()
                if is_garbage(sentence):
                    continue
                sentence = normalize_text(sentence)
                if len(sentence) < 3: 
                    continue
                if sentence not in unique_sentences:
                    unique_sentences.add(sentence)
                    sentences_list.append(sentence)

    process_file('sentence_bank.txt')
    process_file('LibriTTS_sentence_bank.txt')
    
    print(f"Total unique clean sentences available: {len(sentences_list)}")
    random.shuffle(sentences_list)
    
    # Generate multi-emotion lines
    multi_emotion_lines = []
    available_for_multi = [s for s in sentences_list if len(s) > 40 and (', but ' in s or '. ' in s)]
    
    for s in available_for_multi:
        if len(multi_emotion_lines) >= MULTI_EMOTION_TARGET:
            break
        parts = split_sentence(s)
        if parts:
            part1, part2 = parts
            e1, i1 = classify_emotion(part1)
            e2, i2 = classify_emotion(part2)
            
            # Require at least one part to be emotional
            if e1 or e2:
                formatted_part1 = f"[{e1}={i1}] {part1}" if e1 else part1
                formatted_part2 = f"[{e2}={i2}] {part2}" if e2 else part2
                
                # if part1 was ', but ' we need to join it back correctly
                if ', but ' in s:
                    full = f"{formatted_part1}, but {formatted_part2}"
                else:
                    full = f"{formatted_part1} {formatted_part2}"
                    
                multi_emotion_lines.append(full)
                sentences_list.remove(s)
                
    print(f"Generated {len(multi_emotion_lines)} multi-emotion lines.")
    
    # Process the rest
    final_lines = []
    neutral_count = 0
    tagged_count = 0
    emotion_distribution = collections.Counter()
    
    # Prioritize technical sentences if possible, but keep it simple, just run through them
    for s in sentences_list:
        if len(final_lines) + len(multi_emotion_lines) >= TARGET_LINES:
            break
            
        e, i = classify_emotion(s)
        if e:
            final_lines.append(f"[{e}={i}] {s}")
            tagged_count += 1
            emotion_distribution[e] += 1
        else:
            final_lines.append(s)
            neutral_count += 1
            
    # Combine and shuffle
    final_dataset = final_lines + multi_emotion_lines
    random.shuffle(final_dataset)
    
    with open('final_sentence_bank.txt', 'w', encoding='utf-8') as f:
        for line in final_dataset:
            f.write(line + '\n')
            
    print(f"\n--- DATASET REPORT ---")
    print(f"Total Lines: {len(final_dataset)}")
    print(f"Multi-emotion: {len(multi_emotion_lines)}")
    print(f"Neutral: {neutral_count}")
    print(f"Tagged (Single Emotion): {tagged_count}")
    print("\nEmotion Distribution:")
    for e, c in emotion_distribution.most_common():
        bar = '█' * int(c / 50)
        print(f"{e.ljust(20)} {str(c).rjust(5)} {bar}")

if __name__ == '__main__':
    main()
