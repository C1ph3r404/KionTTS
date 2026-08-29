import sys
import random

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

def append_compounds(libri_path, bank_path):
    # Read LibriTTS sentences
    with open(libri_path, 'r', encoding='utf-8') as f:
        libri_lines = [line.strip() for line in f if line.strip()]
        
    # Shuffle for randomness or take from the top
    # We will take from the beginning to maintain consistency, or shuffle to avoid always picking the same
    # Let's shuffle with a fixed seed so it's reproducible but random
    random.seed(42)
    random.shuffle(libri_lines)
    
    sentences_needed = len(tags) * 100
    if len(libri_lines) < sentences_needed:
        print(f"Not enough sentences in LibriTTS. Needed {sentences_needed}, found {len(libri_lines)}")
        return
        
    selected_sentences = libri_lines[:sentences_needed]
    
    new_bank_lines = []
    idx = 0
    for tag in tags:
        for _ in range(100):
            sentence = selected_sentences[idx]
            # Format: [tag1=val, tag2=val] sentence
            formatted_line = f"[{tag}] {sentence}\n"
            new_bank_lines.append(formatted_line)
            idx += 1
            
    # Append to sentence_bank.txt
    with open(bank_path, 'a', encoding='utf-8') as f:
        f.writelines(new_bank_lines)
        
    print(f"Successfully added {sentences_needed} sentences to the sentence bank.")
    print(f"Total tags: {len(tags)}. Added 100 for each.")

if __name__ == "__main__":
    append_compounds(sys.argv[1], sys.argv[2])
