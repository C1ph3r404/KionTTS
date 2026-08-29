import re

# We consider these emotions generally "inappropriate" for dry numerical facts, unless context fits.
weird_num_emotions = {"heartbroken", "affectionate", "overjoyed", "soothing", "playful", "teasing", "dramatic", "sad", "happy", "angry", "disappointed", "excited"}

# Templates from generate_numbers.py to match against
templates_regex = [
    r"We counted exactly .* items in the warehouse",
    r"The final score was .* points",
    r"There are .* reasons why this won't work",
    r"I have told you .* times already",
    r"The population is roughly .* people",
    r"The multiplier is set to",
    r"We recorded a reading of",
    r"The probability of that happening is",
    r"It was exactly .* units off",
    r"Her average score was .* over the season",
    r"Only .* of the users completed the survey",
    r"Profits are up by",
    r"I am .* sure that we locked the door",
    r"The battery is at",
    r"We have a .* chance of success",
    r"He was born on",
    r"The event took place in",
    r"We are scheduled to meet on",
    r"The deadline is",
    r"By .*, everything had changed",
    r"The train leaves at",
    r"Call me back around",
    r"Set your alarm for",
    r"We arrived precisely at",
    r"Is .* a good time for you",
    r"That will cost you",
    r"I only have .* left in my wallet",
    r"The total balance is",
    r"Can you lend me",
    r"The invoice was for",
    r"It weighs about",
    r"We drove .* before stopping",
    r"Cut a piece exactly",
    r"The capacity is",
    r"He is standing .* away",
    r"My new number is",
    r"Please call .* for assistance",
    r"Is your phone number still",
    r"You can reach him at",
    r"I dialed .* by mistake",
    r"We are upgrading to version",
    r"The error code was",
    r"Please install update",
    r"Your authorization ID is",
    r"Build .* seems stable",
    r"The temperature will be between",
    r"We expect .* to .* guests",
    r"It dropped to .* overnight",
    r"Add .* of a cup of sugar",
    r"Only .* of the items survived the trip",
    r"This is the .* time we've tried",
    r"He finished in .* place",
    r"We live on the .* floor",
    r"It is their .* anniversary",
    r"She was the .* person in line"
]

conflicts = []

with open("sentence_bank_cleaned.txt", "r") as f:
    for i, line in enumerate(f, 1):
        if line.strip() and not line.startswith("["):
            # Check if it matches any template
            if any(re.search(t, line, re.IGNORECASE) for t in templates_regex):
                conflicts.append(f"- Line {i}: [MISSING TAG] `{line.strip()}`")
            continue
            
        m = re.match(r"^\[(.*?)\]\s*(.*)", line)
        if m:
            tags_str = m.group(1)
            text = m.group(2)
            
            tags = [t.split("=")[0].strip() for t in tags_str.split(",")]
            
            if any(re.search(t, text, re.IGNORECASE) for t in templates_regex):
                has_weird_emotion = any(t in weird_num_emotions for t in tags)
                
                if has_weird_emotion:
                    conflicts.append(f"- Line {i}: `{line.strip()}`")

with open("/home/nate/.gemini/antigravity-ide/brain/b64b997c-b1ad-4035-9c5f-56ec00c975a1/num_conflicts.md", "w") as f:
    f.write("# Numerical Emotion Conflicts\n\n")
    f.write("The following lines have intense or inappropriate emotions assigned to dry, numerical statements.\n\n")
    for c in conflicts:
        f.write(c + "\n")

print(f"Found {len(conflicts)} potential numerical conflicts.")
