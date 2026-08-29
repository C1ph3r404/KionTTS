import random
import sys

output_file = "/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank_fixed.txt"

CATEGORIES = {
    "integers": 200,
    "decimals": 100,
    "percentages": 100,
    "dates": 150,
    "times": 100,
    "currency": 150,
    "measurements": 150,
    "phones": 75,
    "versions": 75,
    "ranges": 100,
    "ordinals": 50
}

EMOTIONS = [
    "neutral", "curious", "playful", "calm", "happy", "sad", "serious", 
    "affectionate", "concerned", "excited", "frustrated", "surprised", 
    "dramatic", "annoyed", "disappointed", "authoritative", "confused", 
    "soothing", "angry", "bored", "deadpan", "sarcasm", "teasing", 
    "heartbroken", "overjoyed"
]

def generate_sentence(category):
    if category == "integers":
        n = random.randint(10, 999999)
        if n > 1000:
            n = f"{n:,}"
        templates = [
            f"We counted exactly {n} items in the warehouse.",
            f"The final score was {n} points.",
            f"There are {n} reasons why this won't work.",
            f"I have told you {n} times already!",
            f"The population is roughly {n} people."
        ]
        return random.choice(templates)
        
    elif category == "decimals":
        n = round(random.uniform(0.1, 99.99), random.choice([1, 2, 3]))
        templates = [
            f"The multiplier is set to {n}.",
            f"We recorded a reading of {n} on the scale.",
            f"The probability of that happening is {n}.",
            f"It was exactly {n} units off.",
            f"Her average score was {n} over the season."
        ]
        return random.choice(templates)
        
    elif category == "percentages":
        n = round(random.uniform(0.1, 100.0), random.choice([0, 1, 2]))
        templates = [
            f"Only {n}% of the users completed the survey.",
            f"Profits are up by {n}%.",
            f"I am {n}% sure that we locked the door.",
            f"The battery is at {n}%.",
            f"We have a {n}% chance of success."
        ]
        return random.choice(templates)
        
    elif category == "dates":
        years = random.randint(1800, 2050)
        days = random.randint(1, 31)
        months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month = random.choice(months)
        templates = [
            f"He was born on {month} {days}, {years}.",
            f"The event took place in {years}.",
            f"We are scheduled to meet on {days} {month}.",
            f"The deadline is {month} {days} {years}.",
            f"By {years}, everything had changed."
        ]
        return random.choice(templates)
        
    elif category == "times":
        h = random.randint(1, 12)
        m = random.choice(["00", "15", "30", "45", str(random.randint(10, 59))])
        ampm = random.choice(["AM", "PM"])
        templates = [
            f"The train leaves at {h}:{m} {ampm}.",
            f"Call me back around {h}:{m}.",
            f"Set your alarm for {h}:{m} {ampm} tomorrow.",
            f"We arrived precisely at {h}:{m}.",
            f"Is {h}:{m} {ampm} a good time for you?"
        ]
        return random.choice(templates)
        
    elif category == "currency":
        amount = round(random.uniform(1.0, 9999.99), 2)
        if amount > 1000:
            amount = f"{amount:,.2f}"
        sym = random.choice(["$", "£", "€"])
        templates = [
            f"That will cost you {sym}{amount}.",
            f"I only have {sym}{amount} left in my wallet.",
            f"The total balance is {sym}{amount}.",
            f"Can you lend me {sym}{amount}?",
            f"The invoice was for {sym}{amount}."
        ]
        return random.choice(templates)
        
    elif category == "measurements":
        n = round(random.uniform(1.0, 500.0), random.choice([0, 1]))
        unit = random.choice(["kg", "lbs", "km", "miles", "cm", "inches", "liters", "gallons"])
        templates = [
            f"It weighs about {n} {unit}.",
            f"We drove {n} {unit} before stopping.",
            f"Cut a piece exactly {n} {unit} long.",
            f"The capacity is {n} {unit}.",
            f"He is standing {n} {unit} away."
        ]
        return random.choice(templates)
        
    elif category == "phones":
        def r(d): return ''.join(str(random.randint(0, 9)) for _ in range(d))
        n = f"{r(3)}-{r(3)}-{r(4)}"
        templates = [
            f"My new number is {n}.",
            f"Please call {n} for assistance.",
            f"Is your phone number still {n}?",
            f"You can reach him at {n}.",
            f"I dialed {n} by mistake."
        ]
        return random.choice(templates)
        
    elif category == "versions":
        n = f"{random.randint(1,5)}.{random.randint(0,9)}.{random.randint(0,20)}"
        code = f"{random.choice('ABCDEFGHIJKLMNOPQRSTUVWXYZ')}{random.randint(100,999)}"
        templates = [
            f"We are upgrading to version {n} today.",
            f"The error code was {code}.",
            f"Please install update {n}.",
            f"Your authorization ID is {code}.",
            f"Build {n} seems stable."
        ]
        return random.choice(templates)
        
    elif category == "ranges":
        n1 = random.randint(-100, 50)
        n2 = random.randint(n1 + 1, n1 + 100)
        frac = f"{random.randint(1,9)}/{random.randint(2,10)}"
        templates = [
            f"The temperature will be between {n1} and {n2} degrees.",
            f"We expect {n1} to {n2} guests.",
            f"It dropped to {n1} overnight.",
            f"Add {frac} of a cup of sugar.",
            f"Only {frac} of the items survived the trip."
        ]
        return random.choice(templates)
        
    elif category == "ordinals":
        n = random.randint(1, 99)
        suffix = "th"
        if n % 10 == 1 and n % 100 != 11: suffix = "st"
        elif n % 10 == 2 and n % 100 != 12: suffix = "nd"
        elif n % 10 == 3 and n % 100 != 13: suffix = "rd"
        templates = [
            f"This is the {n}{suffix} time we've tried.",
            f"He finished in {n}{suffix} place.",
            f"We live on the {n}{suffix} floor.",
            f"It is their {n}{suffix} anniversary.",
            f"She was the {n}{suffix} person in line."
        ]
        return random.choice(templates)
    
    return "This is a fallback sentence 42."

def main():
    generated_lines = []
    
    for category, count in CATEGORIES.items():
        for _ in range(count):
            text = generate_sentence(category)
            emotion = random.choice(EMOTIONS)
            
            if emotion == "neutral":
                generated_lines.append(f"{text}\n")
            else:
                intensity = round(random.uniform(0.3, 0.8), 1)
                generated_lines.append(f"[{emotion}={intensity}] {text}\n")
                
    # Shuffle the generated lines to distribute them well
    random.shuffle(generated_lines)
    
    # Append to the file
    with open(output_file, 'a', encoding='utf-8') as f:
        f.writelines(generated_lines)
        
    print(f"Successfully generated and appended {len(generated_lines)} numerical sentences.")

if __name__ == "__main__":
    main()
