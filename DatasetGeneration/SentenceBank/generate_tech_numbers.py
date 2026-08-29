import random

output_file = "/home/nate/AI/Kiontts/DatasetGeneration/SentenceBank/sentence_bank_cleaned.txt"

CATEGORIES = {
    "ips": 25,
    "cpus": 25,
    "gpus": 25,
    "standards": 25,
    "http": 25,
    "ports": 25
}

EMOTIONS = [
    "neutral", "curious", "playful", "calm", "happy", "sad", "serious", 
    "affectionate", "concerned", "excited", "frustrated", "surprised", 
    "dramatic", "annoyed", "disappointed", "authoritative", "confused", 
    "soothing", "angry", "bored", "deadpan", "sarcasm", "teasing", 
    "heartbroken", "overjoyed"
]

def generate_tech_sentence(category):
    if category == "ips":
        ips = ["192.168.1.1", "10.0.0.1", "127.0.0.1", "8.8.8.8", "1.1.1.1", f"192.168.{random.randint(0,255)}.{random.randint(2,254)}"]
        ip = random.choice(ips)
        templates = [
            f"The default gateway is {ip}.",
            f"I can't ping {ip} from this subnet.",
            f"Please route all traffic through {ip}.",
            f"The server is running locally at {ip}.",
            f"We blocked access from {ip}."
        ]
        return random.choice(templates)
        
    elif category == "cpus":
        pct = random.randint(10, 100)
        models = ["i9-14900K", "Ryzen 9 7950X", "M3 Max", "Xeon W-3400"]
        model = random.choice(models)
        templates = [
            f"The server crashed with the CPU at {pct}%.",
            f"I am seeing CPU at {pct}% across all cores.",
            f"We upgraded the rig to an {model}.",
            f"The {model} handles the workload beautifully.",
            f"Wait, why is the CPU at {pct}% just sitting idle?"
        ]
        return random.choice(templates)
        
    elif category == "gpus":
        gpus = ["RTX 5090", "RTX 4090", "RX 7900 XTX", "RTX 4080 Super", "A100"]
        gpu = random.choice(gpus)
        templates = [
            f"I managed to secure an {gpu} for the build.",
            f"The {gpu} is completely maxed out on VRAM.",
            f"Are you really going to buy an {gpu} for gaming?",
            f"We deployed a cluster of {gpu} accelerators.",
            f"My frame rate doubled after installing the {gpu}."
        ]
        return random.choice(templates)
        
    elif category == "standards":
        techs = ["USB 3.2", "USB 4.0", "Wi-Fi 6", "Wi-Fi 7", "Bluetooth 5.3", "PCIe 5.0", "HDMI 2.1"]
        tech = random.choice(techs)
        templates = [
            f"The new motherboard supports {tech} natively.",
            f"Make sure you buy a cable that is {tech} certified.",
            f"We are finally upgrading our network to {tech}.",
            f"Does this device even have {tech}?",
            f"The throughput on {tech} is absolutely incredible."
        ]
        return random.choice(templates)
        
    elif category == "http":
        codes = ["HTTP 404", "HTTP 500", "HTTP 200", "HTTP 403", "502 Bad Gateway", "401 Unauthorized"]
        code = random.choice(codes)
        templates = [
            f"The API is returning an {code} error.",
            f"I just got an {code} when I tried to log in.",
            f"We expected a 200 OK, but received {code} instead.",
            f"The browser threw an {code} out of nowhere.",
            f"If you see {code}, it means the server is down."
        ]
        return random.choice(templates)
        
    elif category == "ports":
        ports = ["443", "80", "22", "8080", "3306", "5432", "27017"]
        port = random.choice(ports)
        templates = [
            f"Ensure that port {port} is open on the firewall.",
            f"The service is listening on port {port}.",
            f"I can't connect, is port {port} blocked?",
            f"We need to forward port {port} to the internal server.",
            f"Traffic on port {port} is unusually high."
        ]
        return random.choice(templates)

def main():
    generated_lines = []
    
    for category, count in CATEGORIES.items():
        for _ in range(count):
            text = generate_tech_sentence(category)
            emotion = random.choice(EMOTIONS)
            
            if emotion == "neutral":
                generated_lines.append(f"{text}\n")
            else:
                intensity = round(random.uniform(0.3, 0.8), 1)
                generated_lines.append(f"[{emotion}={intensity}] {text}\n")
                
    random.shuffle(generated_lines)
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.writelines(generated_lines)
        
    print(f"Appended {len(generated_lines)} tech-formatted numerical sentences to {output_file}.")

if __name__ == "__main__":
    main()
