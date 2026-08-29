import re

replacements = {
    7675: "Miraculously, the inference server is accepting connections!",
    9293: "The server is acting erratically! And somehow, the service has started restarting itself, which means the next step matters.",
    11116: "The server is responding normally. But there is one mischievous thing I want to try.",
    19117: "Does this ancient device even have PCIe 5.0?",
    19120: "It is perfectly normal that you can't ping 127.0.0.1 from this subnet.",
    19123: "I am glad to see that port 5432 is open on the firewall.",
    19126: "I am devastated that I can't ping 10.0.0.1 from this subnet.",
    19131: "Are you hiding something? I can't connect, is port 3306 blocked?",
    19134: "I absolutely love that we deployed a cluster of RTX 5090 accelerators.",
    19146: "All my work is gone, I just got an HTTP 404 when I tried to log in.",
    19147: "Let's go! We just need to forward port 80 to the internal server.",
    19149: "You messed up, we expected a 200 OK, but received 502 Bad Gateway instead.",
    19150: "Fantastic news, the server is running locally at 8.8.8.8!",
    19159: "Looks like someone's popular! Traffic on port 80 is unusually high.",
    19161: "Why is the service just listening on port 80 and not doing anything?",
    19164: "Are we throwing a party? Traffic on port 443 is unusually high.",
    19167: "Unfortunately, the default gateway is just 1.1.1.1.",
    19171: "It's tragic, are you really going to buy an RTX 4080 Super for gaming?",
    19174: "That's great, the CPU is only at 44% while sitting idle.",
    19176: "Did you seriously max out the RX 7900 XTX on VRAM already?",
    19178: "Don't worry, are you really going to buy an RTX 4080 Super for gaming?",
    19180: "Guess what, the default gateway is 1.1.1.1.",
    19182: "Don't panic, but the RTX 4080 Super is completely maxed out on VRAM.",
    19183: "Wow, I am seeing CPU at 96% across all cores!",
    19186: "I am crushed... wait, why is the CPU at 54% just sitting idle?",
    19191: "Rest easy, just make sure you buy a cable that is PCIe 5.0 certified.",
    19195: "I am so thrilled we are finally upgrading our network to PCIe 5.0.",
    19200: "Sadly, I only managed to secure an RTX 5090 for the build.",
    19203: "We expected a 500 error, but received an annoying HTTP 200 instead.",
    19207: "Did you break it? Wait, why is the CPU at 87% just sitting idle?",
    19213: "I am so happy that we need to forward port 80 to the internal server!",
    19214: "Luckily, the server recovered when it crashed with the CPU at 77%.",
    19215: "Why is the service listening on port 3306?",
    19218: "Everything will be fine now that we deployed a cluster of RTX 4080 Super accelerators.",
    19219: "Don't forget to ensure that port 27017 is open on the firewall, genius.",
    19221: "That is amazing! Are you really going to buy an RTX 4090 for gaming?",
    19222: "It breaks my heart... are you really going to buy an RTX 5090 for gaming?",
    19223: "Great news! I can't ping 192.168.1.1 from this subnet.",
    19230: "I lost all my progress when the browser threw an HTTP 403 out of nowhere.",
    19231: "You must absolutely ensure that port 443 is open on the firewall!",
    19233: "Unfortunately, the service is stuck listening on port 3306.",
    19234: "Don't worry, it's normal that the RTX 5090 is completely maxed out on VRAM.",
    19237: "Please be a dear and ensure that port 27017 is open on the firewall.",
    19238: "It's okay, sometimes the browser throws an HTTP 500 out of nowhere.",
    19241: "I wanted it to fail, but I just got an HTTP 200 when I tried to log in.",
    19244: "Peekaboo! The service is listening on port 3306.",
    19246: "I am so excited! Make sure you buy a cable that is Wi-Fi 7 certified.",
    19248: "It makes me so sad, are you really going to buy an RTX 5090 for gaming?",
    19255: "Darling, make sure you buy a cable that is Wi-Fi 7 certified.",
    19257: "Oh sure, the service is just happily listening on port 80.",
    19263: "I am devastated that we only deployed a cluster of RTX 4080 Super accelerators.",
    19264: "I wonder why the service is listening on port 443.",
    19265: "It's about damn time we are finally upgrading our network to PCIe 5.0!"
}

with open("sentence_bank_cleaned.txt", "r") as f:
    lines = f.readlines()

for line_num, new_text in replacements.items():
    idx = line_num - 1
    old_line = lines[idx]
    m = re.match(r"^(\[.*?\])\s*(.*)", old_line)
    if m:
        tags = m.group(1)
        lines[idx] = f"{tags} {new_text}\n"

with open("sentence_bank_cleaned.txt", "w") as f:
    f.writelines(lines)

print("Tech conflicts resolved successfully.")
