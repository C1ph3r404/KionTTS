import json
import os

file_path = "/home/nate/AI/Kiontts/DatasetGeneration/IndexTTS2_Dataset_Generator.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        if any("def get_audio_ref(base_emotion, intensity):" in line for line in source):
            new_source = []
            skip = False
            for line in source:
                if line.startswith("EMOTIONS_SET = "):
                    new_source.append("EMOTIONS_SET = {\"angry\", \"annoyed\", \"bored\", \"concerned\", \"confused\", \"curious\", \"disappointed\", \"excited\", \"frustrated\", \"happy\", \"heartbroken\", \"overjoyed\", \"sad\", \"surprised\"}\n")
                    continue
                    
                if line.startswith("def get_audio_ref(base_emotion, intensity):"):
                    skip = True
                    new_source.append(line)
                    new_source.append("    mapping = {\n")
                    new_source.append("        \"angry\": [(0.6, \"angry-mild\"), (1.1, \"angry-strong\")],\n")
                    new_source.append("        \"annoyed\": [(0.6, \"annoyed-mild\"), (1.1, \"annoyed-strong\")],\n")
                    new_source.append("        \"concerned\": [(0.6, \"concerned-mild\"), (1.1, \"concerned-strong\")],\n")
                    new_source.append("        \"confused\": [(0.6, \"confused-mild\"), (1.1, \"confused-strong\")],\n")
                    new_source.append("        \"curious\": [(0.6, \"curious-mild\"), (1.1, \"curious-strong\")],\n")
                    new_source.append("        \"disappointed\": [(0.5, \"disappointed-mild\"), (0.7, \"disappointed-medium\"), (1.1, \"disappointed-strong\")],\n")
                    new_source.append("        \"dramatic\": [(0.6, \"dramatic-medium\"), (1.1, \"dramatic-strong\")],\n")
                    new_source.append("        \"excited\": [(0.6, \"excited-medium\"), (1.1, \"excited-strong\")],\n")
                    new_source.append("        \"frustrated\": [(0.6, \"frustrated-mild\"), (1.1, \"frustrated-strong\")],\n")
                    new_source.append("        \"happy\": [(0.6, \"happy-mild\"), (1.1, \"happy-strong\")],\n")
                    new_source.append("        \"sad\": [(0.6, \"sad-mild\"), (1.1, \"sad-strong\")],\n")
                    new_source.append("        \"suprised\": [(0.6, \"surprised-mild\"), (1.1, \"surprised-strong\")],\n")
                    new_source.append("        \"surprised\": [(0.6, \"surprised-mild\"), (1.1, \"surprised-strong\")],\n")
                    new_source.append("        \"sarcasm\": [(1.1, \"dry-sarcasm\")]\n")
                    new_source.append("    }\n")
                    new_source.append("    if base_emotion in mapping:\n")
                    new_source.append("        for threshold, name in mapping[base_emotion]:\n")
                    new_source.append("            if intensity < threshold:\n")
                    new_source.append("                return f\"voice_preview_{name}.wav\"\n")
                    new_source.append("    return f\"voice_preview_{base_emotion}.wav\"\n")
                    new_source.append("\n")
                elif line.startswith("def get_compound_audio_ref(styles_dict):"):
                    skip = True
                    new_source.append(line)
                    new_source.append("    try:\n")
                    new_source.append("        available_refs = os.listdir(EMO_REFS_DIR)\n")
                    new_source.append("    except:\n")
                    new_source.append("        return None\n")
                    new_source.append("        \n")
                    new_source.append("    expected_parts = []\n")
                    new_source.append("    for k, v in styles_dict.items():\n")
                    new_source.append("        norm_k = 'heartbroken' if k == 'heartbreak' else 'surprised' if k == 'suprised' else 'sarcasm' if k == 'sarcastic' else k\n")
                    new_source.append("        val_str = str(v).replace('.', '-')\n")
                    new_source.append("        expected_parts.append(f\"{norm_k}{val_str}\")\n")
                    new_source.append("        \n")
                    new_source.append("    import itertools\n")
                    new_source.append("    for file in available_refs:\n")
                    new_source.append("        if file.startswith(\"voice_preview_\") and file.endswith(\".wav\"):\n")
                    new_source.append("            basename = file.replace(\"voice_preview_\", \"\").replace(\".wav\", \"\")\n")
                    new_source.append("            for perm in itertools.permutations(expected_parts):\n")
                    new_source.append("                if basename == \"-\".join(perm):\n")
                    new_source.append("                    return file\n")
                    new_source.append("    return None\n")
                    new_source.append("\n")
                elif line.startswith("def generate_segment("):
                    skip = False
                    
                if not skip:
                    new_source.append(line)
                    
            cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully.")
