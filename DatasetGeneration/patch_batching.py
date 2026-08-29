import json
import copy

file_path = "/home/nate/AI/Kiontts/DatasetGeneration/IndexTTS2_Dataset_Generator.ipynb"

with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

new_cells = []
drive_mounted = False

for cell in nb.get("cells", []):
    if cell.get("cell_type") == "code":
        source = cell.get("source", [])
        
        # 1. Insert drive mount cell after the first code cell (installations)
        if not drive_mounted:
            new_cells.append(cell)
            new_cells.append({
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from google.colab import drive\n",
                    "drive.mount('/content/drive')\n"
                ]
            })
            drive_mounted = True
            continue
            
        # 2. Remove dataset_metadata = [] from process_utterance cell
        if any("def process_utterance(full_text):" in line for line in source):
            new_source = [line for line in source if line.strip() != "dataset_metadata = []"]
            cell["source"] = new_source
            new_cells.append(cell)
            continue
            
        # 3. Replace the sentence generation cell
        if any("bank_path = \"/content/dataset/sentence_bank.txt\"" in line for line in source) and any("process_utterance" in line for line in source):
            batch_source = [
                "import os\n",
                "import json\n",
                "import shutil\n",
                "import math\n",
                "\n",
                "# Configuration\n",
                "BATCH_SIZE = 500\n",
                "DRIVE_OUTPUT_DIR = \"/content/drive/MyDrive/KionTTS_Dataset\"\n",
                "bank_path = \"/content/dataset/sentence_bank.txt\"\n",
                "\n",
                "os.makedirs(DRIVE_OUTPUT_DIR, exist_ok=True)\n",
                "\n",
                "if not os.path.exists(bank_path):\n",
                "    print(f\"Please upload your sentence bank to {bank_path}\")\n",
                "else:\n",
                "    with open(bank_path, \"r\") as f:\n",
                "        sentence_bank = [line.strip() for line in f.readlines() if line.strip()]\n",
                "    \n",
                "    total_sentences = len(sentence_bank)\n",
                "    print(f\"Loaded {total_sentences} sentences from the bank.\")\n",
                "    \n",
                "    total_batches = math.ceil(total_sentences / BATCH_SIZE)\n",
                "    \n",
                "    # Check Drive to find the last completed batch\n",
                "    highest_batch = 0\n",
                "    existing_files = os.listdir(DRIVE_OUTPUT_DIR)\n",
                "    for file in existing_files:\n",
                "        if file.startswith(\"batch_\") and file.endswith(\".zip\"):\n",
                "            try:\n",
                "                batch_num = int(file.replace(\"batch_\", \"\").replace(\".zip\", \"\"))\n",
                "                if batch_num >= highest_batch:\n",
                "                    highest_batch = batch_num + 1\n",
                "            except ValueError:\n",
                "                pass\n",
                "                \n",
                "    if highest_batch > 0:\n",
                "        print(f\"Found existing batches up to batch_{highest_batch-1:04d}.zip on Drive.\")\n",
                "        print(f\"Resuming from batch_{highest_batch:04d}...\")\n",
                "    else:\n",
                "        print(\"No existing batches found. Starting from batch_0000...\")\n",
                "        \n",
                "    for batch_idx in range(highest_batch, total_batches):\n",
                "        start_idx = batch_idx * BATCH_SIZE\n",
                "        end_idx = min(start_idx + BATCH_SIZE, total_sentences)\n",
                "        batch_sentences = sentence_bank[start_idx:end_idx]\n",
                "        batch_name = f\"batch_{batch_idx:04d}\"\n",
                "        \n",
                "        print(f\"\\n--- Processing {batch_name} (Sentences {start_idx} to {end_idx-1}) ---\")\n",
                "        \n",
                "        # Clear out previous dataset audio files\n",
                "        shutil.rmtree(\"/content/dataset/wavs\", ignore_errors=True)\n",
                "        shutil.rmtree(\"/content/dataset/temp\", ignore_errors=True)\n",
                "        os.makedirs(\"/content/dataset/wavs\", exist_ok=True)\n",
                "        os.makedirs(\"/content/dataset/temp\", exist_ok=True)\n",
                "        \n",
                "        # Reset metadata for the new batch\n",
                "        dataset_metadata = []\n",
                "        \n",
                "        for text in batch_sentences:\n",
                "            process_utterance(text)\n",
                "            \n",
                "        # Save metadata for this batch\n",
                "        with open(\"/content/dataset/metadata.json\", \"w\") as f:\n",
                "            json.dump(dataset_metadata, f, indent=2)\n",
                "            \n",
                "        print(f\"Zipping {batch_name}...\")\n",
                "        zip_path = f\"/content/{batch_name}.zip\"\n",
                "        shutil.make_archive(f\"/content/{batch_name}\", 'zip', \"/content/dataset\")\n",
                "        \n",
                "        print(f\"Uploading {batch_name}.zip to Google Drive...\")\n",
                "        shutil.copy(zip_path, os.path.join(DRIVE_OUTPUT_DIR, f\"{batch_name}.zip\"))\n",
                "        \n",
                "        # Clean up local zip file\n",
                "        os.remove(zip_path)\n",
                "        \n",
                "    print(\"\\nAll batches completed successfully!\")\n"
            ]
            cell["source"] = batch_source
            new_cells.append(cell)
            continue
            
        new_cells.append(cell)
    else:
        new_cells.append(cell)

nb["cells"] = new_cells

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Notebook updated successfully.")
