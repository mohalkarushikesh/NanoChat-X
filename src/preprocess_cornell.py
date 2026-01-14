import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "data", "cornell_movie_dialogs")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "data.txt")

# Load movie lines
lines_path = os.path.join(DATASET_DIR, "movie_lines.txt")
conversations_path = os.path.join(DATASET_DIR, "movie_conversations.txt")

# Map line IDs to text
id2line = {}
with open(lines_path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.split("+++$+++")
        if len(parts) == 5:
            line_id = parts[0].strip()
            text = parts[4].strip()
            id2line[line_id] = text

pairs = []
with open(conversations_path, encoding="utf-8", errors="ignore") as f:
    for line in f:
        parts = line.split("+++$+++")
        if len(parts) == 4:
            line_ids = eval(parts[3].strip())  # list of line IDs
            for i in range(len(line_ids) - 1):
                input_line = id2line.get(line_ids[i], "")
                target_line = id2line.get(line_ids[i+1], "")
                if input_line and target_line:
                    pairs.append(f"{input_line} -> {target_line}")

# Save to data.txt
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for pair in pairs:
        f.write(pair + "\n")

print(f"Saved {len(pairs)} pairs to {OUTPUT_PATH}")
