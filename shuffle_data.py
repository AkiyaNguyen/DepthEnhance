import argparse
import json
import random
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Shuffle filenames from a directory and save as datalist JSON.")
    parser.add_argument("--input-dir", type=Path, required=True, \
        help="Directory containing files to include in the datalist.")
    parser.add_argument("--seed", type=int, required=True, \
        help="Random seed used to shuffle file names.")
    parser.add_argument("--output-json", type=Path, required=True, \
        help="Output JSON path, e.g. datalist_info/official/kvasir_SEG_seed_42.json")
    args = parser.parse_args()

    file_names = sorted(path.name for path in args.input_dir.iterdir() \
        if path.is_file())
    rng = random.Random(args.seed)
    rng.shuffle(file_names)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as output_file:
        json.dump(file_names, output_file, indent=2)

    print(f"Saved {len(file_names)} shuffled entries to \
        {args.output_json} (seed={args.seed}).")


if __name__ == "__main__":
    main()


# !python shuffle_data.py --input-dir /kaggle/input/datasets/akiyanguyen/polypdataset/polypDataset_final1/kvasir_SEG/Train \
# --seed 42 --output-json datalist_info/official/kvasir_SEG_seed_42.json

