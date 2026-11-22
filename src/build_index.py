"""Build chunks.jsonl from every meeting transcript in data/meetings/.

Run from anywhere:  python src/build_index.py
"""

import glob
import json
import os
import statistics

from capture import chunk_turns, count_words, parse_transcript

# Paths are relative to the repo root, not the current directory, so the
# script behaves the same locally and on Colab.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEETINGS_DIR = os.path.join(REPO_ROOT, "data", "meetings")
OUTPUT_PATH = os.path.join(REPO_ROOT, "chunks.jsonl")


def main():
    all_chunks = []
    # sorted() so chunk order is the same on every machine.
    for path in sorted(glob.glob(os.path.join(MEETINGS_DIR, "*.txt"))):
        doc_meta, turns = parse_transcript(path)
        all_chunks.extend(chunk_turns(turns, doc_meta))

    # ensure_ascii=False keeps Arabic and French readable in the file.
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    lengths = [count_words(chunk["text"]) for chunk in all_chunks]
    print(f"Wrote {len(all_chunks)} chunks to {OUTPUT_PATH}")
    print(f"Chunk length in words: min {min(lengths)}, "
          f"median {statistics.median(lengths)}, max {max(lengths)}")


if __name__ == "__main__":
    main()
