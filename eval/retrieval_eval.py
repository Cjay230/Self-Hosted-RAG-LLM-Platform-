"""Retrieval evaluation: Recall@1, @3 and @5 over the 60 planted facts.

A fact counts as found at rank r if any of the top r chunks comes from the
fact's meeting and contains its source_quote. Results are reported overall,
by question language and by source language, and written to
results/retrieval.csv.

Usage:
    python eval/retrieval_eval.py --method tfidf --k 5
    python eval/retrieval_eval.py --method tfidf --k 5 --note PROVISIONAL
"""

import argparse
import csv
import os
import sys
from datetime import date

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
from retrieval import retrieve  # noqa: E402

FACTS_PATH = os.path.join(REPO_ROOT, "facts.csv")
CHUNKS_PATH = os.path.join(REPO_ROOT, "chunks.jsonl")
RESULTS_PATH = os.path.join(REPO_ROOT, "results", "retrieval.csv")
COLUMNS = ["component", "setting", "metric", "value", "date"]


def first_hit_rank(fact, chunks):
    """Rank (1-based) of the first retrieved chunk holding the fact, or None.

    We accept any chunk that holds the quote, not one specific chunk_id:
    with overlap, a quote can legitimately sit in two neighbouring chunks,
    and either one gives the LLM what it needs.
    """
    for rank, chunk in enumerate(chunks, start=1):
        if chunk["doc_id"] == fact["doc_id"] and fact["source_quote"] in chunk["text"]:
            return rank
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default="tfidf", choices=["tfidf", "dense", "hybrid"])
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--note", default="",
                        help='appended to the setting, e.g. "PROVISIONAL"')
    args = parser.parse_args()

    # Recall is reported at 1, 3 and 5, plus k itself, but never past k:
    # we cannot judge rank 5 if we only asked for 3 chunks.
    cutoffs = sorted({c for c in (1, 3, 5, args.k) if c <= args.k})

    with open(FACTS_PATH, encoding="utf-8", newline="") as f:
        facts = list(csv.DictReader(f))
    with open(CHUNKS_PATH, encoding="utf-8") as f:
        n_chunks = sum(1 for _ in f)

    # The chunk count is in the setting so numbers from a small corpus are
    # never confused with numbers from the full one.
    setting = f"{args.method}, k={args.k}, {n_chunks} chunks"
    if args.note:
        setting += ", " + args.note

    # Each fact belongs to three groups: everything, its question language,
    # and its source language. Count totals and hits per group.
    totals = {}
    hits = {}
    for fact in facts:
        rank = first_hit_rank(fact, retrieve(fact["question"], k=args.k, method=args.method))
        groups = ["", "_qlang_" + fact["question_language"], "_srclang_" + fact["language"]]
        for group in groups:
            totals[group] = totals.get(group, 0) + 1
            for cutoff in cutoffs:
                key = (cutoff, group)
                if rank is not None and rank <= cutoff:
                    hits[key] = hits.get(key, 0) + 1
                else:
                    hits.setdefault(key, 0)

    today = date.today().isoformat()
    group_order = sorted(totals)  # "" (overall) sorts first
    new_rows = []
    for group in group_order:
        new_rows.append(["retrieval", setting, "n" + group, totals[group], today])
    for cutoff in cutoffs:
        for group in group_order:
            recall = hits[(cutoff, group)] / totals[group]
            new_rows.append(["retrieval", setting, f"recall@{cutoff}{group}",
                             round(recall, 3), today])

    # Rerunning the same setting replaces its old rows instead of adding
    # duplicates; rows for every other setting are kept untouched.
    kept_rows = []
    if os.path.exists(RESULTS_PATH):
        with open(RESULTS_PATH, encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                if row["setting"] != setting:
                    kept_rows.append([row[c] for c in COLUMNS])

    with open(RESULTS_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows(kept_rows)
        writer.writerows(new_rows)

    # Human-readable summary with raw counts, since some groups are small.
    print(f"Setting: {setting}")
    header = "group".ljust(14) + "n".rjust(4)
    for cutoff in cutoffs:
        header += f"  R@{cutoff}".rjust(13)
    print(header)
    for group in group_order:
        line = (group.strip("_") or "all").ljust(14) + str(totals[group]).rjust(4)
        for cutoff in cutoffs:
            h = hits[(cutoff, group)]
            line += f"  {h:2d}/{totals[group]:<2d} {h / totals[group]:.2f}".rjust(13)
        print(line)
    print(f"Wrote {len(new_rows)} rows to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
