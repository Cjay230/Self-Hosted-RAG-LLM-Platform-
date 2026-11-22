"""Capture: turn raw meeting files and documents into chunk records.

Only parse_transcript, chunk_turns and chunk_document live here so far.
transcribe() (Carla) and ocr_page() (Hadi) are specified in INTERFACES.md
and will be added separately.
"""

import os
import re
from datetime import datetime

# Header "Language:" values look like "English" or
# "English / Arabic / French (code-switching)".
LANGUAGE_CODES = {"english": "en", "arabic": "ar", "french": "fr"}

# A dialogue line is an all-caps speaker name, a colon, then the words.
TURN_PATTERN = re.compile(r"^([A-Z]+):\s*(.*)$")

# A sentence ends at . ! ? or the Arabic question mark, followed by whitespace.
SENTENCE_END = re.compile(r"(?<=[.!?؟])\s+")


def count_words(text):
    # TODO: replace word counts with the real Qwen tokenizer once Ollama is
    # installed. Arabic costs more tokens per word than English, so word
    # counts under-estimate Arabic chunks against the context budget.
    return len(text.split())


def parse_transcript(path: str) -> tuple[dict, list[dict]]:
    """Read a meeting file. Returns (doc_meta, turns) where
    doc_meta = {"doc_id", "date", "language", "source_type", "attendees"}
    turns    = [{"speaker": "RANA", "text": "..."}, ...]"""
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Everything above the "---" line is the header, everything below is dialogue.
    header, dialogue = content.split("\n---\n", 1)
    header_lines = header.strip().splitlines()

    # Line 2 is "Meridian Logistics · 12 January 2026 · Conference Room B".
    date_text = header_lines[1].split("·")[1].strip()
    date = datetime.strptime(date_text, "%d %B %Y").strftime("%Y-%m-%d")

    language = None
    attendees = []
    for line in header_lines:
        if line.startswith("Language:"):
            # For code-switched meetings we take the first language listed.
            # Per-chunk language detection replaces this later.
            first = line.split(":", 1)[1].split("/")[0].strip().lower()
            language = LANGUAGE_CODES[first]
        if line.startswith("Attendees:"):
            # "Rana Khalil (Operations Director), Samir Haddad (Finance), ..."
            # Roles never contain commas, so splitting on commas is safe.
            for person in line.split(":", 1)[1].split(","):
                name = person.split("(")[0].strip()
                attendees.append(name)

    doc_id = os.path.splitext(os.path.basename(path))[0]
    doc_meta = {
        "doc_id": doc_id,
        "date": date,
        "language": language,
        "source_type": "transcript",
        "attendees": attendees,
    }

    turns = []
    for line in dialogue.splitlines():
        line = line.strip()
        if not line:
            continue
        match = TURN_PATTERN.match(line)
        if match:
            turns.append({"speaker": match.group(1), "text": match.group(2)})
        elif turns:
            # A line without a speaker label continues the previous turn.
            turns[-1]["text"] += " " + line

    return doc_meta, turns


def make_chunk_id(doc_id, index):
    # "meeting_01" -> "m01-001", matching the example in INTERFACES.md.
    # Other documents keep their full doc_id as the prefix.
    if doc_id.startswith("meeting_"):
        prefix = "m" + doc_id[len("meeting_"):]
    else:
        prefix = doc_id
    return f"{prefix}-{index:03d}"


def pack_units(units, target_words, overlap_words):
    """Group whole units (turns or sentences) into chunks of about target_words.

    A unit is never split. If adding a unit would go past the target, the
    current chunk is closed and a new one starts. A single unit longer than
    the target becomes a chunk on its own rather than being cut.

    Overlap is made of whole units too: the new chunk starts with the last
    units of the previous chunk, as many as fit inside overlap_words. We
    overlap whole units rather than a fixed 50 words because a partial unit
    would mean splitting a turn or a sentence, which is exactly what we must
    never do.
    """
    chunks = []
    current = []
    current_words = 0

    for unit in units:
        unit_words = count_words(unit["text"])

        if current and current_words + unit_words > target_words:
            chunks.append(current)

            # Walk backwards through the closed chunk collecting overlap.
            # Never repeat the whole previous chunk, or it adds nothing new.
            overlap = []
            overlap_total = 0
            for previous in reversed(current[1:]):
                previous_words = count_words(previous["text"])
                if overlap_total + previous_words > overlap_words:
                    break
                overlap.insert(0, previous)
                overlap_total += previous_words

            # Drop the overlap if it would push the new chunk past the target.
            if overlap_total + unit_words > target_words:
                overlap = []
                overlap_total = 0

            current = overlap
            current_words = overlap_total

        current.append(unit)
        current_words += unit_words

    if current:
        chunks.append(current)
    return chunks


def make_record(doc_meta, index, text, speaker):
    return {
        "chunk_id": make_chunk_id(doc_meta["doc_id"], index),
        "doc_id": doc_meta["doc_id"],
        "source_type": doc_meta["source_type"],
        "date": doc_meta["date"],
        "language": doc_meta["language"],
        "speaker": speaker,
        "text": text,
        # INTERFACES.md: set by is_confidential(), not by hand. That SVM does
        # not exist yet, so we leave it unknown rather than guess False.
        "confidential": None,
    }


def chunk_turns(turns: list[dict], doc_meta: dict,
                target_words: int = 150, overlap_words: int = 30) -> list[dict]:
    """Chunk a transcript into chunk records.
    Never splits a speaker's turn across two chunks — if a turn would
    overflow the target, start a new chunk instead. A planted fact that
    gets cut in half can never be retrieved intact.
    speaker is set when the chunk holds one turn, null when it spans several."""
    # Each turn keeps its speaker label in the chunk text, so the LLM can
    # still answer "who owns X?" from a chunk that spans several speakers.
    units = []
    for turn in turns:
        units.append({"speaker": turn["speaker"],
                      "text": turn["speaker"] + ": " + turn["text"]})

    records = []
    for index, group in enumerate(pack_units(units, target_words, overlap_words), start=1):
        text = "\n".join(unit["text"] for unit in group)
        if len(group) == 1:
            speaker = group[0]["speaker"]
        else:
            speaker = None
        records.append(make_record(doc_meta, index, text, speaker))
    return records


def chunk_document(text: str, doc_meta: dict,
                   target_words: int = 150, overlap_words: int = 30) -> list[dict]:
    """Chunk a document or OCR page. Splits on paragraph, then sentence.
    Never splits mid-sentence. speaker is always null."""
    # Paragraphs that fit are kept whole; only an oversized paragraph is
    # broken into sentences. Paragraphs are the more natural unit of meaning.
    units = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        if count_words(paragraph) <= target_words:
            units.append({"text": paragraph})
        else:
            for sentence in SENTENCE_END.split(paragraph):
                units.append({"text": sentence})

    records = []
    for index, group in enumerate(pack_units(units, target_words, overlap_words), start=1):
        chunk_text = "\n".join(unit["text"] for unit in group)
        records.append(make_record(doc_meta, index, chunk_text, None))
    return records
