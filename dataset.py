from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from audio_core import AudioSignal, load_wav_mono


DIGIT_WORDS = {
    "0": "zero",
    "1": "jeden",
    "2": "dwa",
    "3": "trzy",
    "4": "cztery",
    "5": "piec",
    "6": "szesc",
    "7": "siedem",
    "8": "osiem",
    "9": "dziewiec",
    "10": "dziesiec",
}


@dataclass(slots=True)
class RecordingRecord:
    path: Path
    speaker_id: str
    speaker_label: str
    gender: str
    normalized: bool
    repetition: int | None
    raw_token: str
    canonical_word: str

    @property
    def file_label(self) -> str:
        return self.path.name


def is_normalized_directory_name(name: str) -> bool:
    lowered = name.strip().lower()
    return lowered in {"znormalizowane", "normalized"}


def _ascii_fold(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    stripped = "".join(char for char in normalized if not unicodedata.combining(char))
    return stripped


def canonicalize_token(token: str) -> str:
    cleaned = token.strip().replace("-", " ").replace("_", " ")
    cleaned = cleaned.replace("α", "o").replace("½", "z").replace("⌐", "e").replace("å", "c")
    cleaned = cleaned.replace("╛", "z").replace("Σ", "n").replace("Ñ", "n").replace("ÿ", "s")
    cleaned = _ascii_fold(cleaned).lower()
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    if cleaned in DIGIT_WORDS:
        return DIGIT_WORDS[cleaned]

    if "ala ma rudego kota" in cleaned or cleaned == "zdanie ala" or cleaned == "zdanie1":
        return "zdanie_ala"
    if "dlaczego ta zima" in cleaned or cleaned == "zdanie zima" or cleaned == "zdanie2":
        return "zdanie_zima"
    if "samoch" in cleaned:
        return "samochod"
    if cleaned.startswith("dzies"):
        return "dziesiec"
    if cleaned.startswith("dziew"):
        return "dziewiec"
    if cleaned.startswith("pie"):
        return "piec"
    if cleaned.startswith("sze"):
        return "szesc"
    if cleaned.startswith("chrz"):
        return "chrzaszcz"
    if cleaned.startswith("dzw") or cleaned.startswith("dzw"):
        return "dzwiek"
    if cleaned == "zit":
        return "zid"
    if cleaned in {"bob", "bobola", "bab"}:
        return "bub"

    return cleaned.replace(" ", "_")


def parse_repetition(stem: str) -> tuple[str, int | None]:
    match = re.match(r"^(.*)_(\d+)$", stem)
    if match:
        return match.group(1), int(match.group(2))
    return stem, None


def build_record_from_path(path: str | Path) -> RecordingRecord:
    path = Path(path).resolve()
    speaker_dir = next((parent for parent in path.parents if parent.name.startswith("speaker_")), None)
    if speaker_dir is None:
        raise ValueError(f"Cannot infer speaker from path: {path}")

    speaker_parts = speaker_dir.name.split("_")
    speaker_id = speaker_parts[1] if len(speaker_parts) >= 2 else "unknown"
    gender = speaker_parts[2] if len(speaker_parts) >= 3 else "u"
    normalized = is_normalized_directory_name(path.parent.name)

    raw_token, repetition = parse_repetition(path.stem)
    canonical_word = canonicalize_token(raw_token)
    return RecordingRecord(
        path=path,
        speaker_id=speaker_id,
        speaker_label=f"speaker_{speaker_id}_{gender}",
        gender=gender,
        normalized=normalized,
        repetition=repetition,
        raw_token=raw_token,
        canonical_word=canonical_word,
    )


def discover_recordings(
    root_dir: str | Path,
    *,
    normalized_only: bool = True,
) -> list[RecordingRecord]:
    root = Path(root_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")

    records: list[RecordingRecord] = []
    for speaker_dir in sorted(root.glob("speaker_*")):
        if not speaker_dir.is_dir():
            continue
        for variant_dir in speaker_dir.iterdir():
            if not variant_dir.is_dir():
                continue
            is_normalized = is_normalized_directory_name(variant_dir.name)
            if normalized_only and not is_normalized:
                continue
            for wav_path in sorted(variant_dir.glob("*.wav")):
                records.append(build_record_from_path(wav_path))
    return records


def common_vocabulary(records: Iterable[RecordingRecord], min_speakers: int = 2) -> list[str]:
    speakers_per_word: dict[str, set[str]] = {}
    for record in records:
        speakers_per_word.setdefault(record.canonical_word, set()).add(record.speaker_id)
    return sorted(
        word for word, speakers in speakers_per_word.items() if len(speakers) >= min_speakers
    )


def filter_records_for_common_words(records: Iterable[RecordingRecord]) -> list[RecordingRecord]:
    items = list(records)
    if not items:
        return []
    speaker_count = len({record.speaker_id for record in items})
    allowed = set(common_vocabulary(items, min_speakers=speaker_count))
    return [record for record in items if record.canonical_word in allowed]


def load_signal(record: RecordingRecord | str | Path) -> AudioSignal:
    if isinstance(record, RecordingRecord):
        return load_wav_mono(record.path)
    return load_wav_mono(record)
