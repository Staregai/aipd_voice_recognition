from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dataset import RecordingRecord
from recognizer import MFCCDTWRecognizer, RecognitionResult


@dataclass(slots=True)
class ClassificationReport:
    task: str
    accuracy: float
    labels: list[str]
    confusion_matrix: np.ndarray
    results: list[RecognitionResult]


@dataclass(slots=True)
class VerificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int
    results: list[RecognitionResult]


def split_by_repetition(
    records: list[RecordingRecord],
    *,
    gallery_repetition: int,
    query_repetition: int,
) -> tuple[list[RecordingRecord], list[RecordingRecord]]:
    gallery = [record for record in records if record.repetition == gallery_repetition]
    query = [record for record in records if record.repetition == query_repetition]
    if not gallery or not query:
        raise ValueError("Chosen repetition split produced an empty gallery or query set")
    return gallery, query


def _build_confusion(labels: list[str], truth: list[str], predictions: list[str]) -> np.ndarray:
    label_to_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for expected, predicted in zip(truth, predictions):
        if expected not in label_to_index or predicted not in label_to_index:
            continue
        matrix[label_to_index[expected], label_to_index[predicted]] += 1
    return matrix


def evaluate_word_recognition(
    recognizer: MFCCDTWRecognizer,
    gallery: list[RecordingRecord],
    query: list[RecordingRecord],
) -> ClassificationReport:
    gallery_words = {record.canonical_word for record in gallery}
    effective_query = [record for record in query if record.canonical_word in gallery_words]
    results = [recognizer.recognize_word(record, gallery) for record in effective_query]
    truth = [result.expected_label or "" for result in results]
    predictions = [result.predicted_label for result in results]
    labels = sorted(set(truth) | set(predictions))
    matrix = _build_confusion(labels, truth, predictions)
    accuracy = float(np.mean([expected == predicted for expected, predicted in zip(truth, predictions)]))
    return ClassificationReport(
        task="word_recognition",
        accuracy=accuracy,
        labels=labels,
        confusion_matrix=matrix,
        results=results,
    )


def evaluate_speaker_identification(
    recognizer: MFCCDTWRecognizer,
    gallery: list[RecordingRecord],
    query: list[RecordingRecord],
) -> ClassificationReport:
    results = [recognizer.identify_speaker(record, gallery) for record in query]
    truth = [result.expected_label or "" for result in results]
    predictions = [result.predicted_label for result in results]
    labels = sorted(set(truth) | set(predictions))
    matrix = _build_confusion(labels, truth, predictions)
    accuracy = float(np.mean([expected == predicted for expected, predicted in zip(truth, predictions)]))
    return ClassificationReport(
        task="speaker_identification",
        accuracy=accuracy,
        labels=labels,
        confusion_matrix=matrix,
        results=results,
    )


def evaluate_speaker_verification(
    recognizer: MFCCDTWRecognizer,
    gallery: list[RecordingRecord],
    query: list[RecordingRecord],
    *,
    ratio_threshold: float = 0.92,
) -> VerificationReport:
    results: list[RecognitionResult] = []
    speakers = sorted({record.speaker_label for record in gallery})

    for record in query:
        results.append(
            recognizer.verify_speaker(
                record,
                gallery,
                claimed_speaker=record.speaker_label,
                ratio_threshold=ratio_threshold,
            )
        )

        impostor_candidates = [speaker for speaker in speakers if speaker != record.speaker_label]
        impostor_claim = impostor_candidates[0]
        results.append(
            recognizer.verify_speaker(
                record,
                gallery,
                claimed_speaker=impostor_claim,
                ratio_threshold=ratio_threshold,
            )
        )

    tp = sum(1 for result in results if result.expected_label == "genuine" and result.accepted)
    fn = sum(1 for result in results if result.expected_label == "genuine" and not result.accepted)
    fp = sum(1 for result in results if result.expected_label == "impostor" and result.accepted)
    tn = sum(1 for result in results if result.expected_label == "impostor" and not result.accepted)

    accuracy = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
    return VerificationReport(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        results=results,
    )
