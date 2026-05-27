from __future__ import annotations

import argparse
from pathlib import Path

from dataset import build_record_from_path, discover_recordings, filter_records_for_common_words
from evaluation import (
    evaluate_speaker_identification,
    evaluate_speaker_verification,
    evaluate_word_recognition,
    split_by_repetition,
)
from feature_extraction import FormantConfig, MFCCConfig
from recognizer import MFCCDTWRecognizer


def load_records(dataset_dir: str, normalized_only: bool, common_words_only: bool):
    records = discover_recordings(dataset_dir, normalized_only=normalized_only)
    if common_words_only:
        records = filter_records_for_common_words(records)
    return records


def build_recognizer(args: argparse.Namespace) -> MFCCDTWRecognizer:
    mfcc_config = MFCCConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        n_fft=args.n_fft,
    )
    formant_config = FormantConfig(
        frame_ms=args.frame_ms,
        hop_ms=args.hop_ms,
        lpc_order=args.lpc_order,
        n_formants=args.n_formants,
    )
    recognizer = MFCCDTWRecognizer(
        mfcc_config=mfcc_config,
        formant_config=formant_config,
        feature_type=args.features,
    )
    return recognizer


def run_benchmark(args: argparse.Namespace) -> None:
    records = load_records(args.dataset_dir, args.normalized_only, args.common_words_only)
    gallery, query = split_by_repetition(
        records,
        gallery_repetition=args.gallery_repetition,
        query_repetition=args.query_repetition,
    )
    recognizer = build_recognizer(args)

    if args.limit is not None:
        query = query[: args.limit]

    if args.task == "word_recognition":
        report = evaluate_word_recognition(recognizer, gallery, query)
        print(f"task={report.task}")
        print(f"accuracy={report.accuracy:.4f}")
        print(f"samples={len(report.results)}")
        return

    if args.task == "speaker_identification":
        report = evaluate_speaker_identification(recognizer, gallery, query)
        print(f"task={report.task}")
        print(f"accuracy={report.accuracy:.4f}")
        print(f"samples={len(report.results)}")
        return

    report = evaluate_speaker_verification(recognizer, gallery, query)
    print("task=speaker_verification")
    print(f"accuracy={report.accuracy:.4f}")
    print(f"precision={report.precision:.4f}")
    print(f"recall={report.recall:.4f}")
    print(f"f1={report.f1:.4f}")
    print(f"decisions={len(report.results)}")


def run_recognition(args: argparse.Namespace) -> None:
    records = load_records(args.dataset_dir, args.normalized_only, args.common_words_only)
    recognizer = build_recognizer(args)
    query = build_record_from_path(args.query_file)
    gallery = [record for record in records if Path(record.path) != Path(query.path)]

    if args.task == "word_recognition":
        result = recognizer.recognize_word(query, gallery)
    elif args.task == "speaker_identification":
        result = recognizer.identify_speaker(query, gallery)
    else:
        if not args.claimed_speaker:
            raise ValueError("--claimed_speaker is required for speaker_verification")
        result = recognizer.verify_speaker(query, gallery, args.claimed_speaker)

    print(f"task={result.task}")
    print(f"query={result.query.path.name}")
    print(f"predicted={result.predicted_label}")
    print(f"expected={result.expected_label}")
    print(f"best_template={result.best_template.path.name}")
    print(f"dtw_cost={result.best_dtw.normalized_cost:.6f}")
    if result.claimed_speaker is not None:
        print(f"claimed_speaker={result.claimed_speaker}")
        print(f"accepted={result.accepted}")
        print(f"speaker_score={result.claimed_speaker_score:.6f}")
        print(f"impostor_score={result.nearest_impostor_score:.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AiPD Projekt 3 - MFCC + DTW")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run benchmark on database split")
    benchmark.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "dzwiek_data"))
    benchmark.add_argument(
        "--task",
        type=str,
        choices=["word_recognition", "speaker_identification", "speaker_verification"],
        default="word_recognition",
    )
    benchmark.add_argument("--normalized_only", action="store_true", default=True)
    benchmark.add_argument("--common_words_only", action="store_true", default=True)
    benchmark.add_argument("--gallery_repetition", type=int, default=1, choices=[1, 2])
    benchmark.add_argument("--query_repetition", type=int, default=2, choices=[1, 2])
    benchmark.add_argument("--limit", type=int, default=None)
    benchmark.add_argument(
        "--features",
        type=str,
        choices=["mfcc", "fft", "spectrogram", "formants"],
        default="mfcc",
    )
    benchmark.add_argument("--frame_ms", type=float, default=25.0)
    benchmark.add_argument("--hop_ms", type=float, default=10.0)
    benchmark.add_argument("--n_fft", type=int, default=512)
    benchmark.add_argument("--lpc_order", type=int, default=14)
    benchmark.add_argument("--n_formants", type=int, default=3)
    benchmark.set_defaults(func=run_benchmark)

    recognize = subparsers.add_parser("recognize", help="Recognize one WAV file")
    recognize.add_argument("query_file", type=str)
    recognize.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "dzwiek_data"))
    recognize.add_argument(
        "--task",
        type=str,
        choices=["word_recognition", "speaker_identification", "speaker_verification"],
        default="word_recognition",
    )
    recognize.add_argument(
        "--features",
        type=str,
        choices=["mfcc", "fft", "spectrogram", "formants"],
        default="mfcc",
    )
    recognize.add_argument("--claimed_speaker", type=str, default="")
    recognize.add_argument("--normalized_only", action="store_true", default=True)
    recognize.add_argument("--common_words_only", action="store_true", default=True)
    recognize.add_argument("--frame_ms", type=float, default=25.0)
    recognize.add_argument("--hop_ms", type=float, default=10.0)
    recognize.add_argument("--n_fft", type=int, default=512)
    recognize.add_argument("--lpc_order", type=int, default=14)
    recognize.add_argument("--n_formants", type=int, default=3)
    recognize.set_defaults(func=run_recognition)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
