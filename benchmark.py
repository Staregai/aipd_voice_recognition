from __future__ import annotations

import argparse
from pathlib import Path

from dataset import discover_recordings, filter_records_for_common_words
from evaluation import (
    evaluate_speaker_identification,
    evaluate_speaker_verification,
    evaluate_word_recognition,
    split_by_repetition,
)
from feature_extraction import FormantConfig, MFCCConfig
from recognizer import MFCCDTWRecognizer


FEATURE_CHOICES = ["mfcc", "fft", "spectrogram", "formants", "all"]


def load_records(dataset_dir: str, normalized_only: bool, common_words_only: bool):
    records = discover_recordings(dataset_dir, normalized_only=normalized_only)
    if common_words_only:
        records = filter_records_for_common_words(records)
    return records


def build_recognizer(args: argparse.Namespace, feature_type: str) -> MFCCDTWRecognizer:
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
    return MFCCDTWRecognizer(
        mfcc_config=mfcc_config,
        formant_config=formant_config,
        feature_type=feature_type,
    )


def run_task(
    recognizer: MFCCDTWRecognizer,
    task: str,
    gallery,
    query,
) -> None:
    if task == "word_recognition":
        report = evaluate_word_recognition(recognizer, gallery, query)
        print(f"task={report.task}")
        print(f"accuracy={report.accuracy:.4f}")
        print(f"samples={len(report.results)}")
        return

    if task == "speaker_identification":
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


def run_benchmark(args: argparse.Namespace) -> None:
    records = load_records(args.dataset_dir, args.normalized_only, args.common_words_only)
    gallery, query = split_by_repetition(
        records,
        gallery_repetition=args.gallery_repetition,
        query_repetition=args.query_repetition,
    )

    if args.limit is not None:
        query = query[: args.limit]

    if args.features == "all":
        features = ["mfcc", "fft", "spectrogram", "formants"]
    else:
        features = [args.features]

    for index, feature_type in enumerate(features, start=1):
        print("=" * 72)
        print(f"run={index}/{len(features)}")
        print(f"features={feature_type}")
        print(
            f"params=frame_ms={args.frame_ms:g}, hop_ms={args.hop_ms:g}, n_fft={args.n_fft}, "
            f"lpc_order={args.lpc_order}, n_formants={args.n_formants}"
        )
        recognizer = build_recognizer(args, feature_type)
        run_task(recognizer, args.task, gallery, query)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AiPD Projekt 3 - benchmark")
    parser.add_argument("--dataset_dir", type=str, default=str(Path(__file__).resolve().parent.parent / "dzwiek_data"))
    parser.add_argument(
        "--task",
        type=str,
        choices=["word_recognition", "speaker_identification", "speaker_verification"],
        default="word_recognition",
    )
    parser.add_argument("--normalized_only", action="store_true", default=True)
    parser.add_argument("--common_words_only", action="store_true", default=True)
    parser.add_argument("--gallery_repetition", type=int, default=1, choices=[1, 2])
    parser.add_argument("--query_repetition", type=int, default=2, choices=[1, 2])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--features", type=str, choices=FEATURE_CHOICES, default="mfcc")
    parser.add_argument("--frame_ms", type=float, default=25.0)
    parser.add_argument("--hop_ms", type=float, default=10.0)
    parser.add_argument("--n_fft", type=int, default=512)
    parser.add_argument("--lpc_order", type=int, default=14)
    parser.add_argument("--n_formants", type=int, default=3)
    parser.set_defaults(func=run_benchmark)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
