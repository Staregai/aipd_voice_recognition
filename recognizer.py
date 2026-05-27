from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dataset import RecordingRecord, build_record_from_path, load_signal
from dtw import DTWResult, dynamic_time_warping
from feature_extraction import FormantConfig, MFCCConfig, extract_fft_features, extract_formants, extract_mfcc, extract_spectrogram


@dataclass(slots=True)
class RecognitionCandidate:
    label: str
    score: float
    template: RecordingRecord


@dataclass(slots=True)
class RecognitionResult:
    task: str
    query: RecordingRecord
    predicted_label: str
    expected_label: str | None
    best_template: RecordingRecord
    best_dtw: DTWResult
    candidates: list[RecognitionCandidate]
    accepted: bool | None = None
    claimed_speaker: str | None = None
    claimed_speaker_score: float | None = None
    nearest_impostor_score: float | None = None


class MFCCDTWRecognizer:
    def __init__(
        self,
        *,
        mfcc_config: MFCCConfig | None = None,
        formant_config: FormantConfig | None = None,
        feature_type: str = "mfcc",
        dtw_window_ratio: float | None = 0.2,
        distance_metric: str = "euclidean",
    ) -> None:
        self.mfcc_config = mfcc_config or MFCCConfig()
        self.formant_config = formant_config or FormantConfig()
        self.feature_type = feature_type
        self.dtw_window_ratio = dtw_window_ratio
        self.distance_metric = distance_metric
        self._feature_cache: dict[tuple[Path, str], np.ndarray] = {}

    def clear_cache(self) -> None:
        self._feature_cache.clear()

    def set_feature_type(self, feature_type: str) -> None:
        if feature_type != self.feature_type:
            self.feature_type = feature_type
            self.clear_cache()

    def get_features(self, record: RecordingRecord) -> np.ndarray:
        cache_key = (record.path, self.feature_type)
        if cache_key not in self._feature_cache:
            signal = load_signal(record)
            if self.feature_type == "mfcc":
                features = extract_mfcc(
                    signal.samples,
                    signal.sample_rate,
                    self.mfcc_config,
                )
            elif self.feature_type == "fft":
                features = extract_fft_features(
                    signal.samples,
                    signal.sample_rate,
                    self.mfcc_config,
                    use_log=True,
                    use_power=True,
                    normalize=True,
                )
            elif self.feature_type == "spectrogram":
                features = extract_spectrogram(
                    signal.samples,
                    signal.sample_rate,
                    self.mfcc_config,
                )
            elif self.feature_type == "formants":
                features = extract_formants(
                    signal.samples,
                    signal.sample_rate,
                    self.formant_config,
                )
            else:
                raise ValueError(f"Unknown feature type: {self.feature_type}")

            self._feature_cache[cache_key] = features
        return self._feature_cache[cache_key]

    def compare(self, query: RecordingRecord, template: RecordingRecord) -> DTWResult:
        query_features = self.get_features(query)
        template_features = self.get_features(template)
        return dynamic_time_warping(
            query_features,
            template_features,
            metric=self.distance_metric,
            window_ratio=self.dtw_window_ratio,
        )

    def _score_candidates(
        self,
        query: RecordingRecord,
        gallery: list[RecordingRecord],
        label_getter,
    ) -> tuple[list[RecognitionCandidate], RecordingRecord, DTWResult]:
        candidates: list[RecognitionCandidate] = []
        best_template: RecordingRecord | None = None
        best_dtw: DTWResult | None = None

        for template in gallery:
            dtw_result = self.compare(query, template)
            candidates.append(
                RecognitionCandidate(
                    label=label_getter(template),
                    score=dtw_result.normalized_cost,
                    template=template,
                )
            )
            if best_dtw is None or dtw_result.normalized_cost < best_dtw.normalized_cost:
                best_template = template
                best_dtw = dtw_result

        if best_template is None or best_dtw is None:
            raise ValueError("Gallery is empty")

        return candidates, best_template, best_dtw

    @staticmethod
    def _aggregate_by_label(candidates: list[RecognitionCandidate]) -> list[RecognitionCandidate]:
        best_by_label: dict[str, RecognitionCandidate] = {}
        for candidate in candidates:
            current = best_by_label.get(candidate.label)
            if current is None or candidate.score < current.score:
                best_by_label[candidate.label] = candidate
        return sorted(best_by_label.values(), key=lambda item: item.score)

    def recognize_word(
        self,
        query: RecordingRecord,
        gallery: list[RecordingRecord],
    ) -> RecognitionResult:
        candidates, best_template, best_dtw = self._score_candidates(
            query,
            gallery,
            lambda record: record.canonical_word,
        )
        aggregated = self._aggregate_by_label(candidates)
        return RecognitionResult(
            task="word_recognition",
            query=query,
            predicted_label=aggregated[0].label,
            expected_label=query.canonical_word,
            best_template=best_template,
            best_dtw=best_dtw,
            candidates=aggregated[:10],
        )

    def identify_speaker(
        self,
        query: RecordingRecord,
        gallery: list[RecordingRecord],
    ) -> RecognitionResult:
        same_word_gallery = [record for record in gallery if record.canonical_word == query.canonical_word]
        effective_gallery = same_word_gallery or gallery
        candidates, best_template, best_dtw = self._score_candidates(
            query,
            effective_gallery,
            lambda record: record.speaker_label,
        )
        aggregated = self._aggregate_by_label(candidates)
        return RecognitionResult(
            task="speaker_identification",
            query=query,
            predicted_label=aggregated[0].label,
            expected_label=query.speaker_label,
            best_template=best_template,
            best_dtw=best_dtw,
            candidates=aggregated[:10],
        )

    def verify_speaker(
        self,
        query: RecordingRecord,
        gallery: list[RecordingRecord],
        claimed_speaker: str,
        ratio_threshold: float = 0.92,
    ) -> RecognitionResult:
        same_word_gallery = [record for record in gallery if record.canonical_word == query.canonical_word]
        effective_gallery = same_word_gallery or gallery

        speaker_templates = [record for record in effective_gallery if record.speaker_label == claimed_speaker]
        impostor_templates = [record for record in effective_gallery if record.speaker_label != claimed_speaker]
        if not speaker_templates:
            raise ValueError(f"No templates found for claimed speaker: {claimed_speaker}")
        if not impostor_templates:
            raise ValueError("Verification needs at least one impostor template")

        speaker_scores: list[tuple[float, RecordingRecord, DTWResult]] = []
        for template in speaker_templates:
            dtw_result = self.compare(query, template)
            speaker_scores.append((dtw_result.normalized_cost, template, dtw_result))

        impostor_scores: list[tuple[float, RecordingRecord, DTWResult]] = []
        for template in impostor_templates:
            dtw_result = self.compare(query, template)
            impostor_scores.append((dtw_result.normalized_cost, template, dtw_result))

        speaker_score, best_template, best_dtw = min(speaker_scores, key=lambda item: item[0])
        impostor_score, _, _ = min(impostor_scores, key=lambda item: item[0])

        accepted = speaker_score <= impostor_score * ratio_threshold
        ranked = self._aggregate_by_label(
            [
                RecognitionCandidate(
                    label=template.speaker_label,
                    score=score,
                    template=template,
                )
                for score, template, _ in speaker_scores + impostor_scores
            ]
        )

        return RecognitionResult(
            task="speaker_verification",
            query=query,
            predicted_label=claimed_speaker if accepted else "impostor",
            expected_label="genuine" if query.speaker_label == claimed_speaker else "impostor",
            best_template=best_template,
            best_dtw=best_dtw,
            candidates=ranked[:10],
            accepted=accepted,
            claimed_speaker=claimed_speaker,
            claimed_speaker_score=speaker_score,
            nearest_impostor_score=impostor_score,
        )

    def create_ad_hoc_record(self, filepath: str | Path) -> RecordingRecord:
        return build_record_from_path(filepath)
