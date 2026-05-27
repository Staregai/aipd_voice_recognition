from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


WINDOW_OPTIONS = [
    "rectangular",
    "hann",
    "hamming",
    "blackman",
]


@dataclass(slots=True)
class AudioSignal:
    path: Path
    samples: np.ndarray
    sample_rate: int
    channels: int
    bit_depth: int

    @property
    def duration_seconds(self) -> float:
        return float(len(self.samples)) / float(self.sample_rate)


def _decode_pcm(raw: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float64)
        return (samples - 128.0) / 128.0

    if sample_width == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float64)
        return samples / 32768.0

    if sample_width == 3:
        data = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3)
        values = (
            data[:, 0].astype(np.int32)
            | (data[:, 1].astype(np.int32) << 8)
            | (data[:, 2].astype(np.int32) << 16)
        )
        sign_bit = 1 << 23
        values = (values ^ sign_bit) - sign_bit
        return values.astype(np.float64) / float(1 << 23)

    if sample_width == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float64)
        return samples / float(1 << 31)

    raise ValueError(f"Unsupported WAV sample width: {sample_width} bytes")


def load_wav_mono(filepath: str | Path) -> AudioSignal:
    path = Path(filepath).resolve()
    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_rate = wav_file.getframerate()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)

    samples = _decode_pcm(raw, sample_width)
    samples = samples.reshape(-1, channels)
    if channels > 1:
        samples = samples.mean(axis=1)
    else:
        samples = samples[:, 0]

    return AudioSignal(
        path=path,
        samples=np.asarray(samples, dtype=np.float64),
        sample_rate=sample_rate,
        channels=channels,
        bit_depth=sample_width * 8,
    )


def normalize_peak(samples: np.ndarray, peak: float = 0.99) -> np.ndarray:
    if samples.size == 0:
        return samples.astype(np.float64, copy=True)
    max_abs = float(np.max(np.abs(samples)))
    if max_abs <= 1e-12:
        return samples.astype(np.float64, copy=True)
    return np.asarray(samples, dtype=np.float64) * (peak / max_abs)


def pre_emphasis(samples: np.ndarray, coefficient: float = 0.97) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.size == 0:
        return samples.copy()
    emphasized = np.empty_like(samples)
    emphasized[0] = samples[0]
    emphasized[1:] = samples[1:] - coefficient * samples[:-1]
    return emphasized


def milliseconds_to_samples(milliseconds: float, sample_rate: int) -> int:
    return max(1, int(round(milliseconds * sample_rate / 1000.0)))


def frame_signal(
    samples: np.ndarray,
    frame_size: int,
    hop_size: int,
    *,
    pad_end: bool = True,
) -> np.ndarray:
    samples = np.asarray(samples, dtype=np.float64)
    if frame_size <= 0 or hop_size <= 0:
        raise ValueError("frame_size and hop_size must be positive")
    if samples.size == 0:
        return np.zeros((0, frame_size), dtype=np.float64)

    if samples.size < frame_size:
        if not pad_end:
            return np.zeros((0, frame_size), dtype=np.float64)
        padding = frame_size - samples.size
        samples = np.pad(samples, (0, padding))

    if pad_end:
        remainder = (samples.size - frame_size) % hop_size
        if remainder != 0:
            padding = hop_size - remainder
            samples = np.pad(samples, (0, padding))

    frame_count = 1 + max(0, (samples.size - frame_size) // hop_size)
    frames = np.zeros((frame_count, frame_size), dtype=np.float64)

    for index in range(frame_count):
        start = index * hop_size
        frames[index] = samples[start : start + frame_size]

    return frames


def get_window(window_type: str, size: int) -> np.ndarray:
    if size <= 0:
        return np.zeros(0, dtype=np.float64)
    if size == 1:
        return np.ones(1, dtype=np.float64)

    if window_type == "rectangular":
        return np.ones(size, dtype=np.float64)
    if window_type == "hann":
        return np.hanning(size)
    if window_type == "blackman":
        return np.blackman(size)
    return np.hamming(size)


def apply_window(frames: np.ndarray, window_type: str) -> np.ndarray:
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim == 1:
        return frames * get_window(window_type, frames.shape[0])
    window = get_window(window_type, frames.shape[1])
    return frames * window[np.newaxis, :]


def time_axis(length: int, sample_rate: int) -> np.ndarray:
    return np.arange(length, dtype=np.float64) / float(sample_rate)


def frame_time_axis(frame_count: int, hop_size: int, sample_rate: int) -> np.ndarray:
    return np.arange(frame_count, dtype=np.float64) * hop_size / float(sample_rate)


def safe_mean(vectors: Iterable[np.ndarray]) -> np.ndarray:
    items = [np.asarray(vector, dtype=np.float64) for vector in vectors]
    if not items:
        return np.zeros(0, dtype=np.float64)
    return np.mean(np.stack(items, axis=0), axis=0)
