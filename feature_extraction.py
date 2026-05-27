from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from audio_core import apply_window, frame_signal, milliseconds_to_samples, normalize_peak, pre_emphasis


@dataclass(slots=True)
class MFCCConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    pre_emphasis: float = 0.97
    n_fft: int = 512
    n_mels: int = 26
    n_mfcc: int = 13
    low_freq_hz: float = 0.0
    high_freq_hz: float | None = None
    lifter: int = 22
    include_energy: bool = True
    include_deltas: bool = True
    include_delta_deltas: bool = False
    cepstral_mean_normalization: bool = True
    window_type: str = "hamming"

    def feature_dim(self) -> int:
        base = self.n_mfcc
        if self.include_deltas:
            base += self.n_mfcc
        if self.include_delta_deltas:
            base += self.n_mfcc
        return base


@dataclass(slots=True)
class FormantConfig:
    frame_ms: float = 25.0
    hop_ms: float = 10.0
    pre_emphasis: float = 0.97
    lpc_order: int = 14
    n_formants: int = 3
    window_type: str = "hamming"


def hz_to_mel(frequency_hz: np.ndarray | float) -> np.ndarray | float:
    return 2595.0 * np.log10(1.0 + np.asarray(frequency_hz) / 700.0)


def mel_to_hz(mel_value: np.ndarray | float) -> np.ndarray | float:
    return 700.0 * (10.0 ** (np.asarray(mel_value) / 2595.0) - 1.0)


def build_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    low_freq_hz: float = 0.0,
    high_freq_hz: float | None = None,
) -> np.ndarray:
    high_freq_hz = float(high_freq_hz or (sample_rate / 2.0))
    low_mel = float(hz_to_mel(low_freq_hz))
    high_mel = float(hz_to_mel(high_freq_hz))
    mel_points = np.linspace(low_mel, high_mel, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)
    bins = np.clip(bins, 0, n_fft // 2)

    filterbank = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float64)
    for index in range(1, n_mels + 1):
        left = bins[index - 1]
        center = bins[index]
        right = bins[index + 1]
        if center == left:
            center += 1
        if right == center:
            right += 1
        right = min(right, n_fft // 2)

        for bin_index in range(left, min(center, filterbank.shape[1])):
            filterbank[index - 1, bin_index] = (bin_index - left) / max(1, center - left)
        for bin_index in range(center, min(right + 1, filterbank.shape[1])):
            filterbank[index - 1, bin_index] = (right - bin_index) / max(1, right - center)

    normalizer = np.maximum(filterbank.sum(axis=1, keepdims=True), 1e-12)
    return filterbank / normalizer


def build_dct_basis(n_mfcc: int, n_mels: int) -> np.ndarray:
    basis = np.zeros((n_mfcc, n_mels), dtype=np.float64)
    scale0 = np.sqrt(1.0 / n_mels)
    scale = np.sqrt(2.0 / n_mels)

    for coeff in range(n_mfcc):
        for mel_index in range(n_mels):
            angle = np.pi * coeff * (2 * mel_index + 1) / (2.0 * n_mels)
            value = np.cos(angle)
            basis[coeff, mel_index] = value * (scale0 if coeff == 0 else scale)
    return basis


def lifter_ceps(coefficients: np.ndarray, lifter: int) -> np.ndarray:
    if lifter <= 0:
        return coefficients
    n_coeffs = coefficients.shape[1]
    index = np.arange(n_coeffs, dtype=np.float64)
    lift = 1.0 + (lifter / 2.0) * np.sin(np.pi * index / lifter)
    return coefficients * lift[np.newaxis, :]


def compute_deltas(features: np.ndarray, window: int = 2) -> np.ndarray:
    if features.size == 0:
        return features.copy()
    padded = np.pad(features, ((window, window), (0, 0)), mode="edge")
    denominator = 2.0 * sum(index * index for index in range(1, window + 1))
    deltas = np.zeros_like(features)
    for frame_index in range(features.shape[0]):
        numerator = np.zeros(features.shape[1], dtype=np.float64)
        for offset in range(1, window + 1):
            numerator += offset * (
                padded[frame_index + window + offset] - padded[frame_index + window - offset]
            )
        deltas[frame_index] = numerator / denominator
    return deltas


def extract_mfcc(samples: np.ndarray, sample_rate: int, config: MFCCConfig) -> np.ndarray:
    signal = normalize_peak(np.asarray(samples, dtype=np.float64))
    signal = pre_emphasis(signal, config.pre_emphasis)

    frame_size = milliseconds_to_samples(config.frame_ms, sample_rate)
    hop_size = milliseconds_to_samples(config.hop_ms, sample_rate)
    frames = frame_signal(signal, frame_size, hop_size, pad_end=True)
    frames = apply_window(frames, config.window_type)

    spectrum = np.fft.rfft(frames, n=config.n_fft, axis=1)
    power_spectrum = (np.abs(spectrum) ** 2) / float(config.n_fft)

    filterbank = build_mel_filterbank(
        sample_rate=sample_rate,
        n_fft=config.n_fft,
        n_mels=config.n_mels,
        low_freq_hz=config.low_freq_hz,
        high_freq_hz=config.high_freq_hz,
    )
    mel_energies = power_spectrum @ filterbank.T
    mel_energies = np.maximum(mel_energies, 1e-12)
    log_mel_energies = np.log(mel_energies)

    dct_basis = build_dct_basis(config.n_mfcc, config.n_mels)
    mfcc = log_mel_energies @ dct_basis.T

    if config.include_energy:
        frame_energy = np.log(np.maximum(np.sum(frames ** 2, axis=1), 1e-12))
        mfcc[:, 0] = frame_energy

    mfcc = lifter_ceps(mfcc, config.lifter)

    if config.cepstral_mean_normalization and mfcc.shape[0] > 0:
        mfcc = mfcc - np.mean(mfcc, axis=0, keepdims=True)

    feature_blocks = [mfcc]
    if config.include_deltas:
        delta = compute_deltas(mfcc)
        feature_blocks.append(delta)
        if config.include_delta_deltas:
            feature_blocks.append(compute_deltas(delta))

    return np.concatenate(feature_blocks, axis=1)


def _frames_for_fft(samples: np.ndarray, sample_rate: int, config: MFCCConfig) -> np.ndarray:
    signal = normalize_peak(np.asarray(samples, dtype=np.float64))
    signal = pre_emphasis(signal, config.pre_emphasis)
    frame_size = milliseconds_to_samples(config.frame_ms, sample_rate)
    hop_size = milliseconds_to_samples(config.hop_ms, sample_rate)
    frames = frame_signal(signal, frame_size, hop_size, pad_end=True)
    return apply_window(frames, config.window_type)


def extract_fft_features(
    samples: np.ndarray,
    sample_rate: int,
    config: MFCCConfig,
    *,
    use_log: bool = False,
    use_power: bool = False,
    normalize: bool = False,
) -> np.ndarray:
    frames = _frames_for_fft(samples, sample_rate, config)
    spectrum = np.fft.rfft(frames, n=config.n_fft, axis=1)
    if use_power:
        values = (np.abs(spectrum) ** 2) / float(config.n_fft)
    else:
        values = np.abs(spectrum)
    if use_log:
        values = np.log(np.maximum(values, 1e-12))
    if normalize and values.shape[0] > 0:
        mean = np.mean(values, axis=0, keepdims=True)
        std = np.std(values, axis=0, keepdims=True)
        values = (values - mean) / np.maximum(std, 1e-12)
    return values


def extract_spectrogram(samples: np.ndarray, sample_rate: int, config: MFCCConfig) -> np.ndarray:
    return extract_fft_features(
        samples,
        sample_rate,
        config,
        use_log=True,
        use_power=True,
        normalize=True,
    )


def _levinson_durbin(autocorr: np.ndarray, order: int) -> np.ndarray:
    if autocorr[0] <= 1e-12:
        coeffs = np.zeros(order + 1, dtype=np.float64)
        coeffs[0] = 1.0
        return coeffs

    coeffs = np.zeros(order + 1, dtype=np.float64)
    coeffs[0] = 1.0
    error = float(autocorr[0])

    for i in range(1, order + 1):
        acc = autocorr[i]
        for j in range(1, i):
            acc += coeffs[j] * autocorr[i - j]
        reflection = -acc / max(error, 1e-12)

        prev = coeffs.copy()
        for j in range(1, i):
            coeffs[j] = prev[j] + reflection * prev[i - j]
        coeffs[i] = reflection

        error *= 1.0 - reflection * reflection
        if error <= 1e-12:
            break

    return coeffs


def _frame_formants(frame: np.ndarray, sample_rate: int, config: FormantConfig) -> np.ndarray:
    frame = np.asarray(frame, dtype=np.float64)
    if frame.size < 2:
        return np.zeros(config.n_formants, dtype=np.float64)

    order = min(config.lpc_order, frame.size - 1)

    autocorr = np.correlate(frame, frame, mode="full")
    autocorr = autocorr[frame.size - 1 : frame.size + order]
    lpc = _levinson_durbin(autocorr, order)
    roots = np.roots(lpc)
    roots = roots[np.imag(roots) > 1e-6]

    angles = np.arctan2(np.imag(roots), np.real(roots))
    freqs = angles * (sample_rate / (2.0 * np.pi))
    freqs = freqs[(freqs > 50.0) & (freqs < (sample_rate / 2.0))]
    freqs = np.sort(freqs)

    if freqs.size < config.n_formants:
        padded = np.zeros(config.n_formants, dtype=np.float64)
        padded[: freqs.size] = freqs
        return padded
    return freqs[: config.n_formants]


def extract_formants(samples: np.ndarray, sample_rate: int, config: FormantConfig) -> np.ndarray:
    signal = normalize_peak(np.asarray(samples, dtype=np.float64))
    signal = pre_emphasis(signal, config.pre_emphasis)

    frame_size = milliseconds_to_samples(config.frame_ms, sample_rate)
    hop_size = milliseconds_to_samples(config.hop_ms, sample_rate)
    frames = frame_signal(signal, frame_size, hop_size, pad_end=True)
    frames = apply_window(frames, config.window_type)

    formants = np.zeros((frames.shape[0], config.n_formants), dtype=np.float64)
    for index, frame in enumerate(frames):
        formants[index] = _frame_formants(frame, sample_rate, config)
    return formants
