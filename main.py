from __future__ import annotations

import threading
from pathlib import Path

import tkinter as tk
import numpy as np
from tkinter import filedialog, messagebox, ttk

from audio_core import frame_time_axis, milliseconds_to_samples, time_axis
from dataset import (
    RecordingRecord,
    build_record_from_path,
    common_vocabulary,
    discover_recordings,
    filter_records_for_common_words,
    load_signal,
)
from export_handler import export_text
from feature_extraction import MFCCConfig, extract_spectrogram
from recognizer import MFCCDTWRecognizer, RecognitionResult

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class Project3GUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("AiPD Projekt 3 - MFCC + DTW")
        self.root.geometry("1200x720")
        self.root.minsize(1000, 640)

        self.dataset_dir = Path(__file__).resolve().parent.parent / "dzwiek_data"
        self.records: list[RecordingRecord] = []
        self.query_record: RecordingRecord | None = None
        self.last_result: RecognitionResult | None = None
        self.recognizer = MFCCDTWRecognizer(mfcc_config=MFCCConfig())
        self._busy = False
        self._action_buttons: list[ttk.Button] = []

        self.frame_ms_var = tk.DoubleVar(value=self.recognizer.mfcc_config.frame_ms)
        self.hop_ms_var = tk.DoubleVar(value=self.recognizer.mfcc_config.hop_ms)
        self.n_fft_var = tk.IntVar(value=self.recognizer.mfcc_config.n_fft)
        self.lpc_order_var = tk.IntVar(value=self.recognizer.formant_config.lpc_order)
        self.n_formants_var = tk.IntVar(value=self.recognizer.formant_config.n_formants)

        self._build_ui()

    def _build_ui(self) -> None:
        control_frame = ttk.LabelFrame(self.root, text="Sterowanie", padding=10)
        control_frame.pack(fill=tk.X, padx=6, pady=6)

        self.dataset_var = tk.StringVar(value=str(self.dataset_dir))
        ttk.Label(control_frame, text="Baza:").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(control_frame, textvariable=self.dataset_var, width=70).grid(
            row=0, column=1, columnspan=4, sticky="ew", padx=4, pady=4
        )
        load_db_button = ttk.Button(control_frame, text="Wczytaj bazę", command=self.load_database)
        load_db_button.grid(row=0, column=5, padx=4, pady=4)

        self.normalized_only_var = tk.BooleanVar(value=True)
        self.common_words_only_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(control_frame, text="Tylko znormalizowane", variable=self.normalized_only_var).grid(
            row=1, column=0, sticky="w", padx=4, pady=4
        )
        ttk.Checkbutton(control_frame, text="Tylko wspólne słowa", variable=self.common_words_only_var).grid(
            row=1, column=1, sticky="w", padx=4, pady=4
        )

        self.task_var = tk.StringVar(value="word_recognition")
        ttk.Label(control_frame, text="Zadanie:").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        ttk.Combobox(
            control_frame,
            textvariable=self.task_var,
            state="readonly",
            values=["word_recognition", "speaker_identification", "speaker_verification"],
            width=24,
        ).grid(row=1, column=3, sticky="w", padx=4, pady=4)

        self.feature_var = tk.StringVar(value="mfcc")
        ttk.Label(control_frame, text="Cechy:").grid(row=1, column=4, sticky="e", padx=4, pady=4)
        ttk.Combobox(
            control_frame,
            textvariable=self.feature_var,
            state="readonly",
            values=["mfcc", "fft", "spectrogram", "formants"],
            width=14,
        ).grid(row=1, column=5, sticky="w", padx=4, pady=4)

        self.claimed_speaker_var = tk.StringVar(value="")
        ttk.Label(control_frame, text="Mówca (verify):").grid(row=2, column=0, sticky="w", padx=4, pady=4)
        self.claimed_speaker_combo = ttk.Combobox(
            control_frame,
            textvariable=self.claimed_speaker_var,
            state="readonly",
            width=22,
        )
        self.claimed_speaker_combo.grid(row=2, column=1, sticky="w", padx=4, pady=4)

        load_query_button = ttk.Button(control_frame, text="Wczytaj plik zapytania", command=self.load_query)
        load_query_button.grid(row=2, column=2, padx=4, pady=4)
        recognize_button = ttk.Button(control_frame, text="Rozpoznaj", command=self.recognize_query)
        recognize_button.grid(row=2, column=3, padx=4, pady=4)
        export_button = ttk.Button(control_frame, text="Eksportuj raport", command=self.export_report)
        export_button.grid(row=2, column=4, padx=4, pady=4)

        ttk.Label(control_frame, text="Ramka [ms]:").grid(row=3, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(control_frame, textvariable=self.frame_ms_var, width=8).grid(
            row=3, column=1, sticky="w", padx=4, pady=4
        )
        ttk.Label(control_frame, text="Hop [ms]:").grid(row=3, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(control_frame, textvariable=self.hop_ms_var, width=8).grid(
            row=3, column=3, sticky="w", padx=4, pady=4
        )
        ttk.Label(control_frame, text="N FFT:").grid(row=3, column=4, sticky="w", padx=4, pady=4)
        ttk.Entry(control_frame, textvariable=self.n_fft_var, width=8).grid(
            row=3, column=5, sticky="w", padx=4, pady=4
        )

        ttk.Label(control_frame, text="LPC rzad:").grid(row=4, column=0, sticky="w", padx=4, pady=4)
        ttk.Entry(control_frame, textvariable=self.lpc_order_var, width=8).grid(
            row=4, column=1, sticky="w", padx=4, pady=4
        )
        ttk.Label(control_frame, text="Formanty:").grid(row=4, column=2, sticky="w", padx=4, pady=4)
        ttk.Entry(control_frame, textvariable=self.n_formants_var, width=8).grid(
            row=4, column=3, sticky="w", padx=4, pady=4
        )
        ttk.Button(control_frame, text="Zastosuj parametry", command=self._apply_params).grid(
            row=4, column=4, padx=4, pady=4
        )

        self.status_var = tk.StringVar(value="Gotowy")
        ttk.Label(control_frame, textvariable=self.status_var).grid(row=2, column=6, columnspan=2, sticky="w", padx=4, pady=4)

        for column in range(8):
            control_frame.columnconfigure(column, weight=1 if column == 1 else 0)

        self._action_buttons = [
            load_db_button,
            load_query_button,
            recognize_button,
            export_button,
        ]

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

        self.summary_text = tk.Text(self.notebook, wrap="word")
        self.notebook.add(self.summary_text, text="Podsumowanie")

        self.waveform_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.waveform_frame, text="Przebieg")

        self.mfcc_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.mfcc_frame, text="MFCC")

        self.spectrogram_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.spectrogram_frame, text="Spektrogram")

        self.dtw_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.dtw_frame, text="DTW / Macierz")

        self.feature_var.trace_add("write", self._on_feature_change)

    def set_status(self, text: str) -> None:
        self.status_var.set(text)
        self.root.update_idletasks()

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self._busy = busy
        state = "disabled" if busy else "normal"
        for button in self._action_buttons:
            button.configure(state=state)
        if status is not None:
            self.set_status(status)

    def _run_async(self, action_name: str, worker, on_success) -> None:
        if self._busy:
            return

        self._set_busy(True, f"{action_name}...")

        def job() -> None:
            try:
                result = worker()
            except Exception as exc:
                self.root.after(0, lambda: self._handle_error(action_name, exc))
                return

            self.root.after(0, lambda: self._handle_success(on_success, result))

        threading.Thread(target=job, daemon=True).start()

    def _handle_success(self, callback, result) -> None:
        try:
            callback(result)
        finally:
            self._set_busy(False, "Gotowy")

    def _handle_error(self, action_name: str, exc: Exception) -> None:
        messagebox.showerror("Błąd", f"{action_name} nie powiódł się:\n{exc}")
        self._set_busy(False, "Błąd")

    def _on_feature_change(self, *_args: object) -> None:
        feature_type = self.feature_var.get()
        self.recognizer.set_feature_type(feature_type)
        if self.query_record is not None:
            self._draw_query_mfcc()

    def _apply_params(self) -> None:
        try:
            frame_ms = float(self.frame_ms_var.get())
            hop_ms = float(self.hop_ms_var.get())
            n_fft = int(self.n_fft_var.get())
            lpc_order = int(self.lpc_order_var.get())
            n_formants = int(self.n_formants_var.get())
        except (TypeError, ValueError):
            messagebox.showerror("Błąd", "Nieprawidłowe parametry (sprawdź liczby)")
            return

        if frame_ms <= 0 or hop_ms <= 0 or n_fft <= 0:
            messagebox.showerror("Błąd", "Ramka, hop i N FFT muszą być dodatnie")
            return

        if hop_ms > frame_ms:
            messagebox.showerror("Błąd", "Hop nie może być większy niż ramka")
            return

        if n_fft & (n_fft - 1) != 0:
            messagebox.showerror("Błąd", "N FFT powinno być potęgą 2")
            return

        if lpc_order <= 0 or n_formants <= 0:
            messagebox.showerror("Błąd", "Rząd LPC i liczba formantów muszą być dodatnie")
            return

        if self.query_record is not None:
            signal = load_signal(self.query_record)
            frame_size = milliseconds_to_samples(frame_ms, signal.sample_rate)
            if lpc_order >= frame_size:
                messagebox.showerror(
                    "Błąd",
                    "Rząd LPC musi być mniejszy niż liczba próbek w ramce",
                )
                return

        self.recognizer.mfcc_config.frame_ms = frame_ms
        self.recognizer.mfcc_config.hop_ms = hop_ms
        self.recognizer.mfcc_config.n_fft = n_fft
        self.recognizer.formant_config.frame_ms = frame_ms
        self.recognizer.formant_config.hop_ms = hop_ms
        self.recognizer.formant_config.lpc_order = lpc_order
        self.recognizer.formant_config.n_formants = n_formants
        self.recognizer.clear_cache()

        if self.query_record is not None:
            self._draw_query_signal()
            self._draw_query_mfcc()
            self._draw_spectrogram()

    def load_database(self) -> None:
        try:
            records = discover_recordings(
                self.dataset_var.get(),
                normalized_only=self.normalized_only_var.get(),
            )
            if self.common_words_only_var.get():
                records = filter_records_for_common_words(records)

            if not records:
                raise ValueError("Nie znaleziono plików WAV w bazie")

            self.records = records
            speakers = sorted({record.speaker_label for record in records})
            self.claimed_speaker_combo["values"] = speakers
            if speakers:
                self.claimed_speaker_var.set(speakers[0])

            words = common_vocabulary(records, min_speakers=2)
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(
                "1.0",
                "\n".join(
                    [
                        "BAZA NAGRAŃ",
                        "=" * 72,
                        f"Liczba rekordów: {len(records)}",
                        f"Liczba mówców: {len(speakers)}",
                        f"Liczba słów we wspólnym słowniku: {len(words)}",
                        "",
                        "Mówcy:",
                        *[f"- {speaker}" for speaker in speakers],
                        "",
                        "Słownik:",
                        ", ".join(words),
                    ]
                ),
            )
            self.set_status("Baza wczytana")
        except Exception as exc:
            messagebox.showerror("Błąd", f"Nie udało się wczytać bazy:\n{exc}")
            self.set_status("Błąd bazy")

    def load_query(self) -> None:
        filename = filedialog.askopenfilename(
            title="Wybierz plik WAV",
            filetypes=[("Pliki WAV", "*.wav"), ("Wszystkie pliki", "*.*")],
        )
        if not filename:
            return

        try:
            self.query_record = build_record_from_path(filename)
            self._draw_query_signal()
            self._draw_query_mfcc()
            self._draw_spectrogram()
            self.summary_text.delete("1.0", tk.END)
            self.summary_text.insert(
                "1.0",
                "\n".join(
                    [
                        "PLIK ZAPYTANIA",
                        "=" * 72,
                        f"Plik: {self.query_record.path.name}",
                        f"Mówca: {self.query_record.speaker_label}",
                        f"Słowo: {self.query_record.canonical_word}",
                        f"Powtórzenie: {self.query_record.repetition}",
                    ]
                ),
            )
            self.set_status("Plik zapytania wczytany")
        except Exception as exc:
            messagebox.showerror("Błąd", f"Nie udało się wczytać pliku:\n{exc}")
            self.set_status("Błąd pliku")

    def _build_gallery_for_query(self) -> list[RecordingRecord]:
        if not self.records:
            raise ValueError("Najpierw wczytaj bazę danych")
        if self.query_record is None:
            raise ValueError("Najpierw wczytaj plik zapytania")

        gallery = [record for record in self.records if record.path != self.query_record.path]
        if not gallery:
            raise ValueError("Galeria jest pusta po odjęciu pliku zapytania")
        return gallery

    def recognize_query(self) -> None:
        def worker():
            self.recognizer.set_feature_type(self.feature_var.get())
            gallery = self._build_gallery_for_query()
            task = self.task_var.get()
            if task == "word_recognition":
                return self.recognizer.recognize_word(self.query_record, gallery)
            if task == "speaker_identification":
                return self.recognizer.identify_speaker(self.query_record, gallery)
            return self.recognizer.verify_speaker(
                self.query_record,
                gallery,
                self.claimed_speaker_var.get(),
            )

        def on_success(result: RecognitionResult) -> None:
            self.last_result = result
            self._display_recognition_result(result)
            self._draw_dtw_result(result)
            self.set_status("Rozpoznawanie zakończone")

        self._run_async("Rozpoznawanie", worker, on_success)

    def _clear_frame(self, frame: ttk.Frame) -> None:
        for widget in frame.winfo_children():
            widget.destroy()

    def _attach_figure(self, figure: Figure, frame: ttk.Frame) -> None:
        self._clear_frame(frame)
        canvas = FigureCanvasTkAgg(figure, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _build_waveform_figure(self) -> Figure | None:
        if not MATPLOTLIB_AVAILABLE or self.query_record is None:
            return None
        signal = load_signal(self.query_record)
        figure = Figure(figsize=(12, 4), dpi=100)
        axis = figure.add_subplot(111)
        axis.plot(time_axis(len(signal.samples), signal.sample_rate), signal.samples, linewidth=0.8)
        axis.set_title(f"Przebieg - {self.query_record.path.name}")
        axis.set_xlabel("Czas [s]")
        axis.set_ylabel("Amplituda")
        axis.grid(True, alpha=0.25)
        return figure

    def _build_mfcc_figure(self) -> Figure | None:
        if not MATPLOTLIB_AVAILABLE or self.query_record is None:
            return None
        features = self.recognizer.get_features(self.query_record)
        feature_name = self.recognizer.feature_type
        title_map = {
            "mfcc": "MFCC",
            "fft": "FFT",
            "spectrogram": "Spektrogram",
            "formants": "Formanty",
        }
        figure = Figure(figsize=(12, 5), dpi=100)
        axis = figure.add_subplot(111)
        image = axis.imshow(features.T, aspect="auto", origin="lower", cmap="magma")
        axis.set_title(f"{title_map.get(feature_name, feature_name)} - {self.query_record.path.name}")
        axis.set_xlabel("Ramka")
        axis.set_ylabel("Cecha")
        figure.colorbar(image, ax=axis)
        return figure

    def _build_spectrogram_figure(self) -> Figure | None:
        if not MATPLOTLIB_AVAILABLE or self.query_record is None:
            return None
        signal = load_signal(self.query_record)
        spectrogram = extract_spectrogram(
            signal.samples,
            signal.sample_rate,
            self.recognizer.mfcc_config,
        )
        if spectrogram.size == 0:
            return None

        frame_count = spectrogram.shape[0]
        time_axis_s = frame_time_axis(
            frame_count,
            int(round(self.recognizer.mfcc_config.hop_ms * signal.sample_rate / 1000.0)),
            signal.sample_rate,
        )
        freqs = np.fft.rfftfreq(self.recognizer.mfcc_config.n_fft, d=1.0 / signal.sample_rate)

        figure = Figure(figsize=(12, 5), dpi=100)
        axis = figure.add_subplot(111)
        extent = [time_axis_s[0], time_axis_s[-1], freqs[0], freqs[-1]]
        image = axis.imshow(
            spectrogram.T,
            aspect="auto",
            origin="lower",
            extent=extent,
            cmap="magma",
        )
        axis.set_title(f"Spektrogram - {self.query_record.path.name}")
        axis.set_xlabel("Czas [s]")
        axis.set_ylabel("Częstotliwość [Hz]")
        figure.colorbar(image, ax=axis)
        return figure

    def _build_dtw_figure(self, result: RecognitionResult) -> Figure | None:
        if not MATPLOTLIB_AVAILABLE:
            return None
        figure = Figure(figsize=(12, 5), dpi=100)
        axis = figure.add_subplot(121)
        local = result.best_dtw.local_cost_matrix
        image = axis.imshow(local, origin="lower", aspect="auto", cmap="viridis")
        path_y = [row for row, _ in result.best_dtw.path]
        path_x = [col for _, col in result.best_dtw.path]
        axis.plot(path_x, path_y, color="white", linewidth=1.0)
        axis.set_title("Macierz kosztów lokalnych")
        axis.set_xlabel("Template")
        axis.set_ylabel("Query")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

        axis_2 = figure.add_subplot(122)
        global_cost = result.best_dtw.global_cost_matrix
        image_2 = axis_2.imshow(global_cost, origin="lower", aspect="auto", cmap="plasma")
        axis_2.plot(path_x, path_y, color="white", linewidth=1.0)
        axis_2.set_title("Macierz kosztów globalnych")
        axis_2.set_xlabel("Template")
        axis_2.set_ylabel("Query")
        figure.colorbar(image_2, ax=axis_2, fraction=0.046, pad=0.04)
        return figure

    def _draw_query_signal(self) -> None:
        figure = self._build_waveform_figure()
        if figure is None:
            return
        self._attach_figure(figure, self.waveform_frame)

    def _draw_query_mfcc(self) -> None:
        figure = self._build_mfcc_figure()
        if figure is None:
            return
        self._attach_figure(figure, self.mfcc_frame)

    def _draw_dtw_result(self, result: RecognitionResult) -> None:
        figure = self._build_dtw_figure(result)
        if figure is None:
            return
        self._attach_figure(figure, self.dtw_frame)

    def _draw_spectrogram(self) -> None:
        figure = self._build_spectrogram_figure()
        if figure is None:
            return
        self._attach_figure(figure, self.spectrogram_frame)

    def _display_recognition_result(self, result: RecognitionResult) -> None:
        lines = [
            "WYNIK ROZPOZNAWANIA",
            "=" * 72,
            f"Zadanie: {result.task}",
            f"Cechy: {self.recognizer.feature_type}",
            f"Parametry: {self._format_params()}",
            f"Plik: {result.query.path.name}",
            f"Oczekiwane: {result.expected_label}",
            f"Przewidziane: {result.predicted_label}",
            f"Najlepszy wzorzec: {result.best_template.path.name} ({result.best_template.speaker_label})",
            f"Koszt DTW: {result.best_dtw.normalized_cost:.6f}",
        ]

        if result.task == "speaker_verification":
            lines.extend(
                [
                    f"Deklarowany mówca: {result.claimed_speaker}",
                    f"Akceptacja: {result.accepted}",
                    f"Koszt deklarowanego mówcy: {result.claimed_speaker_score:.6f}",
                    f"Najbliższy impostor: {result.nearest_impostor_score:.6f}",
                ]
            )

        lines.extend(["", "TOP kandydaci:"])
        for candidate in result.candidates:
            if result.task == "word_recognition":
                lines.append(
                    f"- {candidate.label}: {candidate.score:.6f} ({candidate.template.speaker_label})"
                )
            else:
                lines.append(f"- {candidate.label}: {candidate.score:.6f} ({candidate.template.path.name})")

        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "\n".join(lines))

    def export_report(self) -> None:
        if self.last_result is None:
            messagebox.showwarning("Ostrzeżenie", "Najpierw wykonaj rozpoznawanie")
            return

        output_dir = filedialog.askdirectory(title="Wybierz katalog raportu")
        if not output_dir:
            return

        output_path = Path(output_dir)
        speaker_id = self.last_result.query.speaker_id
        file_stem = self.last_result.query.path.stem
        method_tag = self.recognizer.feature_type
        param_tag = self._format_params(compact=True)
        report_dir = output_path / f"speaker_{speaker_id}_{file_stem}_{method_tag}_{param_tag}"
        try:
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = report_dir / "recognition_report.txt"
            export_text(report_path, self.summary_text.get("1.0", tk.END))

            if MATPLOTLIB_AVAILABLE:
                waveform_figure = self._build_waveform_figure()
                if waveform_figure is not None:
                    waveform_figure.savefig(report_dir / "waveform.png", dpi=150, bbox_inches="tight")

                mfcc_figure = self._build_mfcc_figure()
                if mfcc_figure is not None:
                    mfcc_figure.savefig(report_dir / "mfcc.png", dpi=150, bbox_inches="tight")

                spectrogram_figure = self._build_spectrogram_figure()
                if spectrogram_figure is not None:
                    spectrogram_figure.savefig(report_dir / "spectrogram.png", dpi=150, bbox_inches="tight")

                dtw_figure = self._build_dtw_figure(self.last_result)
                if dtw_figure is not None:
                    dtw_figure.savefig(report_dir / "dtw.png", dpi=150, bbox_inches="tight")

            self.set_status("Raport wyeksportowany")
        except Exception as exc:
            messagebox.showerror("Błąd", f"Nie udało się wyeksportować raportu:\n{exc}")
            self.set_status("Błąd eksportu")

    def _format_params(self, *, compact: bool = False) -> str:
        frame_ms = self.recognizer.mfcc_config.frame_ms
        hop_ms = self.recognizer.mfcc_config.hop_ms
        n_fft = self.recognizer.mfcc_config.n_fft
        lpc_order = self.recognizer.formant_config.lpc_order
        n_formants = self.recognizer.formant_config.n_formants
        feature_type = self.recognizer.feature_type

        if feature_type in {"mfcc", "fft", "spectrogram"}:
            if compact:
                return f"fm{frame_ms:g}_hm{hop_ms:g}_fft{n_fft}"
            return f"frame_ms={frame_ms:g}, hop_ms={hop_ms:g}, n_fft={n_fft}"

        if compact:
            return f"fm{frame_ms:g}_hm{hop_ms:g}_lpc{lpc_order}_form{n_formants}"
        return (
            f"frame_ms={frame_ms:g}, hop_ms={hop_ms:g}, lpc_order={lpc_order}, n_formants={n_formants}"
        )


def main() -> None:
    root = tk.Tk()
    Project3GUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
