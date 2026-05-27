# Projekt 3 - rozpoznawanie mowy metodą MFCC + DTW

Projekt realizuje klasyczne rozpoznawanie mowy zgodne z wymaganiami `AiPD_2026_Projekt_3.pdf`.
Rdzeń rozwiązania to:

- własna implementacja `MFCC`,
- własna implementacja `DTW`,
- rozpoznawanie słowa,
- identyfikacja mówcy,
- weryfikacja mówcy.

Projekt działa na bazie `../dzwiek_data` i domyślnie używa nagrań z katalogów znormalizowanych.

## Najważniejsze założenia

- bez użycia gotowych bibliotek, które liczą `MFCC` lub `DTW` za użytkownika,
- cechy są liczone ramkowo: preemfaza, ramkowanie, okno Hamminga, FFT, bank filtrów melowych, logarytm energii i własna `DCT`,
- dopasowanie realizuje klasyczne `DTW` z opcjonalnym ograniczeniem pasma,
- dla identyfikacji i weryfikacji mówcy porównania są wykonywane w pierwszej kolejności na tym samym wypowiedzianym słowie.

## Struktura plików

- [main.py](/C:/Users/2002g/Desktop/laby/aipd/p3/main.py) - aplikacja `tkinter`
- [audio_core.py](/C:/Users/2002g/Desktop/laby/aipd/p3/audio_core.py) - wczytywanie `WAV`, preemfaza, ramkowanie, okna
- [feature_extraction.py](/C:/Users/2002g/Desktop/laby/aipd/p3/feature_extraction.py) - własne `MFCC`
- [dtw.py](/C:/Users/2002g/Desktop/laby/aipd/p3/dtw.py) - własne `DTW`
- [dataset.py](/C:/Users/2002g/Desktop/laby/aipd/p3/dataset.py) - skanowanie bazy i normalizacja etykiet
- [recognizer.py](/C:/Users/2002g/Desktop/laby/aipd/p3/recognizer.py) - logika rozpoznawania
- [evaluation.py](/C:/Users/2002g/Desktop/laby/aipd/p3/evaluation.py) - benchmarki i metryki
- [export_handler.py](/C:/Users/2002g/Desktop/laby/aipd/p3/export_handler.py) - eksport raportów

## Instalacja

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Uruchomienie

```bash
python main.py
```

GUI pozwala wybrać typ cech (`mfcc`, `fft`, `spectrogram`, `formants`) oraz zmieniać parametry
ramkowania i LPC (ramka, hop, N FFT, rząd LPC, liczba formantów).

## CLI

Benchmark słów:

```bash
python cli.py benchmark --task word_recognition --limit 60
```

Benchmark identyfikacji mówcy:

```bash
python cli.py benchmark --task speaker_identification --limit 60
```

Benchmark weryfikacji mówcy:

```bash
python cli.py benchmark --task speaker_verification --limit 60
```

Osobny skrypt benchmarku (pojedyncza metoda lub wszystkie):

```bash
python benchmark.py --task word_recognition --features mfcc
python benchmark.py --task word_recognition --features all
```

Pojedyncze rozpoznanie:

```bash
python cli.py recognize ..\dzwiek_data\speaker_01_m\Znormalizowane\0_2.wav --task word_recognition
```

Opcje cech i parametrów (dla `benchmark` i `recognize`):

```bash
--features {mfcc,fft,spectrogram,formants}
--frame_ms 25.0
--hop_ms 10.0
--n_fft 512
--lpc_order 14
--n_formants 3
```

## Obsługiwane tryby

### 1. Rozpoznawanie słowa

Model wybiera klasę słowa o najmniejszym koszcie `DTW` względem bazy wzorców.

### 2. Identyfikacja mówcy

Model wybiera mówcę o najmniejszym koszcie `DTW`. Gdy to możliwe, porównanie odbywa się tylko
na tym samym słowie co plik zapytania, co tworzy sensowną wersję identyfikacji tekstozależnej.

### 3. Weryfikacja mówcy

Model sprawdza, czy plik należy do deklarowanego mówcy. Decyzja opiera się na porównaniu kosztu
do zadeklarowanego mówcy z kosztem najbliższego impostora.

## Benchmark

Benchmark działa na prostym podziale między powtórzeniami:

- galeria: np. nagrania `_1`,
- zapytania: np. nagrania `_2`.

Pozwala to ocenić działanie systemu bez uczenia sieci neuronowej i bez mieszania pliku zapytania
z jego odpowiednikiem w galerii.

## Ograniczenia

- skuteczność zależy od jakości i spójności nazw plików w bazie,
- dla słów zapisanych różnie u różnych mówców stosowana jest warstwa normalizacji etykiet,
- benchmark weryfikacji używa jednego przypadku pozytywnego i jednego negatywnego na próbkę,
  więc jest to ocena praktyczna, a nie pełna analiza ROC.
