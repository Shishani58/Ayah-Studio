---
language:
- ar
license: apache-2.0
tags:
- quran
- arabic
- phonetics
- wav2vec2
- tajweed
- forced-alignment
- speech-recognition
datasets:
- Buraaq/quran-md-words
model_type: wav2vec2
library_name: transformers
metrics:
- accuracy
base_model:
- facebook/wav2vec2-base
---

# Quranic Wav2Vec2 Phonetic ASR

## Model Description

**Quranic Wav2Vec2 Phonetic ASR** is a fine-tuned `wav2vec2` model designed specifically for **phonetic transcription of Quranic recitation**.

Unlike standard Arabic ASR systems that output orthographic Arabic text, this model outputs a **phonetic (sound-level) representation**, making it suitable for:

- Tajweed research and education
- Word-level pronunciation analysis
- Forced alignment of Quranic recitation
- Linguistic studies of Classical Arabic phonetics

To the best of our knowledge, this is the **first publicly released wav2vec2 model trained explicitly for Quranic phonetic transcription**.

---

## Key Features

- 🎙️ Outputs **phonetic strings**, not Arabic text
- 📖 Trained on **Quranic recitation**, not conversational Arabic
- 🔤 Custom **phonetic vocabulary + tokenizer**
- 🧠 Compatible with **CTC forced alignment**
- 🧩 Optimized for **word-level Tajweed analysis**

---

## Intended Use

This model is intended for:

- Quranic pronunciation analysis
- Tajweed educational tools
- Phoneme-level alignment
- Academic research on Quranic recitation

### ⚠️ Not intended for:
- Modern Arabic ASR
- End-to-end Tajweed grading of full ayat
- Automatic religious judgments
- Replacement of qualified Quran teachers

---

## Model Architecture

- **Base model:** `facebook/wav2vec2-base`
- **Training objective:** CTC (Connectionist Temporal Classification)
- **Sampling rate:** 16 kHz
- **Input:** Mono audio waveform
- **Output:** Phonetic transcription string
- **Tokenizer:** Custom character-level phonetic tokenizer

---

## Training Dataset

### 📦 Dataset Used
**Buraaq/quran-md-words**

This dataset contains **word-level Quranic recitation audio** with aligned metadata.

Each sample includes:
- Audio of a **single Quranic word**
- Phonetic transcription (`word_tr`)
- Arabic word (`word_ar`)
- Surah and ayah identifiers
- Word index within the ayah

This dataset was chosen intentionally to:
- Preserve **clear phonetic boundaries**
- Avoid coarticulation noise from full-ayah audio
- Enable high-accuracy phonetic learning

> 🔗 Dataset: https://huggingface.co/datasets/Buraaq/quran-md-words

---

## Phonetic Vocabulary & Tokenizer

The model uses a **custom phonetic vocabulary** built directly from the dataset:

- Vocabulary constructed from all unique characters in `word_tr`
- Character-level CTC tokenizer
- Includes:
  - `|` as **word delimiter**
  - `[PAD]` for CTC blank
  - `[UNK]` for unknown symbols

This design allows:
- Robust forced alignment
- Fine-grained phonetic decoding
- Word boundary detection via delimiter token

---

## Training Procedure

### Training Setup

- **Framework:** Hugging Face Transformers + Datasets
- **Platform:** Windows-safe (no multiprocessing crashes)
- **Audio handling:** Hugging Face `Audio` feature
- **Batch size:** 16
- **Epochs:** 20
- **Learning rate:** 2e-5
- **Optimizer:** AdamW (default Trainer)
- **Precision:** FP16 when CUDA available
- **Train / Eval split:** 90% / 10%

The wav2vec2 feature extractor was **not frozen**, allowing adaptation to Quranic recitation acoustics.

---

## Training Results

On the held-out evaluation split:

- **Training accuracy:** **99.7%**
- **Evaluation accuracy:** **99.8%**
- **Test split size:** 10% of the dataset

> ⚠️ Note:  
> These results reflect **word-level phonetic transcription accuracy** on a dataset with consistent recitation style.  
> Performance may degrade on:
> - Fast recitation
> - Strong coarticulation
> - Unseen riwayat styles

---

## Example Usage

### Load Model and Processor

```python
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import torch
import soundfile as sf
import librosa
import numpy as np

processor = Wav2Vec2Processor.from_pretrained(
    "USERNAME/quranic-wav2vec2-phonetic"
)
model = Wav2Vec2ForCTC.from_pretrained(
    "USERNAME/quranic-wav2vec2-phonetic"
)

model.eval()
```

## Phonetic Transcription Example
```
audio, sr = sf.read("recitation.wav")
# convert to mono
if audio.ndim > 1:
    audio = audio.mean(axis=1)

# resample to 16kHz
if sr != 16000:
    audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

inputs = processor(
    audio,
    sampling_rate=16000,
    return_tensors="pt",
    padding=True,
)

with torch.inference_mode():
    logits = model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)

phonetics = processor.batch_decode(
    predicted_ids,
    skip_special_tokens=True
)[0]

print(phonetics)
```

## Example output:

tālik