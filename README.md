<div align="center">

# 🕌 surah-cutter

**Точные пер-аятные тайм-коды для аудио Корана — с проверкой на слух.**
**Accurate per-ayah timecodes for Qur'an recitation audio — with audio-visual review.**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Platforms: macOS | Windows](https://img.shields.io/badge/platforms-macOS%20%7C%20Windows-lightgrey.svg)](#)

**[Русский](#русский)** · **[English](#english)**

</div>

---

<a name="русский"></a>
## 🇷🇺 Русский

### Что это

`surah-cutter` берёт папку с аудио Корана (один файл = одна сура, имена `001.mp3`…`114.mp3`)
и строит базу SQLite с тайм-кодом начала каждого аята. Затем прямо в приложении можно
**послушать** результат, увидеть текст аята, перелистывать аяты и **поправить границы**
мышью — правки сохраняются и не затираются при повторной генерации.

Формат БД совместим с существующими наборами тайм-кодов:
`timings(sura, ayah, time)` — где `time` это старт аята в миллисекундах.

### Как это работает (простыми словами)

Это **«чтение пальцем по строчке»**, а не распознавание речи. Текст Корана известен заранее,
поэтому программа не угадывает, что произнесено — она находит, **где во времени** звучит каждое
слово известного текста (это называется *forced alignment*). Старт аята = время его первого слова.
Транскрибации нет.

Чтобы было быстро, используется **гибрид**:
1. Дёшево ищем паузы (чтецы делают паузу — *waqf* — в конце аятов) и прикидываем ожидаемые
   позиции границ по длинам аятов.
2. Где есть чёткая пауза — это и есть граница, нейросеть не нужна (70–90% аятов).
3. Нейросеть (forced alignment) запускаем **только на трудных коротких участках** — где паузы
   нет (связанные аяты, *wasl*) или число пауз не сошлось.

Итог: весь Коран на обычном ноутбуке размечается за **минуты-десятки минут**; для уже известного
чтеца — за секунды (берём готовые данные).

### Возможности

- 📁 Папка или один файл; автоопределение суры из имени; форматы mp3/m4a/ogg/flac/wav.
- ⚡ Гибридный движок (паузы + лёгкая Quran-модель), возобновляемый батч по всем 114 сурам.
- 🎧 **Экран проверки:** плеер, проигрывание/луп отдельного аята, волна с маркерами начала/конца,
  текст Корана по центру с подсветкой, перелистывание аятов.
- ✏️ **Правка границ** перетаскиванием и точной подстройкой; отметка «проверено».
- 💾 **Сохранение:** правки пишутся в рабочую БД и **не затираются** при перегенерации;
  экспорт чистой БД для раздачи.
- 🖥️ macOS и Windows.

### Установка

**Готовые сборки:** см. [Releases](#) (`.dmg` для macOS, `.exe` для Windows).

**Из исходников** (нужен Python ≥3.10):
```bash
git clone <repo-url> surah-cutter
cd surah-cutter
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .            # всё для работы: просмотр, правка И генерация (onnx, без torch)
# pip install -e ".[align]" — только для ре-экспорта int8-модели / torch-оракула в тестах
```
Генерация идёт на лёгком рантайме: int8-модель ONNX (`models/…onnx`, ~120 МБ) +
`onnxruntime`, без torch. Совет: [`uv`](https://github.com/astral-sh/uv) упрощает
установку нужной версии Python — `uv venv --python 3.12 .venv && uv pip install -e .`.

### Использование

**GUI** (лёгкий web-интерфейс в нативном окне — pywebview + локальный сервер, без Qt):
```bash
surah-cutter-gui
```
Укажите папку с аудио и файл проекта `.db` → запустите генерацию (прогресс в реальном
времени) → откройте суру для проверки (плеер, волна с перетаскиваемыми маркерами, текст,
правка границ, отметка «проверено») → экспортируйте чистую `.db`. Старый Qt-GUI доступен
как `surah-cutter-gui-qt` (нужен extra `qt`).

**CLI (батч):**
```bash
surah-cutter /путь/к/аудио -o reciter.db
```

### Точность и проверка

Движок даёт **качественный черновик**, а финальное слово — за человеком. Низкоуверенные
границы подсвечены, чтобы проверять их в первую очередь. Пожалуйста, **проверяйте на слух**
перед публикацией — корректность важнее скорости.

### Сравнение с другими проектами

| Проект | Вход | Подход | Отличие surah-cutter |
|---|---|---|---|
| [cpfair/quran-align](https://github.com/cpfair/quran-align) | по-аятные файлы | Sphinx, модель на чтеца | работаем с **сурой целиком**, без обучения под чтеца |
| [lafzize](https://sr.ht/~rehandaphedar/lafzize/) | аудио + текст | ctc-forced-aligner | + desktop-GUI и **проверка на слух** |
| [QUL Segment Builder](https://qul.tarteel.ai/) | веб (Rails) | машинная разметка + правка | лёгкое **локальное** desktop-приложение |

### Вклад

Issues и PR приветствуются. Перед правкой см. `CLAUDE.md` и `docs/project.md`.

---

<a name="english"></a>
## 🇬🇧 English

### What it is

`surah-cutter` takes a folder of Qur'an audio (one file per surah, named `001.mp3`…`114.mp3`)
and builds a SQLite database with the start timecode of every ayah. You can then **listen** to the
result inside the app, see the ayah text, flip through ayahs, and **adjust the boundaries** by
mouse — your edits are saved and never overwritten on regeneration.

The DB format matches existing timecode sets: `timings(sura, ayah, time)`, where `time` is the
ayah start in milliseconds.

### How it works (in plain words)

It is **"following along with your finger"**, not speech recognition. The Qur'an text is known in
advance, so the program does not guess what is said — it finds **where in time** each word of the
known text is recited (this is *forced alignment*). An ayah's start = the time of its first word.
No transcription.

For speed it uses a **hybrid**:
1. Cheaply detect pauses (reciters pause — *waqf* — at ayah ends) and estimate expected boundary
   positions from ayah lengths.
2. Where there is a clear pause, that *is* the boundary — no neural net needed (70–90% of ayahs).
3. Run the neural forced aligner **only on the hard short windows** — where there is no pause
   (connected ayahs, *wasl*) or the pause count doesn't match.

Result: the whole Qur'an is segmented in **minutes** on a laptop; for an already-known reciter,
in seconds (reuse existing data).

### Features

- 📁 Folder or single file; surah auto-detected from filename; mp3/m4a/ogg/flac/wav.
- ⚡ Hybrid engine (pauses + a small Qur'an model), resumable batch over all 114 surahs.
- 🎧 **Review screen:** player, play/loop a single ayah, waveform with start/end markers,
  centered Qur'an text with highlight, ayah navigation.
- ✏️ **Boundary editing** by drag and fine-nudge; mark as "verified".
- 💾 **Persistence:** edits are stored in the working DB and **not overwritten** on regeneration;
  export a clean DB for distribution.
- 🖥️ macOS and Windows.

### Install

**Prebuilt:** see [Releases](#) (`.dmg` for macOS, `.exe` for Windows).

**From source:**
```bash
git clone <repo-url> surah-cutter
cd surah-cutter
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e .
```

### Usage

**GUI:** `surah-cutter-gui` — pick the audio folder, let it generate, open the review screen,
adjust if needed, export the `.db`.

**CLI:** `surah-cutter /path/to/audio -o reciter.db`

### Accuracy & review

The engine produces a **high-quality draft**; the final say is yours. Low-confidence boundaries are
highlighted so you review them first. Please **verify by ear** before publishing — correctness
matters more than speed.

### Contributing

Issues and PRs welcome. Read `CLAUDE.md` and `docs/project.md` before editing.

---

## 🙏 Благодарности / Acknowledgements

- **Qur'an text** — King Fahd Glorious Qur'an Printing Complex (KFGQPC) scripts (`qpc_hafs`).
- **Reciter timecodes** (`data/references/`) — community datasets
  ([quran.com](https://quran.com), [Tarteel QUL](https://qul.tarteel.ai/),
  [quran-align](https://github.com/cpfair/quran-align)), typically CC-BY 4.0.
- **Forced alignment** — [ctc-forced-aligner](https://github.com/MahmoudAshraf97/ctc-forced-aligner),
  [ctc-segmentation](https://github.com/lumaku/ctc-segmentation), and prior art from
  [lafzize](https://sr.ht/~rehandaphedar/lafzize/).
- **Acoustic models** — Hugging Face community (e.g. `TBOGamer22/wav2vec2-quran-phonetics`,
  MMS forced aligner), Silero VAD.
- **Font** — KFGQPC HAFS Uthmanic Script.

Pull in upstream attributions when redistributing. See [LICENSE](LICENSE).

## 📜 Лицензия / License

[MIT](LICENSE) для кода. Вендоренные данные, шрифт и модели сохраняют лицензии источников —
см. примечание в [LICENSE](LICENSE).
