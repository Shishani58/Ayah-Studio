"""Дешёвая стадия кандидатов: energy-паузы + структурный приор.

Идея гибридного режима: до запуска тяжёлой нейросетевой выравнивалки получить
дешёвые подсказки о том, где проходят границы аятов.

Два источника:

1. **Паузы речи** — RMS/energy detector ищет низкоэнергетические сегменты.
   Границы между аятами почти всегда лежат в паузе; «глубина» паузы (её
   длительность) служит уверенностью кандидата.
2. **Структурный приор** — ожидаемые относительные позиции границ аятов,
   выведенные из ширины рендера слов (``qpc_v4_render_width_millis`` /
   ``qpc_v1_render_width_millis`` в ``data/quran_mushaf.db``). Это пропорция, а
   не абсолютное время: чем длиннее слова аята, тем дольше он звучит. Приор
   нормирован в долю ``[0, 1]`` от полной длительности суры.

Сопоставив паузы с ожидаемыми позициями, для каждой границы аята получаем
кандидата: либо чёткая пауза рядом с ожиданием (берём напрямую, без НС), либо
паузы нет/неоднозначна — аят «трудный», его отдаём выравнивалке.

Детектор пауз не грузит ONNX/VAD-модели: только ``numpy`` поверх уже
декодированного PCM. Чтение БД — чистый ``sqlite3`` локально, чтобы не трогать
``data.quran_text``.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from surah_cutter.data.quran_text import DEFAULT_DB_PATH

# Модель Silero VAD (ONNX). Кладём под models/ репозитория.
# models/ -> <repo>; candidates.py -> core -> surah_cutter -> <repo>.
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
SILERO_VAD_PATH = MODELS_DIR / "silero_vad.onnx"

# Канонический источник весов Silero VAD (ONNX) — официальный репозиторий.
# Скачивается ЛЕНИВО и только при allow_download=True (по умолчанию выключено).
SILERO_VAD_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/"
    "src/silero_vad/data/silero_vad.onnx"
)

# Silero VAD работает на 16 kHz; размер окна инференса — 512 сэмплов (32 мс).
VAD_SR = 16_000
VAD_WINDOW = 512

# Silero VAD v5 (текущий экспорт) ждёт на вход НЕ голые 512 сэмплов, а окно с
# «контекстом»: 64 последних сэмпла предыдущего окна + текущие 512 = 576 сэмплов.
# Без этого контекста сеть на КАЖДОМ окне отдаёт ~0.001 (всё считает тишиной).
# Контекст первого окна — нули. (Контракт скопирован из silero-vad/OnnxWrapper,
# но без torch — на голом onnxruntime.)
VAD_CONTEXT = 64

# Порог вероятности речи: кадр выше — речь, ниже — тишина.
SPEECH_THRESHOLD = 0.5

# Паузу короче этого (мс) не считаем кандидатом границы — это микропаузы внутри
# аята / между словами, а не межаятная цезура.
MIN_PAUSE_MS = 300.0

# Second-pass pause detector: the strict pass intentionally favours precision,
# but weak inter-ayah pauses can be shorter and less contrasted on melodic
# recitation. These candidates still pass a relaxed DSP gate and the fast DP can
# skip extras acoustically.
WEAK_MIN_PAUSE_MS = 200.0

# Допуск сопоставления (доля длительности суры): пауза дальше этого от ожидаемой
# позиции границы не считается «чёткой» для этой границы.
MATCH_TOLERANCE_FRAC = 0.04


@dataclass(frozen=True)
class Pause:
    """Сегмент тишины между двумя сегментами речи.

    Attributes:
        start_ms: Начало паузы (мс от начала аудио).
        end_ms: Конец паузы (мс).
        depth_ms: Длительность паузы (``end_ms - start_ms``) — мера «глубины».
    """

    start_ms: float
    end_ms: float
    depth_ms: float

    @property
    def center_ms(self) -> float:
        """Центр паузы — точка-кандидат границы."""
        return (self.start_ms + self.end_ms) / 2.0


@dataclass(frozen=True)
class PauseDecision:
    """Debug record for the pause pipeline."""

    pause: Pause
    dsp_ok: bool
    vad_ok: bool | None
    accepted: bool
    reason: str
    flatness: float
    zcr: float
    rms_ratio: float


@dataclass(frozen=True)
class BoundaryCandidate:
    """Кандидат границы начала аята.

    Attributes:
        ayah: Номер аята, *начало* которого описывает эта граница.
        expected_ms: Ожидаемая позиция начала (мс) по структурному приору.
        time_ms: Итоговая позиция кандидата (мс) — центр сопоставленной паузы,
            либо ``expected_ms``, если чёткой паузы не нашлось.
        confidence: Уверенность ``[0, 1]``. >0 только если рядом с ожиданием
            нашлась достаточно глубокая пауза; иначе ``0.0`` (аят «трудный»).
        from_pause: ``True``, если кандидат опёрт на реальную energy-паузу.
    """

    ayah: int
    expected_ms: float
    time_ms: float
    confidence: float
    from_pause: bool


# --- структурный приор ------------------------------------------------------


def ayah_render_widths(
    surah: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    variant: str = "v4",
) -> dict[int, int]:
    """Суммарная ширина рендера слов на аят (мс рендера, не звука).

    Локальное read-only чтение ``data/quran_mushaf.db`` (без ``data.quran_text``,
    чтобы не расширять его API). Складывает ``qpc_<variant>_render_width_millis``
    по словам каждого аята, пропуская строки-разделители (пустой ``qpc_hafs``).

    Returns:
        ``{ayah: total_render_width}`` в порядке возрастания аята.

    Raises:
        ValueError: при неизвестном ``variant``.
        FileNotFoundError: если БД не найдена.
    """
    if variant not in ("v4", "v1"):
        raise ValueError(f"unknown render-width variant {variant!r}; use 'v4' or 'v1'")
    col = f"qpc_{variant}_render_width_millis"

    path = Path(db_path)
    if not path.exists():
        raise FileNotFoundError(f"Quran DB not found: {path}")
    uri = f"file:{path.as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    try:
        rows = conn.execute(
            f"SELECT ayah, COALESCE({col}, 0) FROM words "
            "WHERE surah = ? AND TRIM(qpc_hafs) != '' ORDER BY id",
            (surah,),
        ).fetchall()
    finally:
        conn.close()

    widths: dict[int, int] = {}
    for ayah, w in rows:
        widths[int(ayah)] = widths.get(int(ayah), 0) + int(w)
    return widths


def expected_boundary_fractions(
    surah: int,
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    variant: str = "v4",
) -> dict[int, float]:
    """Ожидаемая относительная позиция *начала* каждого аята в ``[0, 1)``.

    Аят 1 всегда начинается в 0.0. Начало аята ``k`` — накопленная доля ширины
    предыдущих аятов от суммарной ширины суры. Это пропорция, переносимая на
    реальную длительность аудио умножением на неё.

    Returns:
        ``{ayah: fraction}`` для всех аятов суры, ``fraction`` неубывающая.
    """
    widths = ayah_render_widths(surah, db_path=db_path, variant=variant)
    ayahs = sorted(widths)
    total = sum(widths.values())
    fractions: dict[int, float] = {}
    acc = 0
    for ayah in ayahs:
        fractions[ayah] = (acc / total) if total > 0 else 0.0
        acc += widths[ayah]
    return fractions


# --- energy-based pauses -----------------------------------------------------


def vad_available(model_path: str | Path = SILERO_VAD_PATH) -> bool:
    """Есть ли локальный файл модели Silero VAD."""
    return Path(model_path).exists()


def ensure_vad_model(
    model_path: str | Path = SILERO_VAD_PATH,
    *,
    allow_download: bool = False,
) -> Path:
    """Гарантировать наличие модели VAD, при необходимости скачав её.

    Скачивание происходит ТОЛЬКО при ``allow_download=True``. По умолчанию, если
    файла нет, бросает ``FileNotFoundError`` с подсказкой — мы не тянем сеть в
    обычном прогоне.

    Returns:
        Путь к существующему файлу модели.
    """
    path = Path(model_path)
    if path.exists():
        return path
    if not allow_download:
        raise FileNotFoundError(
            f"Silero VAD model not found at {path}. "
            f"Download it from {SILERO_VAD_URL} (or call with allow_download=True)."
        )
    # Ленивая загрузка: импорт сети только когда реально качаем.
    import urllib.request

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    urllib.request.urlretrieve(SILERO_VAD_URL, tmp)  # noqa: S310 - фикс. URL
    tmp.replace(path)
    return path


def detect_pauses(
    pcm,
    sr: int,
    *,
    model_path: str | Path = SILERO_VAD_PATH,
    allow_download: bool = False,
    min_pause_ms: float = MIN_PAUSE_MS,
    threshold: float = SPEECH_THRESHOLD,
) -> list[Pause]:
    """Найти паузы по энергии сигнала, без Silero/ONNX VAD.

    Совместимая замена старого VAD-пути: ``model_path``/``allow_download`` и
    ``threshold`` оставлены в сигнатуре для существующих вызовов, но больше не
    используются. Для pause-backed старта аята канонична середина тишины
    (``pause.center_ms``): ``pause.end_ms`` — правый край паузы и может дать
    резкий старт на атаке следующей фразы без отдельного акустического refine.
    """
    del model_path, allow_download, threshold

    return detect_energy_pauses(pcm, sr, min_pause_ms=min_pause_ms)


def detect_combined_pauses(
    pcm,
    sr: int,
    *,
    surah: int | None = None,
    min_pause_ms: float = MIN_PAUSE_MS,
) -> list[Pause]:
    """Combine energy, TEN, and Silero pause detectors for fast alignment.

    Policy:
    - energy remains the main detector because it matches visible waveform
      pauses best;
    - for the first intro-basmala pause (all suras except 1 and 9), run TEN and
      Silero only around that local region and prefer the tighter neural pause
      when it overlaps the broad energy pause;
    - do not add mid-sura VAD/TEN-only pauses: on recitation they overproduce
      breath/decay pauses, while the waveform detector is better there.

    Al-Fatiha's basmala is ayah 1, so it is not treated as an intro. At-Tawba has
    no basmala.
    """
    return [
        d.pause
        for d in pause_pipeline_decisions(pcm, sr, surah=surah, min_pause_ms=min_pause_ms)
        if d.accepted
    ]


def pause_pipeline_decisions(
    pcm,
    sr: int,
    *,
    surah: int | None = None,
    min_pause_ms: float = MIN_PAUSE_MS,
) -> list[PauseDecision]:
    """Run the production pause pipeline and return per-candidate decisions."""
    energy = _pause_recall_candidates(pcm, sr, min_pause_ms=min_pause_ms)
    if not energy:
        return []

    decisions: list[PauseDecision] = []
    intro_basmala = surah not in (None, 1, 9)
    for idx, (p, weak) in enumerate(energy):
        dsp = _dsp_pause_metrics(pcm, sr, p)
        dsp_ok = _dsp_accepts_pause(dsp) or (weak and _dsp_accepts_weak_pause(dsp, p))
        vad_ok: bool | None = None
        reason = "weak-dsp" if weak else "dsp"
        pause = p

        needs_vad = idx == 0 and intro_basmala
        if needs_vad:
            # The first content pause after the recited intro-basmala is often
            # easier for neural VADs. Use them to confirm the candidate, but keep
            # the final mark at the energy midpoint unless a tighter VAD pause
            # almost exactly overlaps the broad energy shelf.
            ten, vad = _local_vad_pauses(pcm, sr, p, min_pause_ms=min_pause_ms)
            tighter = _best_overlapping(p, ten + vad)
            vad_ok = _vad_confirms_pause(p, ten, vad)
            if tighter is not None:
                pause = Pause(tighter.start_ms, tighter.end_ms, p.depth_ms)
                reason = "intro-vad-tight"
            else:
                reason = "intro-vad"

        accepted = dsp_ok and (vad_ok is not False)
        decisions.append(
            PauseDecision(
                pause=pause,
                dsp_ok=dsp_ok,
                vad_ok=vad_ok,
                accepted=accepted,
                reason=reason if accepted else "rejected",
                flatness=dsp["flatness"],
                zcr=dsp["zcr"],
                rms_ratio=dsp["rms_ratio"],
            )
        )
    return decisions


def _pause_recall_candidates(
    pcm,
    sr: int,
    *,
    min_pause_ms: float,
) -> list[tuple[Pause, bool]]:
    """Strict pauses plus non-overlapping weak candidates.

    The strict pass stays unchanged. The weak pass lowers duration and contrast
    requirements to recover quiet inter-ayah pauses; overlapping strict pauses
    win so their timing and reason remain stable.
    """
    strict = detect_energy_pauses(pcm, sr, min_pause_ms=min_pause_ms, use_ten_vad=False)
    if min_pause_ms > MIN_PAUSE_MS:
        return [(p, False) for p in strict]

    weak_min = min(min_pause_ms, WEAK_MIN_PAUSE_MS)
    weak = detect_energy_pauses(
        pcm,
        sr,
        min_pause_ms=weak_min,
        smooth_frames=4,
        merge_gap_ms=180.0,
        side_ratio=1.65,
        one_side_ratio=2.15,
        flat_ratio=3.0,
        long_pause_ms=500.0,
        deep_long_pause_ms=900.0,
        use_ten_vad=True,
    )

    out: list[tuple[Pause, bool]] = [(p, False) for p in strict]
    for p in weak:
        if _best_overlapping(p, strict) is None:
            out.append((p, True))
    out.sort(key=lambda item: (item[0].start_ms, item[0].end_ms))
    return out


def _local_vad_pauses(
    pcm,
    sr: int,
    pause: Pause,
    *,
    min_pause_ms: float,
) -> tuple[list[Pause], list[Pause]]:
    s0_ms = max(0.0, pause.start_ms - 1200.0)
    s1_ms = min(len(pcm) * 1000.0 / sr, pause.end_ms + 1200.0)
    a = int(round(s0_ms * sr / 1000.0))
    b = int(round(s1_ms * sr / 1000.0))
    clip = pcm[a:b]
    ten = _offset_pauses(
        _optional_detector(detect_ten_pauses, clip, sr, min_pause_ms=min_pause_ms),
        s0_ms,
    )
    vad = _offset_pauses(
        _optional_detector(detect_silero_pauses, clip, sr, min_pause_ms=min_pause_ms),
        s0_ms,
    )
    return ten, vad


def _vad_confirms_pause(pause: Pause, ten: list[Pause], vad: list[Pause]) -> bool:
    if not ten and not vad:
        return True
    return _best_overlapping(pause, ten + vad) is not None


def _dsp_pause_metrics(pcm, sr: int, pause: Pause) -> dict[str, float]:
    """Cheap DSP features for an energy pause shelf."""
    import numpy as np

    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    # The energy pass works on smoothed frames, so candidate edges often include
    # a bit of speech/decay. Measure the middle shelf; keep the original pause
    # bounds for timing.
    trim_ms = min(80.0, max(0.0, pause.depth_ms * 0.22))
    start_ms = pause.start_ms + trim_ms
    end_ms = pause.end_ms - trim_ms
    if end_ms <= start_ms:
        start_ms, end_ms = pause.start_ms, pause.end_ms
    a = max(0, int(round(start_ms * sr / 1000.0)))
    b = min(audio.size, int(round(end_ms * sr / 1000.0)))
    seg = audio[a:b]
    if seg.size < max(32, int(0.04 * sr)):
        return {"flatness": 0.0, "zcr": 0.0, "rms_ratio": 1.0}

    seg = seg - float(np.mean(seg))
    eps = 1e-10
    rms = float(np.sqrt(np.mean(seg * seg)))
    all_rms = float(np.sqrt(np.mean(audio * audio))) if audio.size else eps
    rms_ratio = rms / max(all_rms, eps)

    max_fft = min(seg.size, 2048)
    n_fft = 1
    while n_fft * 2 <= max_fft:
        n_fft *= 2
    frame = seg[:n_fft] * np.hanning(n_fft)
    mag = np.abs(np.fft.rfft(frame)) + eps
    flatness = float(np.exp(np.mean(np.log(mag))) / np.mean(mag))

    signs = np.signbit(seg)
    zcr = float(np.mean(signs[1:] != signs[:-1])) if seg.size > 1 else 0.0
    return {"flatness": flatness, "zcr": zcr, "rms_ratio": rms_ratio}


def _dsp_accepts_pause(metrics: dict[str, float]) -> bool:
    """Accept low-energy noise/silence shelves; reject tonal recitation tails."""
    rms_ratio = metrics["rms_ratio"]
    flatness = metrics["flatness"]
    zcr = metrics["zcr"]
    if rms_ratio <= 0.10:
        return True
    if rms_ratio <= 0.30 and flatness >= 0.12:
        return True
    if rms_ratio <= 0.34 and zcr >= 0.045:
        return True
    return False


def _dsp_accepts_weak_pause(metrics: dict[str, float], pause: Pause) -> bool:
    """Relaxed gate for second-pass pause candidates.

    It still rejects loud tonal tails, but allows weaker low-energy shelves that
    fail the strict two-sided contrast filter.
    """
    rms_ratio = metrics["rms_ratio"]
    flatness = metrics["flatness"]
    zcr = metrics["zcr"]
    if rms_ratio <= 0.16:
        return True
    if pause.depth_ms >= 260.0 and rms_ratio <= 0.24 and flatness >= 0.08:
        return True
    if pause.depth_ms >= 360.0 and rms_ratio <= 0.38 and flatness >= 0.12:
        return True
    if pause.depth_ms >= 360.0 and rms_ratio <= 0.40 and zcr >= 0.035:
        return True
    return False


def _optional_detector(fn, pcm, sr: int, *, min_pause_ms: float) -> list[Pause]:
    try:
        return fn(pcm, sr, min_pause_ms=min_pause_ms)
    except RuntimeError:
        return []


def _offset_pauses(pauses: list[Pause], offset_ms: float) -> list[Pause]:
    return [
        Pause(p.start_ms + offset_ms, p.end_ms + offset_ms, p.depth_ms)
        for p in pauses
    ]


def _best_overlapping(target: Pause, pauses: list[Pause]) -> Pause | None:
    best: Pause | None = None
    best_overlap = 0.0
    for p in pauses:
        overlap = min(target.end_ms, p.end_ms) - max(target.start_ms, p.start_ms)
        if overlap <= 0:
            continue
        denom = max(1.0, min(target.depth_ms, p.depth_ms))
        score = overlap / denom
        if score >= 0.35 and score > best_overlap:
            best = p
            best_overlap = score
    return best


def detect_energy_pauses(
    pcm,
    sr: int,
    *,
    min_pause_ms: float = MIN_PAUSE_MS,
    frame_ms: float = 20.0,
    smooth_frames: int = 5,
    noise_percentile: float = 15.0,
    speech_percentile: float = 70.0,
    merge_gap_ms: float = 80.0,
    side_ms: float = 360.0,
    side_ratio: float = 2.2,
    one_side_ratio: float = 3.0,
    flat_ratio: float = 2.4,
    long_pause_ms: float = 700.0,
    deep_long_pause_ms: float = 1200.0,
    use_ten_vad: bool = True,
    ten_threshold: float = 0.5,
) -> list[Pause]:
    """Найти межречевые паузы по RMS-энергии.

    Алгоритм:
    1. RMS по окнам ``frame_ms``.
    2. Сглаживание скользящим средним.
    3. Адаптивный порог между noise floor и типичной речевой энергией.
    4. Длинные low-energy участки строго между речевыми участками.
    5. Контрастная проверка: внутри — ровная низкая площадка, по обе стороны —
       заметно более высокая волна. Это отсекает короткие слабые провалы внутри
       похожей по уровню речи.
    """
    import numpy as np

    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if audio.size == 0 or sr <= 0:
        return []

    frame = max(1, int(round(sr * frame_ms / 1000.0)))
    n_frames = audio.size // frame
    if n_frames < 3:
        return []

    trimmed = audio[: n_frames * frame].reshape(n_frames, frame)
    raw_rms = np.sqrt(np.mean(trimmed * trimmed, axis=1))
    rms = raw_rms
    if smooth_frames > 1:
        kernel = np.ones(int(smooth_frames), dtype=np.float32) / float(smooth_frames)
        rms = np.convolve(rms, kernel, mode="same")

    nz = rms[rms > 0]
    if nz.size == 0:
        return []
    noise = float(np.percentile(nz, noise_percentile))
    speech = float(np.percentile(nz, speech_percentile))
    if speech <= noise:
        cutoff = noise * 1.8
    else:
        cutoff = noise + (speech - noise) * 0.18
    cutoff = max(cutoff, 1e-5)

    speech_flags = (rms > cutoff).tolist()
    pauses = _flags_to_pauses(
        speech_flags,
        sr,
        min_pause_ms,
        window_samples=frame,
    )
    pauses = _merge_close_pauses(pauses, merge_gap_ms)
    pauses = _filter_contrasted_pauses(
        pauses,
        rms,
        pcm=audio,
        sr=sr,
        frame_ms=frame * 1000.0 / sr,
        cutoff=cutoff,
        side_ms=side_ms,
        side_ratio=side_ratio,
        one_side_ratio=one_side_ratio,
        flat_ratio=flat_ratio,
        long_pause_ms=long_pause_ms,
        deep_long_pause_ms=deep_long_pause_ms,
        use_ten_vad=use_ten_vad,
        ten_threshold=ten_threshold,
    )
    return _refine_pause_edges(
        pauses,
        raw_rms,
        frame_ms=frame * 1000.0 / sr,
        cutoff=cutoff,
        min_pause_ms=min_pause_ms,
        pad_frames=max(1, int(smooth_frames)),
    )


_TEN_HOP_SIZE = 256


def detect_silero_pauses(
    pcm,
    sr: int,
    *,
    model_path: str | Path = SILERO_VAD_PATH,
    allow_download: bool = False,
    min_pause_ms: float = MIN_PAUSE_MS,
    merge_gap_ms: float = 180.0,
    threshold: float = SPEECH_THRESHOLD,
) -> list[Pause]:
    """Find Silero VAD non-speech runs.

    This is a debug/compare overlay, not the production energy detector. Silero
    v5 ONNX needs recurrent state plus 64 samples of previous-window context.
    """
    if sr != VAD_SR:
        raise RuntimeError(f"Silero VAD expects {VAD_SR} Hz PCM, got {sr}")
    if not Path(model_path).exists() and not allow_download:
        raise RuntimeError(f"Silero VAD model not found: {model_path}")

    import numpy as np
    import onnxruntime as ort

    path = ensure_vad_model(model_path, allow_download=allow_download)
    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if audio.size < VAD_WINDOW * 3:
        return []

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    input_names = {i.name for i in sess.get_inputs()}
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr_arr = np.array(sr, dtype=np.int64)
    context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)

    n_windows = audio.size // VAD_WINDOW
    flags: list[bool] = []
    for i in range(n_windows):
        chunk = audio[i * VAD_WINDOW:(i + 1) * VAD_WINDOW].reshape(1, -1)
        inp = np.concatenate([context, chunk], axis=1)
        feeds = {"input": inp, "state": state, "sr": sr_arr}
        feeds = {k: v for k, v in feeds.items() if k in input_names}
        out = sess.run(None, feeds)
        prob = float(np.asarray(out[0]).reshape(-1)[0])
        if len(out) > 1:
            state = np.asarray(out[1], dtype=np.float32)
        context = chunk[:, -VAD_CONTEXT:]
        flags.append(prob >= threshold)

    raw_min_ms = min(80.0, min_pause_ms)
    pauses = _flags_to_pauses(flags, sr, raw_min_ms, window_samples=VAD_WINDOW)
    pauses = _merge_close_pauses(pauses, merge_gap_ms)
    return [p for p in pauses if p.depth_ms >= min_pause_ms]


def detect_ten_pauses(
    pcm,
    sr: int,
    *,
    start_ms: float = 0.0,
    end_ms: float | None = None,
    min_pause_ms: float = MIN_PAUSE_MS,
    merge_gap_ms: float = 180.0,
    threshold: float = 0.5,
) -> list[Pause]:
    """Find TEN VAD non-speech runs in a local audio range.

    This is intentionally separate from the production energy detector and is
    mainly for visual comparison/debugging. The Python TEN binding is frame-by-
    frame, so callers should pass a visible time range for long suras.
    """
    if sr != VAD_SR:
        raise RuntimeError(f"TEN VAD expects {VAD_SR} Hz PCM, got {sr}")
    try:
        from ten_vad import TenVad  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "TEN VAD is not installed. Install optional extra: surah-cutter[ten-vad]"
        ) from exc

    import numpy as np

    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    total_ms = audio.size * 1000.0 / sr
    a_ms = max(0.0, start_ms)
    b_ms = total_ms if end_ms is None else min(total_ms, max(a_ms, end_ms))
    s0 = int(round(a_ms * sr / 1000.0))
    s1 = int(round(b_ms * sr / 1000.0))
    if s1 - s0 < _TEN_HOP_SIZE * 3:
        return []

    try:
        vad = TenVad(_TEN_HOP_SIZE, threshold)
    except Exception as exc:
        raise RuntimeError(f"TEN VAD failed to initialize: {exc}") from exc

    chunk = np.clip(audio[s0:s1], -1.0, 1.0)
    int16 = (chunk * 32767.0).astype(np.int16)
    n = int16.size // _TEN_HOP_SIZE
    flags: list[bool] = []
    try:
        for i in range(n):
            frame = int16[i * _TEN_HOP_SIZE:(i + 1) * _TEN_HOP_SIZE]
            _prob, flag = vad.process(frame)
            flags.append(bool(flag))
    except Exception as exc:
        raise RuntimeError(f"TEN VAD failed while processing audio: {exc}") from exc

    raw_min_ms = min(80.0, min_pause_ms)
    local = _flags_to_pauses(flags, sr, raw_min_ms, window_samples=_TEN_HOP_SIZE)
    local = _merge_close_pauses(local, merge_gap_ms)
    local = [p for p in local if p.depth_ms >= min_pause_ms]
    return [
        Pause(p.start_ms + a_ms, p.end_ms + a_ms, p.depth_ms)
        for p in local
    ]


def _filter_contrasted_pauses(
    pauses: list[Pause],
    rms,
    *,
    pcm,
    sr: int,
    frame_ms: float,
    cutoff: float,
    side_ms: float,
    side_ratio: float,
    one_side_ratio: float,
    flat_ratio: float,
    long_pause_ms: float,
    deep_long_pause_ms: float,
    use_ten_vad: bool,
    ten_threshold: float,
) -> list[Pause]:
    """Keep pauses that are low/flat and bordered by stronger wave energy."""
    if not pauses:
        return []

    import numpy as np

    side_frames = max(1, int(round(side_ms / frame_ms)))
    out: list[Pause] = []
    for p in pauses:
        a = max(0, int(round(p.start_ms / frame_ms)))
        b = min(len(rms), int(round(p.end_ms / frame_ms)))
        if b <= a:
            continue

        low = rms[a:b]
        left = rms[max(0, a - side_frames):a]
        right = rms[b:min(len(rms), b + side_frames)]
        if left.size == 0 or right.size == 0:
            continue

        low_med = float(np.median(low))
        low_p80 = float(np.percentile(low, 80))
        left_p70 = float(np.percentile(left, 70))
        right_p70 = float(np.percentile(right, 70))

        # "Ровная низкая волна": внутри нет сильных всплесков выше порога.
        flat_enough = low_p80 <= max(cutoff, low_med * flat_ratio)
        # "По бокам больше волны": обе стороны контрастно выше паузы.
        floor = max(low_p80, cutoff * 0.35, 1e-6)
        both_sides = (
            left_p70 >= floor * side_ratio
            and right_p70 >= floor * side_ratio
            and left_p70 >= cutoff
            and right_p70 >= cutoff
        )
        # Длинная ровная площадка может иметь затухание слева и сильный вход
        # справа (или наоборот). Не требуем обе стороны высокими, если одна
        # сторона контрастная, а вторая всё же выше самой площадки.
        one_side = (
            p.depth_ms >= long_pause_ms
            and (
                left_p70 >= floor * one_side_ratio
                or right_p70 >= floor * one_side_ratio
            )
            and left_p70 >= floor * 1.15
            and right_p70 >= floor * 1.15
        )
        # После басмалы часто бывает длинная ровная низкая площадка: хвост слева
        # уже затух, вход справа ещё не резкий, поэтому обычный контраст мягче.
        deep_long = (
            p.depth_ms >= deep_long_pause_ms
            and low_med <= cutoff * 0.65
            and low_p80 <= cutoff * 0.85
            and max(left_p70, right_p70) >= floor * 1.8
            and min(left_p70, right_p70) >= floor * 1.25
        )
        energy_accept = both_sides or one_side or deep_long
        # TEN VAD is a fallback for borderline energy candidates only. It must
        # not veto strong acoustic pauses: in recitation, TEN can mark long
        # vowel decay inside a real pause as speech.
        ten_rescue = False
        if not energy_accept and use_ten_vad and flat_enough:
            ten_rescue = (
                p.depth_ms >= long_pause_ms
                and low_p80 <= cutoff
                and max(left_p70, right_p70) >= floor * 1.45
                and _ten_confirms_pause(
                    pcm,
                    sr,
                    p,
                    side_ms=side_ms,
                    threshold=ten_threshold,
                )
            )
        if flat_enough and (energy_accept or ten_rescue):
            out.append(p)
    return out


def _refine_pause_edges(
    pauses: list[Pause],
    raw_rms,
    *,
    frame_ms: float,
    cutoff: float,
    min_pause_ms: float,
    pad_frames: int,
) -> list[Pause]:
    """Snap smoothed pause candidates back to the raw low-energy shelf."""
    if not pauses:
        return []

    import numpy as np

    rms = np.asarray(raw_rms, dtype=np.float32).reshape(-1)
    out: list[Pause] = []
    for p in pauses:
        a = max(0, int(round(p.start_ms / frame_ms)) - pad_frames)
        b = min(len(rms), int(round(p.end_ms / frame_ms)) + pad_frames)
        if b <= a:
            continue

        region = rms[a:b]
        low_ref = float(np.percentile(region, 20))
        edge_cutoff = min(
            cutoff * 0.85,
            max(cutoff * 0.25, low_ref * 1.8, 1e-7),
        )
        low = region <= edge_cutoff
        best: tuple[int, int] | None = None
        run_start: int | None = None
        for i, is_low in enumerate(low.tolist() + [False]):
            if is_low:
                if run_start is None:
                    run_start = i
            elif run_start is not None:
                run_end = i
                if best is None or run_end - run_start > best[1] - best[0]:
                    best = (run_start, run_end)
                run_start = None

        if best is None:
            continue
        start_ms = (a + best[0]) * frame_ms
        end_ms = (a + best[1]) * frame_ms
        depth_ms = end_ms - start_ms
        if depth_ms >= min_pause_ms:
            out.append(Pause(start_ms, end_ms, depth_ms))
    return out


def _ten_confirms_pause(
    pcm,
    sr: int,
    pause: Pause,
    *,
    side_ms: float,
    threshold: float,
) -> bool:
    """Whether TEN VAD sees little speech inside and speech near this pause.

    This is intentionally local: running TEN over all of Al-Baqara through the
    Python binding is too slow. We only call it for borderline energy candidates.
    """
    if sr != VAD_SR:
        return False
    try:
        from ten_vad import TenVad  # type: ignore[import-not-found]
    except Exception:
        return False

    import numpy as np

    frame_ms = _TEN_HOP_SIZE * 1000.0 / sr
    clip_start_ms = max(0.0, pause.start_ms - side_ms)
    clip_end_ms = min(len(pcm) * 1000.0 / sr, pause.end_ms + side_ms)
    s0 = int(round(clip_start_ms * sr / 1000.0))
    s1 = int(round(clip_end_ms * sr / 1000.0))
    if s1 - s0 < _TEN_HOP_SIZE * 3:
        return False

    try:
        vad = TenVad(_TEN_HOP_SIZE, threshold)
    except Exception:
        return False

    audio = np.asarray(pcm[s0:s1], dtype=np.float32).reshape(-1)
    int16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    n = int16.size // _TEN_HOP_SIZE
    if n < 3:
        return False
    flags: list[bool] = []
    try:
        for i in range(n):
            chunk = int16[i * _TEN_HOP_SIZE:(i + 1) * _TEN_HOP_SIZE]
            _prob, flag = vad.process(chunk)
            flags.append(bool(flag))
    except Exception:
        return False

    def ratio(start_ms: float, end_ms: float) -> float:
        a = max(0, int(round((start_ms - clip_start_ms) / frame_ms)))
        b = min(len(flags), int(round((end_ms - clip_start_ms) / frame_ms)))
        if b <= a:
            return 0.0
        return sum(1 for x in flags[a:b] if x) / (b - a)

    inside = ratio(pause.start_ms, pause.end_ms)
    left = ratio(pause.start_ms - side_ms, pause.start_ms)
    right = ratio(pause.end_ms, pause.end_ms + side_ms)
    return inside <= 0.25 and max(left, right) >= 0.15


def _merge_close_pauses(pauses: list[Pause], gap_ms: float) -> list[Pause]:
    """Merge pauses separated by a tiny speech blip."""
    if not pauses:
        return []
    out: list[Pause] = [pauses[0]]
    for p in pauses[1:]:
        prev = out[-1]
        if p.start_ms - prev.end_ms <= gap_ms:
            start = prev.start_ms
            end = p.end_ms
            out[-1] = Pause(start, end, end - start)
        else:
            out.append(p)
    return out


def _flags_to_pauses(
    speech_flags: list[bool],
    sr: int,
    min_pause_ms: float,
    *,
    window_samples: int = VAD_WINDOW,
) -> list[Pause]:
    """Свернуть пер-оконные флаги речи в паузы между речевыми участками.

    Берём промежутки ``not speech`` строго между первым и последним речевым
    окном (без лидирующей/хвостовой тишины) и оставляем только длиннее
    ``min_pause_ms``.
    """
    win_ms = window_samples * 1000.0 / sr

    speech_idx = [i for i, s in enumerate(speech_flags) if s]
    if len(speech_idx) < 2:
        return []
    first, last = speech_idx[0], speech_idx[-1]

    pauses: list[Pause] = []
    run_start: int | None = None
    for i in range(first, last + 1):
        if not speech_flags[i]:
            if run_start is None:
                run_start = i
        else:
            if run_start is not None:
                start_ms = run_start * win_ms
                end_ms = i * win_ms
                if end_ms - start_ms >= min_pause_ms:
                    pauses.append(Pause(start_ms, end_ms, end_ms - start_ms))
                run_start = None
    return pauses


# --- сопоставление: кандидаты границ ----------------------------------------


def boundary_candidates(
    surah: int,
    total_ms: float,
    pauses: list[Pause],
    *,
    db_path: str | Path = DEFAULT_DB_PATH,
    variant: str = "v4",
    tolerance_frac: float = MATCH_TOLERANCE_FRAC,
) -> list[BoundaryCandidate]:
    """Собрать кандидатов начала каждого аята из приора + пауз.

    Для каждого аята берём ожидаемую позицию (доля приора * ``total_ms``) и ищем
    ближайшую паузу в пределах ``tolerance_frac * total_ms``. Если нашли —
    кандидат опирается на центр паузы, уверенность растёт с глубиной паузы и
    падает с удалением от ожидания. Иначе кандидат «трудный» (confidence 0).

    Каждая пауза присваивается максимум одной границе (ближайшей по ожиданию),
    чтобы две соседние границы не схватили одну цезуру.

    Аят 1 (начало суры) всегда фиксируется в 0 мс с полной уверенностью.

    Returns:
        Кандидаты по возрастанию аята; ``time_ms`` неубывающая.
    """
    fractions = expected_boundary_fractions(surah, db_path=db_path, variant=variant)
    ayahs = sorted(fractions)
    tol_ms = tolerance_frac * total_ms

    # Жадно сопоставляем паузы границам: для каждой границы (кроме первой) —
    # ближайшая ещё свободная пауза в допуске.
    used = [False] * len(pauses)
    candidates: list[BoundaryCandidate] = []

    for ayah in ayahs:
        expected = fractions[ayah] * total_ms
        if ayah == ayahs[0]:
            # Начало суры — известно точно.
            candidates.append(BoundaryCandidate(ayah, 0.0, 0.0, 1.0, False))
            continue

        best_idx = -1
        best_dist = tol_ms
        for j, p in enumerate(pauses):
            if used[j]:
                continue
            dist = abs(p.center_ms - expected)
            if dist <= best_dist:
                best_dist = dist
                best_idx = j

        if best_idx >= 0:
            used[best_idx] = True
            p = pauses[best_idx]
            conf = _pause_confidence(p.depth_ms, best_dist, tol_ms)
            candidates.append(
                BoundaryCandidate(ayah, expected, p.center_ms, conf, True)
            )
        else:
            candidates.append(
                BoundaryCandidate(ayah, expected, expected, 0.0, False)
            )

    return _enforce_monotonic(candidates)


def _pause_confidence(depth_ms: float, dist_ms: float, tol_ms: float) -> float:
    """Уверенность кандидата из глубины паузы и удаления от ожидания → ``[0,1]``.

    Глубина: пауза ``MIN_PAUSE_MS`` даёт ~0, ``MIN_PAUSE_MS*3`` и глубже — ~1.
    Близость: точное попадание — 1, на краю допуска — 0. Итог — произведение.
    """
    span = max(MIN_PAUSE_MS * 2.0, 1.0)
    depth_score = min(max((depth_ms - MIN_PAUSE_MS) / span, 0.0), 1.0)
    prox_score = 1.0 - (dist_ms / tol_ms if tol_ms > 0 else 1.0)
    prox_score = min(max(prox_score, 0.0), 1.0)
    return round(depth_score * prox_score, 4)


def _enforce_monotonic(
    candidates: list[BoundaryCandidate],
) -> list[BoundaryCandidate]:
    """Гарантировать неубывание ``time_ms`` по порядку аятов.

    Если сопоставленная пауза дала откат назад, поджимаем кандидата к предыдущей
    позиции и обнуляем его уверенность (сигнал, что место подозрительно).
    """
    fixed: list[BoundaryCandidate] = []
    prev = 0.0
    for c in candidates:
        if c.time_ms < prev:
            fixed.append(
                BoundaryCandidate(c.ayah, c.expected_ms, prev, 0.0, c.from_pause)
            )
        else:
            prev = c.time_ms
            fixed.append(c)
    return fixed
