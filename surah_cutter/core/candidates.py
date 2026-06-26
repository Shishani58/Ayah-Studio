"""Дешёвая стадия кандидатов: паузы (Silero VAD) + структурный приор.

Идея гибридного режима: до запуска тяжёлой нейросетевой выравнивалки получить
дешёвые подсказки о том, где проходят границы аятов.

Два источника:

1. **Паузы речи** — Silero VAD (ONNX) даёт сегменты речь/тишина. Границы между
   аятами почти всегда лежат в паузе; «глубина» паузы (её длительность) служит
   уверенностью кандидата.
2. **Структурный приор** — ожидаемые относительные позиции границ аятов,
   выведенные из ширины рендера слов (``qpc_v4_render_width_millis`` /
   ``qpc_v1_render_width_millis`` в ``data/quran_mushaf.db``). Это пропорция, а
   не абсолютное время: чем длиннее слова аята, тем дольше он звучит. Приор
   нормирован в долю ``[0, 1]`` от полной длительности суры.

Сопоставив паузы с ожидаемыми позициями, для каждой границы аята получаем
кандидата: либо чёткая пауза рядом с ожиданием (берём напрямую, без НС), либо
паузы нет/неоднозначна — аят «трудный», его отдаём выравнивалке.

Тяжёлые зависимости (``onnxruntime``, ``numpy`` для ресемпла/инференса)
импортируются лениво внутри функций; модуль импортируется без них. Чтение БД —
чистый ``sqlite3`` локально, чтобы не трогать ``data.quran_text``.

Никаких загрузок модели в момент импорта/вызова без явного запроса
(``allow_download``). По умолчанию при отсутствии файла VAD деградируем в режим
«паузы неизвестны» — кандидаты строятся только по приору (все границы трудные).
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
MIN_PAUSE_MS = 220.0

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
class BoundaryCandidate:
    """Кандидат границы начала аята.

    Attributes:
        ayah: Номер аята, *начало* которого описывает эта граница.
        expected_ms: Ожидаемая позиция начала (мс) по структурному приору.
        time_ms: Итоговая позиция кандидата (мс) — центр сопоставленной паузы,
            либо ``expected_ms``, если чёткой паузы не нашлось.
        confidence: Уверенность ``[0, 1]``. >0 только если рядом с ожиданием
            нашлась достаточно глубокая пауза; иначе ``0.0`` (аят «трудный»).
        from_pause: ``True``, если кандидат опёрт на реальную паузу VAD.
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


# --- Silero VAD: паузы ------------------------------------------------------


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
    """Найти паузы (сегменты тишины) в моно-PCM через Silero VAD (ONNX).

    Прогоняет VAD окнами по :data:`VAD_WINDOW` сэмплов, собирает вероятность
    речи на кадр, затем выделяет промежутки между речевыми кадрами длиннее
    ``min_pause_ms``. Лидирующая и хвостовая тишина в паузы не включаются —
    нас интересуют цезуры *внутри* речи.

    Если модели нет и ``allow_download=False`` — возвращает ``[]`` (паузы
    неизвестны), не падая: вызывающий код деградирует к чистому приору.

    Тяжёлые импорты (``numpy``, ``onnxruntime``) — ленивые.
    """
    if not Path(model_path).exists() and not allow_download:
        return []

    import numpy as np
    import onnxruntime as ort

    path = ensure_vad_model(model_path, allow_download=allow_download)

    audio = np.asarray(pcm, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        return []

    sess = ort.InferenceSession(
        str(path), providers=["CPUExecutionProvider"]
    )
    input_names = {i.name for i in sess.get_inputs()}

    # Silero VAD (ONNX) держит рекуррентное состояние между окнами.
    state = np.zeros((2, 1, 128), dtype=np.float32)
    sr_arr = np.array(sr, dtype=np.int64)

    # Контекст v5: последние VAD_CONTEXT сэмплов прошлого окна. На первом окне —
    # нули. Конкатенируем context+chunk → вход 576 сэмплов; иначе сеть слепнет.
    context = np.zeros((1, VAD_CONTEXT), dtype=np.float32)

    n_windows = audio.size // VAD_WINDOW
    speech_flags: list[bool] = []
    for i in range(n_windows):
        chunk = audio[i * VAD_WINDOW : (i + 1) * VAD_WINDOW].reshape(1, -1)
        inp = np.concatenate([context, chunk], axis=1)
        feeds = {"input": inp, "state": state, "sr": sr_arr}
        # Некоторые экспорты называют ставку "sr" иначе — отфильтруем по схеме.
        feeds = {k: v for k, v in feeds.items() if k in input_names}
        out = sess.run(None, feeds)
        prob = float(np.asarray(out[0]).reshape(-1)[0])
        if len(out) > 1:
            state = np.asarray(out[1], dtype=np.float32)
        # Контекст следующего окна — хвост ТЕКУЩЕГО окна (без префикса контекста).
        context = chunk[:, -VAD_CONTEXT:]
        speech_flags.append(prob >= threshold)

    return _flags_to_pauses(speech_flags, sr, min_pause_ms)


def _flags_to_pauses(
    speech_flags: list[bool],
    sr: int,
    min_pause_ms: float,
) -> list[Pause]:
    """Свернуть пер-оконные флаги речи в паузы между речевыми участками.

    Окно = :data:`VAD_WINDOW` сэмплов. Берём промежутки ``not speech`` строго
    между первым и последним речевым окном (без лидирующей/хвостовой тишины) и
    оставляем только длиннее ``min_pause_ms``.
    """
    win_ms = VAD_WINDOW * 1000.0 / sr

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
