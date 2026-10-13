"""Декодирование аудио в 16 kHz mono float32 через bundled ffmpeg.

Используется ffmpeg из `imageio_ffmpeg` (не системный), вывод гонится через
pipe как f32le и парсится numpy. Модуль не зависит от GUI/torch.
"""

from __future__ import annotations

import subprocess

import numpy as np

from surah_cutter.core.cancellation import current_event, interruptible

DEFAULT_SR = 16000


class DecodeError(RuntimeError):
    """Ошибка декодирования аудио ffmpeg-ом."""


def _ffmpeg_exe() -> str:
    """Путь к bundled ffmpeg (ленивый импорт imageio_ffmpeg)."""
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def decode(path: str, sr: int = DEFAULT_SR) -> np.ndarray:
    """Декодировать аудио в моно float32 numpy при частоте `sr`.

    Любой поддерживаемый ffmpeg-ом формат (mp3/m4a/ogg/flac/wav). Возвращает
    1-D массив float32 в диапазоне [-1, 1]. При ошибке ffmpeg бросает
    `DecodeError` со stderr-выводом.
    """
    cmd = [
        _ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-",
    ]
    if current_event() is None:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            with interruptible(child.kill):
                stdout, stderr = child.communicate()
            proc = subprocess.CompletedProcess(cmd, child.returncode, stdout, stderr)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise DecodeError(f"ffmpeg failed for {path!r}: {msg or 'unknown error'}")

    # frombuffer отдаёт read-only view на bytes; .copy() даёт writable массив.
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def decode_range(
    path: str,
    *,
    start_ms: float,
    end_ms: float,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """Decode a time range to mono float32 PCM at ``sr``.

    Used by debug overlays that inspect only the visible waveform region on long
    suras; avoids decoding multi-hour audio just to run a local detector.
    """
    start_s = max(0.0, start_ms / 1000.0)
    dur_s = max(0.0, (end_ms - start_ms) / 1000.0)
    if dur_s <= 0:
        return np.array([], dtype=np.float32)
    cmd = [
        _ffmpeg_exe(),
        "-nostdin",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-t", f"{dur_s:.3f}",
        "-i", str(path),
        "-f", "f32le",
        "-acodec", "pcm_f32le",
        "-ac", "1",
        "-ar", str(sr),
        "-",
    ]
    if current_event() is None:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            with interruptible(child.kill):
                stdout, stderr = child.communicate()
            proc = subprocess.CompletedProcess(cmd, child.returncode, stdout, stderr)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()
        raise DecodeError(f"ffmpeg failed for {path!r}: {msg or 'unknown error'}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def duration_ms(path: str, sr: int = DEFAULT_SR) -> int:
    """Длительность аудио в миллисекундах (через декод при `sr`)."""
    n = len(decode(path, sr=sr))
    return int(round(n * 1000 / sr))
