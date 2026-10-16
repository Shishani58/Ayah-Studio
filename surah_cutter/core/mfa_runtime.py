"""Quran Hafs MFA adapter; native Kaldi runs outside the GUI process."""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unicodedata
import wave

import numpy as np

VERSION = "mfa-hafs-v1"
MFA_VERSION = "3.4.0"


def executable() -> Path:
    from surah_cutter.core import model_store

    override = os.environ.get("SURAH_CUTTER_MFA")
    path = Path(override) if override else model_store.ROOT.parent / "mfa-runtime" / (
        "Scripts/mfa.exe" if os.name == "nt" else "bin/mfa")
    if not path.is_file():
        found = shutil.which("mfa") if not override else None
        if found:
            path = Path(found)
        else:
            raise ValueError("Установите MFA 3.4.0 в ~/.surah-cutter/mfa-runtime "
                             "или укажите исполняемый файл через SURAH_CUTTER_MFA.")
    return path.resolve()


def _environment(binary: Path, work: Path) -> dict:
    env = os.environ.copy()
    for name in ("PYTHONHOME", "PYTHONPATH"):
        env.pop(name, None)
    env.update({"PATH": str(binary.parent) + os.pathsep + env.get("PATH", ""),
                "MFA_ROOT_DIR": str(work / "mfa"), "OMP_NUM_THREADS": "4",
                "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
    return env


def _run(command: list[str], work: Path, *, timeout: float = 1800) -> None:
    from surah_cutter.core.cancellation import check_cancelled, interruptible

    check_cancelled()
    log_path = work / "mfa.log"
    with log_path.open("ab") as log:
        process = subprocess.Popen(command, stdout=log, stderr=log,
                                   env=_environment(Path(command[0]), work), cwd=work)
        try:
            def abort():
                try:
                    process.kill()
                except ProcessLookupError:
                    pass

            with interruptible(abort):
                deadline = time.monotonic() + timeout
                while process.poll() is None:
                    check_cancelled()
                    if time.monotonic() >= deadline:
                        raise ValueError("MFA превысила время ожидания.")
                    time.sleep(.1)
                check_cancelled()
                if process.returncode:
                    log.flush()
                    logging.getLogger(__name__).warning("MFA: %s", log_path.read_text(errors="replace")[-8000:])
                    raise ValueError("MFA не смогла выровнять запись. Проверьте текст, аудио и окружение.")
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()


def parse_word_intervals(data: dict, expected: list[str], duration_ms: float) -> list[int]:
    """Require exact word ownership; never accept an OOV or a missing interval."""
    tiers = [tier for name, tier in data.get("tiers", {}).items()
             if name == "words" or name.endswith(" - words")]
    if len(tiers) != 1:
        raise ValueError("MFA не вернула единственный слой слов.")
    entries = [entry for entry in tiers[0]["entries"] if entry[2].strip() and entry[2] != "<eps>"]
    if [entry[2] for entry in entries] != expected:
        raise ValueError("Слова MFA не совпадают с каноническими целями.")
    starts, previous_end = [], 0.0
    for start, end, _label in entries:
        start_ms, end_ms = 1000 * float(start), 1000 * float(end)
        if (not np.isfinite([start_ms, end_ms]).all() or start_ms < previous_end - .1
                or end_ms <= start_ms or start_ms < 0 or end_ms > duration_ms + 20):
            raise ValueError("MFA вернула недопустимые границы слов.")
        ms = round(start_ms)
        if starts and ms <= starts[-1]:
            raise ValueError("Границы слов MFA не различимы.")
        starts.append(ms)
        previous_end = end_ms
    return starts


class MfaEmitter:
    model_id = "mfa-quran-hafs"
    text_kind = "mfa"
    vocab = {"unused": 0}
    blank_id = 0

    def __init__(self, folder: Path):
        self.folder = folder
        self.binary = executable()
        self.dictionary = folder / "quran_hafs.dict"
        self.acoustic = folder / "quran_hafs_acoustic.zip"
        # MFA's FileData normalizes transcripts to NFKC, but Kalpy does not
        # normalize dictionary keys. Normalize only the lookup copy of both.
        self.lookup_dictionary = "\n".join(
            unicodedata.normalize("NFKC", key) + "\t" + phones
            for line in self.dictionary.read_text().splitlines() if line.strip()
            for key, phones in [line.split(None, 1)]) + "\n"
        self.lexicon = {line.split()[0] for line in self.lookup_dictionary.splitlines()}
        self.runtime_versions = {"adapter": VERSION, "mfa": MFA_VERSION}

    def probe(self):
        # Check native extensions too: an installed CLI alone does not certify Kaldi.
        with tempfile.TemporaryDirectory(prefix="ayah-mfa-probe-") as name:
            work = Path(name)
            _run([str(self.binary), "version"], work, timeout=60)
            text = (work / "mfa.log").read_text()
            if MFA_VERSION not in text.splitlines():
                raise ValueError(f"Для воспроизводимости требуется MFA {MFA_VERSION}.")
        from surah_cutter.data.quran_text import surah_words

        # Exercise the downloaded acoustic weights, lexicon and native decoder.
        # This is a compatibility probe, not evidence of speech on silence.
        self.align_words(np.zeros(16000, dtype=np.float32), surah_words(112)[:1])

    def align_words(self, pcm, words, intro=False, strict_words=True):
        from surah_cutter.core.model_runtime import inference_words
        from surah_cutter.data.quran_text import surah_words

        pcm = np.asarray(pcm, dtype=np.float32)
        if pcm.ndim != 1 or len(pcm) < 1600 or not np.isfinite(pcm).all() or not words:
            raise ValueError("Недостаточно аудио для MFA.")
        rendered = inference_words(words, "muaalem")  # Verified Uthmani copy, same canonical owners.
        prefix = inference_words([w for w in surah_words(1) if w.ayah == 1], "muaalem") if intro else []
        units = [unicodedata.normalize("NFKC", unit) for text in prefix + rendered for unit in text.split()]
        if any(unit not in self.lexicon for unit in units):
            raise ValueError("Слово отсутствует в словаре MFA; подстановка OOV запрещена.")
        with tempfile.TemporaryDirectory(prefix="ayah-mfa-") as name:
            work = Path(name)
            audio = work / "audio.wav"
            with wave.open(str(audio), "wb") as output:
                output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                output.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())
            transcript = work / "audio.lab"
            transcript.write_text(" ".join(units), encoding="utf-8")
            dictionary = work / "lookup.dict"
            dictionary.write_text(self.lookup_dictionary, encoding="utf-8")
            result = work / "alignment.json"
            _run([str(self.binary), "align_one", str(audio), str(transcript),
                  str(dictionary), str(self.acoustic), str(result),
                  "--output_format", "json", "--beam", "40", "--retry_beam", "160",
                  "--boost_silence", "1.0", "--single_speaker", "--clean"], work)
            starts = parse_word_intervals(json.loads(result.read_text()), units, len(pcm) / 16)
        cursor = sum(len(text.split()) for text in prefix)
        rows = []
        for word, text in zip(words, rendered):
            # Some canonical words contain multiple orthographic units (37:130).
            # Preserve one canonical index, using the first unit's acoustic start.
            rows.append((word.index, starts[cursor], 0.0))
            cursor += len(text.split())
        # MFA's likelihood is not a calibrated probability. Keep every row in review.
        return rows

    @property
    def runtime_identity(self):
        return {"adapter": VERSION, "mfa": MFA_VERSION, "files": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (self.dictionary, self.acoustic)},
            "lookup_sha256": hashlib.sha256(self.lookup_dictionary.encode()).hexdigest()}
