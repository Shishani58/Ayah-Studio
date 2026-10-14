"""Bench shared sequential FAST vs full vs ground truth (Shatri juz).

Run: python -m tools.bench_fast 093 100 080 087 090 078
Presets:  (no args) → the curated hard set; ``long`` → all suras >60 s that
actually exercise the windowed path; ``all`` → 78-114; ``corpus`` → 1-114.
``first77`` → 1-77, for a reproducible split full-corpus audit.
Add ``fast-only`` to skip the full reference (it never changes — only the fast
path is under iteration), roughly halving wall time. Audio dir via $SHATRI_DIR.
External ayah clips belong to fingerprint mode, not shared FAST.
Use ``--diagnose-candidates`` to print the ranked pause/ayah alternatives near
each reference error of at least one second.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from surah_cutter.core import align, align_fast, decode
from surah_cutter.data import quran_text, references

AUDIO = Path(os.environ.get("SHATRI_DIR", str(Path.home() / "Desktop" / "Abu_Bakr_Ash-Shatri")))
REF = references.load_reference("abu_bakr_ash_shatri")

# Historical duration grouping; shared FAST never falls back to FULL.
# The default is a
# curated subset spanning the failure modes (78/83 melodic off-by-one, 80 clean
# high-coverage, 87 the canonical broken sura, 89 wasl-heavy, 98 ~18 s ayahs).
LONG = [78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 96, 98]
HARD = ["078", "080", "083", "087", "089", "098"]


def err_stats(rows, surah):
    es = []
    for ayah, ms, _c in rows:
        gt = REF.get((surah, ayah))
        if gt is not None:
            es.append(abs(ms - gt))
    es.sort()
    if not es:
        return None
    n = len(es)
    return {
        "med": es[n // 2],
        "p90": es[min(n - 1, int(0.9 * n))],
        "max": es[-1],
        "le150": round(100 * sum(e <= 150 for e in es) / n),
        "le500": round(100 * sum(e <= 500 for e in es) / n),
        "n": n,
    }


def pause_contract_violations(rows, trace: dict) -> list[str]:
    """Return pause/wasl contract violations from one fast-alignment trace.

    A pause-backed output must equal the arithmetic centre of its final owner.
    A wasl boundary must be a CTC start rather than a disguised pause centre.
    Independently matched exact-recording clips are exempt from the pause rule.
    """
    starts = {ayah: start_ms for ayah, start_ms, _conf in rows}
    confidences = {ayah: confidence for ayah, _ms, confidence in rows}
    known_pause_centers = {
        float(pause["center_ms"])
        for pause in trace.get("phase1_pauses", [])
        if pause.get("center_ms") is not None
        and pause.get("quality") != "rejected"
    }
    violations: list[str] = []
    for match in trace.get("pause_matches", []):
        ayah = match.get("ayah")
        start_ms = starts.get(ayah)
        if start_ms is None:
            violations.append(f"ayah {ayah}: trace has no output start")
            continue
        if match.get("boundary_evidence") in {"exact_fingerprint", "ctc_pre_owner"}:
            continue
        if not match.get("from_pause"):
            if match.get("boundary_evidence") == "unresolved_local":
                if confidences.get(ayah) != 0.0 or ayah not in trace.get("review_required_ayahs", []):
                    violations.append(f"ayah {ayah}: unresolved local start lacks review flag/confidence=0")
                continue
            if match.get("match_action") != "wasl":
                violations.append(
                    f"ayah {ayah}: non-pause start {start_ms} has action "
                    f"{match.get('match_action')!r}, not wasl"
                )
            elif any(abs(start_ms - center) <= 1.0 for center in known_pause_centers):
                violations.append(
                    f"ayah {ayah}: wasl start {start_ms} equals a detected pause centre"
                )
            continue
        pause_start = match.get("pause_start_ms")
        pause_end = match.get("pause_end_ms")
        if pause_start is None or pause_end is None:
            violations.append(f"ayah {ayah}: pause-backed start {start_ms} has no interval")
        elif not pause_start <= start_ms <= pause_end:
            violations.append(
                f"ayah {ayah}: start {start_ms} outside pause "
                f"{pause_start}-{pause_end}"
            )
        else:
            expected_center = 0.5 * (float(pause_start) + float(pause_end))
            if abs(start_ms - expected_center) > 1.0:
                violations.append(
                    f"ayah {ayah}: pause-backed start {start_ms} is not centre "
                    f"{expected_center:g} of {pause_start}-{pause_end}"
                )
    return violations


def sequence_contract_violations(
    rows: list[tuple[int, int, float]],
    expected_ayahs: list[int],
) -> list[str]:
    """Require a complete in-order ayah sequence with strictly growing starts."""
    actual_ayahs = [ayah for ayah, _start_ms, _conf in rows]
    violations: list[str] = []
    if actual_ayahs != expected_ayahs:
        violations.append(
            f"ayah sequence {actual_ayahs[:8]}… does not equal expected "
            f"{expected_ayahs[:8]}…"
        )
    for (previous_ayah, previous_ms, _), (ayah, start_ms, _) in zip(rows, rows[1:]):
        if start_ms <= previous_ms:
            violations.append(
                f"ayah {previous_ayah}->{ayah}: non-increasing start "
                f"{previous_ms}->{start_ms}"
            )
    return violations


def outlier_diagnoses(
    rows,
    trace: dict,
    surah: int,
    *,
    threshold_ms: int = 500,
) -> list[str]:
    """Explain large reference errors by the matcher decision that produced them."""
    matches = {match.get("ayah"): match for match in trace.get("pause_matches", [])}
    diagnoses: list[str] = []
    for ayah, start_ms, _conf in rows:
        reference = REF.get((surah, ayah))
        if reference is None:
            continue
        error = abs(start_ms - reference)
        if error <= threshold_ms:
            continue
        match = matches.get(ayah, {})
        if not match.get("from_pause"):
            cause = (
                "merged-wasl-interval"
                if match.get("match_action") == "wasl"
                else "unclassified-non-pause"
            )
        elif match.get("match_action") == "recovered" and float(match.get("pause_sim") or 0.0) < 0.45:
            cause = "weak-recovered-pause-needs-edge-analysis"
        elif match.get("match_action") in {
            "preword_micro_recovery",
            "post_selected_micro_recovery",
            "repair_pause_recovery",
        }:
            cause = "post-recovery-pause-selection"
        else:
            cause = "pause-backed-read-or-edge-mismatch"
        diagnoses.append(
            f"ayah {ayah}: error={error}ms cause={cause} "
            f"action={match.get('match_action')} sim={match.get('pause_sim')} "
            f"timing={match.get('timing_strategy')} "
            f"pause={match.get('pause_start_ms')}-{match.get('pause_end_ms')} "
            f"start/ref={start_ms}/{reference}"
        )
    return diagnoses


def candidate_diagnoses(
    rows,
    trace: dict,
    surah: int,
    *,
    min_error_ms: float = 1000.0,
) -> list[str]:
    """Summarise the pause candidates nearest each large benchmark error."""
    rankings = trace.get("anchor_first", {}).get("pause_rankings", [])
    if not rankings:
        return []
    diagnoses: list[str] = []
    for ayah, actual_ms, _confidence in rows:
        reference_ms = REF.get((surah, ayah))
        if reference_ms is None or abs(actual_ms - reference_ms) < min_error_ms:
            continue
        nearby = sorted(
            rankings,
            key=lambda item: abs(float(item.get("pause_ms", 0.0)) - reference_ms),
        )[:3]
        for item in nearby:
            candidates = ",".join(
                f"{candidate['ayah']}:{candidate['similarity']:.3f}"
                f"/p{candidate['prefix_chars']}"
                for candidate in item.get("candidates", [])
            )
            diagnoses.append(
                f"ayah {ayah}: result/ref={round(actual_ms)}/{round(reference_ms)} "
                f"pause={round(float(item.get('pause_ms', 0.0)))} "
                f"accepted={bool(item.get('accepted'))} "
                f"reasons={','.join(item.get('rejection_reasons', [])) or '-'} "
                f"top=[{candidates}]"
            )
    return diagnoses


def select_suras(argv: list[str]) -> list[str]:
    if argv == ["first77"]:
        return [f"{n:03d}" for n in range(1, 78)]
    if argv == ["corpus"]:
        return [f"{n:03d}" for n in range(1, 115)]
    if argv == ["all"]:
        return [f"{n:03d}" for n in range(78, 115)]
    if argv == ["long"]:
        return [f"{n:03d}" for n in LONG]
    return argv or HARD


def _extract_ayah_audio_dir(argv: list[str]) -> tuple[list[str], str | None]:
    """Remove the optional exact-clip directory from simple benchmark args."""
    kept: list[str] = []
    directory: str | None = None
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--ayah-audio-dir":
            if index + 1 >= len(argv):
                raise SystemExit("--ayah-audio-dir needs a directory")
            directory = argv[index + 1]
            index += 2
            continue
        if arg.startswith("--ayah-audio-dir="):
            directory = arg.split("=", 1)[1]
            index += 1
            continue
        kept.append(arg)
        index += 1
    return kept, directory


def main(argv):
    argv, ayah_audio_dir = _extract_ayah_audio_dir(argv)
    fast_only = "fast-only" in argv
    audit_pauses = "--audit-pauses" in argv
    audit_sequence = "--audit-sequence" in argv
    diagnose_outliers = "--diagnose-outliers" in argv
    diagnose_candidates = "--diagnose-candidates" in argv
    surah_args = [
        a
        for a in argv
        if a not in {
            "fast-only",
            "--audit-pauses",
            "--audit-sequence",
            "--diagnose-outliers",
            "--diagnose-candidates",
        }
    ]
    suras = select_suras(surah_args)
    bundle = align._get_onnx_bundle()  # warm + reuse
    all_fast_err, all_full_err = [], []
    all_pause_contract_violations: list[str] = []
    all_sequence_contract_violations: list[str] = []
    tot_tf = tot_tu = 0.0
    for s in suras:
        surah = int(s)
        path = AUDIO / f"{int(s):03d}.mp3"
        if not path.exists():
            print(f"sura {surah}: NO AUDIO {path}")
            continue
        pcm = decode.decode(str(path))
        sr = decode.DEFAULT_SR
        words = quran_text.surah_words(surah)

        trace: dict = {}
        t0 = time.perf_counter()
        fast = align_fast.align_surah_fast(
            pcm,
            sr,
            words,
            surah=surah,
            bundle=bundle,
            trace=trace,
            ayah_audio_dir=ayah_audio_dir,
        )
        tf = time.perf_counter() - t0

        for ayah, ms, _c in fast:
            gt = REF.get((surah, ayah))
            if gt is not None:
                all_fast_err.append(abs(ms - gt))
        tot_tf += tf
        ef = err_stats(fast, surah)
        if audit_pauses:
            violations = pause_contract_violations(fast, trace)
            all_pause_contract_violations.extend(
                f"sura {surah}: {violation}" for violation in violations
            )
        if audit_sequence:
            sequence_violations = sequence_contract_violations(
                fast,
                sorted({word.ayah for word in words}),
            )
            all_sequence_contract_violations.extend(
                f"sura {surah}: {violation}" for violation in sequence_violations
            )
        if diagnose_outliers:
            for diagnosis in outlier_diagnoses(fast, trace, surah):
                print(f"   OUTLIER {diagnosis}")
        if diagnose_candidates:
            for diagnosis in candidate_diagnoses(fast, trace, surah):
                print(f"   CANDIDATE {diagnosis}")
        print(
            f"sura {surah:>3} ({trace.get('total_ms',0)/1000:5.1f}s, {len(words)}w, "
            f"pause={trace.get('n_pause')}/wasl={trace.get('n_wasl')}, "
            f"cover={trace.get('coverage')}):"
        )
        print(
            f"   FAST med={ef['med']:>5} p90={ef['p90']:>5} max={ef['max']:>6} "
            f"<=150={ef['le150']}% <=500={ef['le500']}%  {tf:5.1f}s"
        )
        if audit_pauses:
            status = "OK" if not violations else f"FAILED ({len(violations)})"
            print(f"   PAUSE CONTRACT: {status}")
        if audit_sequence:
            status = "OK" if not sequence_violations else f"FAILED ({len(sequence_violations)})"
            print(f"   SEQUENCE CONTRACT: {status}")
        if not fast_only:
            t0 = time.perf_counter()
            full = align.align_surah(pcm, sr, words, surah=surah, model=bundle)
            tu = time.perf_counter() - t0
            for ayah, ms, _c in full:
                gt = REF.get((surah, ayah))
                if gt is not None:
                    all_full_err.append(abs(ms - gt))
            tot_tu += tu
            eu = err_stats(full, surah)
            speed = tu / tf if tf else 0.0
            print(
                f"   FULL med={eu['med']:>5} p90={eu['p90']:>5} max={eu['max']:>6} "
                f"<=150={eu['le150']}% <=500={eu['le500']}%  {tu:5.1f}s   speed={speed:.2f}x"
            )

    def agg(es, label):
        es = sorted(es)
        n = len(es)
        if not n:
            return
        print(
            f"{label}: med={es[n//2]:>4} p90={es[int(0.9*n)]:>5} p95={es[int(0.95*n)]:>5} "
            f"<=150={round(100*sum(e<=150 for e in es)/n)}% "
            f"<=500={round(100*sum(e<=500 for e in es)/n)}%  (n={n})"
        )
    print("\n===== AGGREGATE =====")
    agg(all_fast_err, "FAST")
    if not fast_only:
        agg(all_full_err, "FULL")
        print(f"TOTAL time: fast={tot_tf:.0f}s full={tot_tu:.0f}s  speedup={tot_tu/tot_tf:.2f}x")
    else:
        print(f"TOTAL fast time: {tot_tf:.0f}s  (full reference skipped)")
    if audit_pauses and all_pause_contract_violations:
        print("\n===== PAUSE CONTRACT VIOLATIONS =====")
        print("\n".join(all_pause_contract_violations))
        return 1
    if audit_sequence and all_sequence_contract_violations:
        print("\n===== SEQUENCE CONTRACT VIOLATIONS =====")
        print("\n".join(all_sequence_contract_violations))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
