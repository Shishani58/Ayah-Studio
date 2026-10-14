"""Readonly corpus analysis; run with .venv/bin/python from any directory."""

import csv
import hashlib
import json
import sqlite3
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]


def read_rows(path, query):
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        return conn.execute(query).fetchall()


def normal(text):
    # Comparison only: never write normalized Quran text back to the source.
    text = "".join(c for c in text if not unicodedata.combining(c) and c != "ـ")
    return text.translate(str.maketrans("ٱأإآ", "اااا"))


def medoids(distance, k):
    centers = [int(np.argmin(distance.sum(axis=1)))]
    while len(centers) < k:
        centers.append(int(np.argmax(distance[:, centers].min(axis=1))))
    for _ in range(100):
        labels = distance[:, centers].argmin(axis=1)
        updated = []
        for i, center in enumerate(centers):
            members = np.flatnonzero(labels == i)
            updated.append(int(members[np.argmin(distance[np.ix_(members, members)].sum(axis=1))])
                           if len(members) else center)
        if updated == centers:
            break
        centers = updated
    return centers, distance[:, centers].argmin(axis=1)


def main():
    paths = [ROOT / "data/quran_mushaf.db", *sorted((ROOT / "data/references").glob("*.db"))]
    source = ROOT / "docs/tasks/2026-10-14-full-pause-windows/surah-selection.json"
    hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [*paths, source]}
    texts = defaultdict(list)
    first_raw = {}
    for s, a, text in read_rows(paths[0], "SELECT surah,ayah,qpc_hafs FROM words ORDER BY id"):
        first_raw.setdefault(s, text)
        while len(texts[s]) < a:
            texts[s].append([])
        texts[s][a - 1].extend(normal(text).split())
    counts = {s: len(v) for s, v in texts.items()}
    assert len(counts) == 114 and sum(counts.values()) == 6236
    assert counts == dict(read_rows(paths[0],
                                   "SELECT surah,COUNT(DISTINCT ayah) FROM words GROUP BY surah"))
    durations = {r["surah"]: r["duration_s"] for r in json.loads(source.read_text())["durations"]}
    assert set(durations) == set(counts)
    intervals = defaultdict(list)
    audit = []
    shatri = {}
    for path in paths[1:]:
        rows = read_rows(path, "SELECT sura,ayah,time FROM timings ORDER BY sura,ayah")
        by_sura = defaultdict(list)
        extra = []
        for s, a, t in rows:
            if s in counts and 1 <= a <= counts[s]:
                by_sura[s].append((a, t))
            else:
                extra.append([s, a, t])
        bad = []
        for s, n in counts.items():
            r = by_sura[s]
            if ([a for a, _ in r] != list(range(1, n + 1))
                    or any(not isinstance(t, int) or t < 0 for _, t in r)):
                bad.append(s)
                continue
            delta = np.diff([t for _, t in r]) / 1000
            if np.any(delta <= 0):
                bad.append(s)
                continue
            intervals[s].append(delta)
            if path.stem == "abu_bakr_ash_shatri":
                assert r[-1][1] / 1000 < durations[s]
                shatri[s] = [t / 1000 for _, t in r]
        audit.append({"reference": path.stem, "invalid_surahs": bad,
                      "ignored_noncanonical_rows": extra})
    features = []
    letters = {"الم", "المص", "الر", "المر", "طسم", "طه", "يس", "ص", "حم", "ق", "ن", "كهيعص", "طس"}
    for s in range(1, 115):
        verses = texts[s]
        lengths = np.array([len(v) for v in verses], dtype=float)
        chars = np.array([sum(len(w) for w in v) for v in verses], dtype=float)
        canonical_delta = np.median(np.stack(intervals[s]), axis=0)
        prefix = []
        for k in [1, 2, 3, 6]:
            keys = [tuple(v[:k]) for v in verses]
            freq = Counter(keys)
            prefix.append(sum(freq[key] > 1 for key in keys) / len(keys))
        full = Counter(tuple(v) for v in verses)
        common = []
        for i, verse in enumerate(verses):
            longest = 0
            for j, other in enumerate(verses):
                if i == j:
                    continue
                lcp = 0
                for w, z in zip(verse, other):
                    if w != z:
                        break
                    lcp += 1
                longest = max(longest, lcp)
            common.append(longest)
        profile = (np.array([part.mean() for part in np.array_split(np.log1p(lengths), 8)])
                   if len(lengths) >= 8 else
                   np.interp(np.linspace(0, 1, 8), np.linspace(0, 1, len(lengths)), np.log1p(lengths)))
        profile -= np.log1p(np.median(lengths))
        p10, p50, p90 = np.percentile(lengths, [10, 50, 90])
        row = {
            "surah": s, "ayahs": len(verses), "words": int(lengths.sum()),
            "audio_shatri_s": durations[s], "valid_references": len(intervals[s]),
            "interval_p10_s": float(np.percentile(canonical_delta, 10)),
            "interval_median_s": float(np.median(canonical_delta)),
            "interval_p90_s": float(np.percentile(canonical_delta, 90)),
            "interval_max_s": float(canonical_delta.max()),
            "word_p10": float(p10), "word_median": float(p50), "word_p90": float(p90),
            "word_max": int(lengths.max()), "word_cv": float(lengths.std() / lengths.mean()),
            "char_median": float(np.median(chars)),
            "neighbor_jump": float(np.abs(np.diff(np.log1p(lengths))).mean()),
            "prefix1_repeat": prefix[0], "prefix2_repeat": prefix[1],
            "prefix3_repeat": prefix[2], "prefix6_repeat": prefix[3],
            "full_repeat": sum(full[tuple(v)] > 1 for v in verses) / len(verses),
            "max_common_prefix_words": max(common),
            "letters_start": int(verses[0][0] in letters and not first_raw[s].startswith("أ")),
            "shatri_last_ayah_s": durations[s] - shatri[s][-1],
            "shatri_max_interval_s": float(np.diff([*shatri[s], durations[s]]).max()),
            "max_ayah": int(lengths.argmax() + 1),
            "profile": profile.tolist(),
        }
        row["duration_group"] = int(np.searchsorted([5, 10, 20], row["interval_median_s"], side="right")) + 1
        features.append(row)
    # Three equally weighted blocks: length distribution, repeats, sequence shape.
    blocks = [np.array([[np.log1p(r[k]) for k in
                         ["word_p10", "word_median", "word_p90", "word_max", "char_median"]]
                        + [r["word_cv"]] for r in features]),
              np.array([[r[k] for k in ["prefix1_repeat", "prefix2_repeat", "prefix3_repeat",
                                        "prefix6_repeat", "full_repeat"]] for r in features]),
              np.array([[r["neighbor_jump"], *r["profile"]] for r in features])]
    scaled = [(b - b.mean(axis=0)) / np.maximum(b.std(axis=0), 1e-9) / np.sqrt(b.shape[1])
              for b in blocks]
    matrix = np.concatenate(scaled, axis=1)
    distance = np.sqrt(((matrix[:, None, :] - matrix[None, :, :]) ** 2).sum(axis=2) / 3)
    centers, labels = medoids(distance, 8)
    groups = []
    for i, center in enumerate(centers):
        members = np.flatnonzero(labels == i)
        short = [int(j) for j in members if durations[int(j) + 1] <= 600]
        candidates = short or members.tolist()
        # Minimise mean group distance within <=10min; duration breaks ties only.
        rep = min(candidates, key=lambda j: (distance[j, members].mean(), durations[j + 1]))
        groups.append({"group": i + 1, "medoid": center + 1, "representative": rep + 1,
                       "members": (members + 1).tolist(),
                       "short_candidates": sorted((j + 1 for j in candidates), key=lambda s: durations[s])})
        for j in members:
            features[j]["structure_group"] = i + 1
    for i, row in enumerate(features):
        row["nearest_surahs"] = [int(j + 1) for j in np.argsort(distance[i]) if j != i][:5]
        shorter = [j for j in range(114) if durations[j + 1] < durations[i + 1] and j != i]
        row["nearest_shorter"] = sorted(shorter, key=lambda j: distance[i, j])[:5]
        row["nearest_shorter"] = [j + 1 for j in row["nearest_shorter"]]
    core = [1, 32, 55, 59, 65, 73, 77, 80, 98, 112]
    extended = sorted(set(core + [83, 95, 109, 113, 114]))
    suites = {}
    for name, selected in [("core", core), ("extended", extended)]:
        suites[name] = {"surahs": selected, "ayahs": sum(counts[s] for s in selected),
                        "audio_shatri_s": sum(durations[s] for s in selected),
                        "structure_groups": sorted({features[s - 1]["structure_group"] for s in selected}),
                        "duration_groups": sorted({features[s - 1]["duration_group"] for s in selected})}
    result = {"method": "8 deterministic k-medoids clusters; 3 equal standardized feature blocks",
              "reference_audit": audit, "source_sha256": hashes,
              "corpus_audio_shatri_s": sum(durations.values()), "groups": groups,
              "features": features, "suites": suites}
    (OUT / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    columns = [k for k in features[0] if k not in {"profile", "nearest_surahs", "nearest_shorter"}]
    with (OUT / "surahs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(features)
    np.savetxt(OUT / "distances.csv", distance, delimiter=",", fmt="%.8f",
               header=",".join(str(s) for s in range(1, 115)), comments="")
    lines = ["# Группы всех 114 сур", "", "## По длительности аятов", "",
             "Медиана start→next start после межчтецовой медианы для каждого аята; "
             "последний аят исключён. Паузы входят в интервал.", "",
             "| Медиана интервала | Сур | Номера |", "|---|---:|---|"]
    for g, title in enumerate(["Меньше 5 с", "От 5 до 10 с", "От 10 до 20 с", "20 с и больше"], 1):
        members = [r["surah"] for r in features if r["duration_group"] == g]
        lines.append(f"| {title} | {len(members)} | {', '.join(map(str, members))} |")
    lines += ["", "## По структуре текста", "",
              "Восемь статистических групп для подбора fixtures. Номер группы не "
              "означает уровень сложности. Близость не гарантирует взаимозаменяемость.", "",
              "| Группа | Центральная сура | Представитель в основном наборе | Все суры |",
              "|---|---:|---|---|"]
    for group in groups:
        chosen = sorted(set(group["members"]) & set(core))
        lines.append(f"| {group['group']} | {group['medoid']} | "
                     f"{', '.join(map(str, chosen))} | {', '.join(map(str, group['members']))} |")
    lines += ["", "Автоматический короткий представитель в JSON минимизирует среднее "
              "расстояние внутри группы среди записей не длиннее 10 минут. "
              "Основной набор дополнительно учитывает начальные буквы, многократные "
              "повторы, длинные общие префиксы и известные регрессии. Полная методика "
              "и ограничения описаны в report.md.", ""]
    (OUT / "groups.md").write_text("\n".join(lines))
    assert all(hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h for p, h in hashes.items())
    print(json.dumps({"suites": suites, "groups": groups}, ensure_ascii=False))


if __name__ == "__main__":
    main()
