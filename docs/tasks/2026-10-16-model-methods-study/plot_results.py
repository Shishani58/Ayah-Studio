"""Static research figure from scored results; no inference or tuning."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import TASK


def main():
    data = json.loads((TASK / "paired-analysis.json").read_text())["paired"]
    secondary = json.loads((TASK / "secondary-scores.json").read_text())["summary"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 13), constrained_layout=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    variants = [("current", "Обязательный |"), ("no_delimiter", "Без обязательного |"),
                ("phones", "Фонемы без форматирования"), ("free_edges", "+ неизвестные края"),
                ("blank_minus1", "Blank logit −1")]
    rows = [next(r for r in data if r["context"] == "strong20" and r["variant"] == v and r["split"] == "all")
            for v, _ in variants]
    ax = axes[0]
    y = np.arange(len(rows))
    shared = np.array([r["over500_in_same_dsp_pause"] for r in rows])
    remaining = np.array([r["method"]["over500"] for r in rows])-shared
    ax.barh(y, remaining, color="#335c81", label="Без общего DSP-интервала")
    ax.barh(y, shared, left=remaining, color="#b8d2df", label="Обе метки в одном DSP-интервале")
    for i, row in enumerate(rows):
        ax.text(row["method"]["over500"]+4, i,
                f"{row['method']['over500']} · reliable: {row['bad_without_review']}", va="center", fontsize=9)
    ax.set_yticks(y, [label for _, label in variants])
    ax.invert_yaxis()
    ax.set_xlim(0, max(r["method"]["over500"] for r in rows)*1.42)
    ax.set_title(f"QPC INT8: отклонения >500 мс от внешних меток\n{rows[0]['records']} записи, {rows[0]['method']['n']} начал")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.08), ncol=2, fontsize=8, frameon=False)
    contexts = [("strong20", "Паузы · цель 20 с"), ("fixed30", "Фиксированные 30 с"),
                ("strong6", "Паузы · цель 6 с"), ("strong20-fp32", "Паузы 20 с · FP32"),
                ("strong20-localnorm", "Паузы 20 с · norm на вход")]
    ax = axes[1]
    labels, current, phones = [], [], []
    # strong20 baseline on the exact same12-record context subset.
    subset = next(r for r in data if r["context"] == "fixed30" and r["variant"] == "current" and r["split"] == "all")
    qpc_records = json.loads((TASK / "qpc-scores.json").read_text())["records"]
    ids = set(subset["record_ids"])
    for context, label in contexts:
        pair = []
        for variant in ("current", "phones"):
            selected = [r for r in qpc_records if r["context"] == context and r["variant"] == variant and r["record"] in ids]
            pair.append(sum(r["stats"]["over500"] for r in selected) if len(selected) == len(ids) else np.nan)
        labels.append(label)
        current.append(pair[0])
        phones.append(pair[1])
    y = np.arange(len(labels))
    ax.barh(y-.17, current, height=.32, color="#335c81", label="Текущая цель")
    ax.barh(y+.17, phones, height=.32, color="#c8823c", label="Фонемная цепь")
    for offset, values in [(-.17,current),(.17,phones)]:
        for i,value in enumerate(values):
            if np.isfinite(value):
                ax.text(value+.5,i+offset,str(int(value)),va="center",fontsize=8)
    ax.set_xlim(0, max(current)*1.15)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_title(f"Контекст, точность весов, нормализация: те же {len(ids)} записей\nЧисло отклонений >500 мс; пропуск означает вариант не измерялся")
    ax.legend(loc="upper center", bbox_to_anchor=(.5,-.08), ncol=2, fontsize=8, frameon=False)
    ax = axes[2]
    choices = [("ctc", "FastConformer · CTC"), ("rnnt", "FastConformer · RNN-T"),
               ("greedy", "Tarteel Base · greedy"), ("beam5", "Tarteel Base · beam5"),
               ("greedy_overlap", "Tarteel Base · overlap 5 с"),
               ("greedy_sdpa", "Tarteel Base · greedy SDPA")]
    selected = [(r, label) for method, label in choices for r in secondary
                if r["method"] == method and r["records"] == 12]
    ax.barh(np.arange(len(selected)), [100*r["wer"] for r,_ in selected], color="#335c81")
    ax.set_yticks(np.arange(len(selected)), [label for _,label in selected])
    for i, (row, _) in enumerate(selected):
        ax.text(100*row["wer"]+.3, i, f"{100*row['wer']:.2f}%", va="center")
    ax.invert_yaxis()
    ax.set_xlim(0,max(100*r["wer"] for r,_ in selected)*1.22)
    ax.set_title("Самостоятельное распознавание · 12 полных записей\nWER относительно канона с басмалой; ожидаемый текст не подаётся")
    ax.set_xlabel("WER, %")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
        ax.set_axisbelow(True)
    fig.savefig(TASK / "comparison.png", dpi=180)


if __name__ == "__main__":
    main()
