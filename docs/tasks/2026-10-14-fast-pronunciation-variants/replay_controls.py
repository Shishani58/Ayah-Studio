"""Derived normalization audit of retained real controls, without new inference."""
import ast
import hashlib
import json
from pathlib import Path

task = Path(__file__).parent
parent = task.parent / "2026-10-14-fast-muaalem-local-analysis"
source = (task / "collapse_probe.py").read_text()
function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "collapse")
namespace = {"LONG": {"ا", "ۦ", "ۥ", "ـ"}}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(task / "collapse_probe.py"), "exec"), namespace)
collapse = namespace["collapse"]
controls = json.loads((parent / "controls.json").read_text())
records = []
for item in controls["records"]:
    units, scores, times, owners = collapse(item["greedy"], item["unit_quality"],
                                          list(range(len(item["greedy"]))))
    target = collapse(item["expected"])[0]
    wrong = collapse(item["wrong_target"])[0]
    matching = [i for i in range(len(units)) if units[i:i + len(target)] == target]
    wrong_matching = [i for i in range(len(units)) if units[i:i + len(wrong)] == wrong]
    records.append({"control": item["control"], "target_matches": matching,
                    "wrong_target_matches": wrong_matching, "normalization": units,
                    "retained_minimum_scores": [scores[i] == min(item["unit_quality"][j] for j in ids)
                                                for i, ids in enumerate(owners)]})
report = {"records": records, "normalizer_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
          "scope": "derived replay of real observed phones, not new acoustic controls"}
(task / "derived-controls.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print([(r["control"],r["target_matches"],r["wrong_target_matches"]) for r in records])
