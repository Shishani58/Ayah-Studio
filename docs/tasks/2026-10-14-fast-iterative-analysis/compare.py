"""Compare independent fresh runs after inference, ignoring timing/identity keys."""
import json
import sys
from pathlib import Path
import pstats

root = Path(__file__).parent
left_name, right_name = sys.argv[1:3]
left, right = (json.loads((root / name).read_text()) for name in (left_name,right_name))
lut = {(i["model"],i["surah"]): i for i in left}
records = []
unpaired = []
for b in right:
    if (b["model"],b["surah"]) not in lut:
        unpaired.append([b["model"], b["surah"]])
        continue
    a = lut[b["model"],b["surah"]]
    ta,tb = a["trace"],b["trace"]
    reads_a=[(r["window_ms"],r["reason"],d["sha256"]) for r,d in zip(ta["acoustic_reads"],a["actual_dispatches"])]
    reads_b=[(r["window_ms"],r["reason"],d["sha256"]) for r,d in zip(tb["acoustic_reads"],b["actual_dispatches"])]
    records.append({"model":b["model"],"surah":b["surah"],
        "rows_equal":a["rows"]==b["rows"],"boundaries_equal":ta["boundaries"]==tb["boundaries"],
        "reads_equal":reads_a==reads_b,"reviews_equal":ta["review_required_ayahs"]==tb["review_required_ayahs"],
        "seconds_before":a["seconds"],"seconds_after":b["seconds"]})
report={"left":left_name,"right":right_name,"records":records,"unpaired_cases":unpaired,"models":{}}
for model in sorted({r["model"] for r in records}):
    subset=[r for r in records if r["model"]==model]
    before=sum(r["seconds_before"] for r in subset)
    after=sum(r["seconds_after"] for r in subset)
    report["models"][model]={"cases":len(subset),"seconds_before":before,"seconds_after":after,
        "saved_percent":100*(before-after)/before,
        "rows_all_equal":all(r["rows_equal"] for r in subset),
        "boundaries_all_equal":all(r["boundaries_equal"] for r in subset),
        "reads_all_equal":all(r["reads_equal"] for r in subset)}
if (root/"proposal.profile").exists():
    report["profile"]={}
    for filename in ("baseline.profile","proposal.profile"):
        stats=pstats.Stats(str(root/filename))
        report["profile"][filename]={key[2]:{"calls":v[1],"cumulative_seconds":v[3]}
            for key,v in stats.stats.items() if key[2] in ("plan_phrase","_competing_prefix","prefix_scores")}
output=root/(Path(right_name).stem+".comparison.json")
output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")
print(json.dumps({k:v for k,v in report.items() if k!="records"},ensure_ascii=False,indent=2))
