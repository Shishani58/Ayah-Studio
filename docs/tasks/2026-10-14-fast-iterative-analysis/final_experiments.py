"""Serial, isolated experiments. Does not invoke pytest or modify sources."""
from pathlib import Path
import os
import subprocess
import sys
import zipfile
import hashlib
import json

root=Path(__file__).resolve().parents[3]
task=Path(__file__).parent
source=Path((task/"snapshot-path.txt").read_text().strip())
for name in ("source","proposal","integrated-before","integrated"):
    folder=source.parent/name
    files=[*folder.joinpath("surah_cutter").rglob("*.py"),*folder.joinpath("tools").rglob("*.py"),folder/"pyproject.toml"]
    with zipfile.ZipFile(task/(name+"-source.zip"),"w",zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path,str(path.relative_to(folder)))
    (task/(name+"-manifest.json")).write_text(json.dumps({str(p.relative_to(folder)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},indent=2)+"\n")

common=["--audio-dir","/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri","--reference",str(root/"data/references/abu_bakr_ash_shatri.db")]
cases=[("short",source,[sys.executable,"-u",str(task/"window_trial.py"),"1000,2000,4000","--models","quran-phonetics","whisper-tarteel-tiny","whisper-tarteel-base","--surahs","1","55","80","98","112"]),
       ("integrated-before",source.parent/"integrated-before",[sys.executable,"-u","-m","tools.bench_fast_models","--models","quran-phonetics","--surahs","1","32","55","59","65","73","77","80","98","112","2"]),
       ("integrated",source.parent/"integrated",[sys.executable,"-u","-m","tools.bench_fast_models","--models","quran-phonetics","whisper-tarteel-tiny","whisper-tarteel-base","--surahs","1","32","55","59","65","73","77","80","98","112","2"])]
for name,folder,args in cases:
    env={**os.environ,"PYTHONPATH":str(folder)}
    print("start",name,flush=True)
    with (task/(name+".log")).open("w") as log:
        result=subprocess.run(args+common+["--output",str(task/(name+".json"))],cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT)
    print("completed",name,"exit",result.returncode,flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
