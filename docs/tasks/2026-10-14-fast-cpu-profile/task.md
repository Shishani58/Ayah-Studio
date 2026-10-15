# FAST: измерениеCPU перед оптимизацией

Предыдущий turn —progress:17native left-context attempts, отвергнут global midpoint.
Цель: найти измеренное лишнееCPU work в FAST, затем сохранить доказательства
и реальную dispatch sequence при оптимизации. Accuracy thresholds неизменны.
Прочитаны fast_executor function/call inventory, named_alif_probe driver,
RTK instructions. Frozen sparse15 corpus2 —исходный контроль.

- [x] cProfile fresh full2 run, actual local reads through existing ledger.
- [x] Guard source/model/audio/reference identity and exact rows/dispatch preservation.
- [x] Рассмотреть только measured hotspots; не оптимизировать worker wait.
- [x] Если hotspot meaningful —isolated benchmark, parity, prospective controls.

Тесты только в финале цели. Full2 означает DSP на всей записи, acoustic только
разрешённые локальные окна; no whole model pass. Production unchanged до evidence.
Docs updated: task.md.

Profile -m cProfile loaded cwd sparse10, not intended sparse15; preserve
profile as actual sparse10 experimental-adapter run (5proofs/23calls), not paired
with sparse15. Identified1169whole-audio RMS computations,36,76s cumulative.
Isolated before/after current main snapshots: compute identical float32RMS
once per pipeline call; _dsp_pause_metrics retains standalone fallback.
Next: full12actual DSP decisions equality + timings, then production promotion.

Result: production RMS reuse,1697exact decisions across12 + full2 exact
rows/dispatch/proof preservation. Prepared regression, pytest notrun.
Docs updated: report.md/task.md/docs/STATUS.md/docs/project.md.
