# FAST: left context у12native onset границ

Предыдущий turn —progress: attribution294 и9fresh seam/control calls.
Цель: выяснить причины ошибок experimental native_onset, сохранив удачные случаи.
Прочитаны temporal-attribution report/source, docs/project.md FAST semantics.

- [x] Проверить текущие12native proof и hashes retained corpus.
- [x] По одному независимому2,5s left-context read на boundary, через ledger.
- [x] Полный canonical previous last word и scalar argmax endpoint; никаких
      подтверждений по order/partial tail, никаких replacements canonical text.
- [x] Mixed-run counterfactual с retained right proof: отдельно от fresh joint.
- [x] Записать выводы и следующий justified experiment; production unchanged.

Тесты только в общем финале; no commits. FAST whole acoustic pass запрещён,
all text reads≤6s; SQLite read-only. Docs updated: task.md.

Initial12 completed,7full tail matches. Predeclared extension gate: full
previous word/minquality≥.25 и observed_right_margin≤2stride (+float epsilon).
Selected2:10/18,55:26; each new2,7s crop adds200ms into DSP pause, no acceptance.

Right-edge3controls stable endpoints within1,5ms, qualitystill≥.25.
Global CTC-gap midpoint rejected:55:26error251→-971.
Left-edge predicate-only audit:1:5 and2:9 begin with exact≥4unit suffix of
previous canonical word/min≥.25, allowing bounded4s previous-context diagnostic.
Others lack explicit matching left-edge suffix; no automatic expansion for them.

Completed analysis:17freshcalls,9/12fulltails; report.md.
Global CTC-gap time replacement rejected on good controls. Production unchanged.
Docs updated: task.md/report.md/docs/STATUS.md; project.md unchanged.
