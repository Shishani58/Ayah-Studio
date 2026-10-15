# FAST: paired sampled-onset controls и повторные чтения

Предыдущий turn: progress — исправлен реальный sample-origin баг, prototype194proofs/0loss.
Цель продолжается. До изменений: читать current executor/reads, bench и прежние отчёты.
Скоуп: свежая пара before/after поддерживаемых моделей на agreed10+long2;
затем причины повторных чтений, не снижение порогов. FAST6s/65%/200%,
канонический текст и ручные БД сохраняются. Все source/results/logs фиксировать.
Pytest только в общем финале, сейчас не запускать.

- [x] Frozen snapshots current + восстановленный до onset-fix.
- [x] Serial paired3models×11surahs×2, hashes/dispatch/reference invariants.
- [x] Сравнение потерь, additions, timestamps, скорости, прочитанных интервалов.
- [x] Причины повторов: early-stop отвергнут из-за реального proof55:30.
- [x] Изолированный32 prototype: первое полное название алиф из штатной
      character mapping, все проверки слова/phone provenance; production не менять.
- [x] 32:11 outlier1867ms: GT в нулевом сигнале, decoded onset рядом с
      началом речи; raw metric/сигнал/proof/audio сохранены, не скрывать.
- [ ] Следующий этап: длинная2, preservation12cases, named-name declaration
      generality/production integration; общий финальный pytest ещё впереди.

Docs updated: этот каталог, STATUS при завершении. Commit/push/DB writes нет.
