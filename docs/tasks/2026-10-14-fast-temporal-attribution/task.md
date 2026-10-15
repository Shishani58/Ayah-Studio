# FAST: attribution временных ошибок

Предыдущий turn — progress: fresh12/294proofs, сохранены все273 старых строки.
Цель: установить, что именно измеряется midpoint и native phoneme onset,
не подгонять timestamp к эталону. Production code unchanged до диагностики.
Прочитаны docs/project.md (контракт FAST/temporal semantics), pause_geometry
и отчёт late-future-anchor (два новых outliers и сохранение прежних результатов).

- [x] Read-only сверка294подтверждений: signed GT, pause location, read geometry.
- [x] Раздельная оценка pause/native/wasl и допустимости midpoint интервала.
- [x] Сохранить источники, скрипт, таблицы; выбрать следующий эксперимент по причине.

Не читать всю запись текстовой моделью. SQLite read-only, Quran text immutable.
Тесты только при окончательном завершении цели. Docs updated: этот task файл.

Predeclared control: все5remaining accepted midpoint паузы шириной>1000мс
в retained corpus, не только ошибочные. Конечный crop≤6s, праваячасть
ограничена остатком бюджета. Не подтверждать при incomplete current/previous.
CTC-gap midpoint —counterfactual, endpoint не приравнивать физическому speech end.

Result: report.md.9fresh local calls/0whole passes. Глобальный native onset
перенос отвергнут, joint CTC-gap показывает и улучшение и регрессии.
Docs updated: task.md/report.md/docs/STATUS.md. Production unchanged.
