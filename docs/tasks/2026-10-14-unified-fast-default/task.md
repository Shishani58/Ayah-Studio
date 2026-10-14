# Штатный единый FAST

2026-10-14. Пользователь поручил запушить все накопленные изменения, затем
перевести приложение на единый FAST вместо legacy. Первый snapshot запушен
в origin/main: aac8f3a. Уточнение пользователя явно выбирает новый FAST.

Прочитаны AGENTS.md/CLAUDE.md и RTK.md (правила), docs/project.md (пайплайн,
инварианты/идентификаторы), fast-review-fixes/report.md и independent-review.md
(ограничения), публичные FAST/auto/GUI/CLI entrypoints и их regression tests.

Скоуп: штатный FAST GUI/CLI/API и auto, если он выбрал FAST, используют один
fast_executor. После вопроса пользователя о необходимости legacy он удалён
из runtime: результаты сравнений и предыдущий код сохранены в git aac8f3a.
Общие phonetics helpers, бюджетный ledger и regression coverage действующих
путей сохранены; исчезают tests только удалённого matcher. Bundle фиксируется
до decode. Auto не меняет доказанные FAST метки
последующей lag-калибровкой/полным repair. Полный режим и данные не меняются.
Внешние аятные клипы не поддерживаются новым FAST; отдельный fingerprint
режим сохраняется. Ручные edited/verified, форма БД и acoustic budgets прежние.

Решение пользователя об активации не означает прохождение M5: coverage FAIL
и Base79:26 +508ms сохраняются как измеренные ограничения. ASR timing=False.
Исторические frozen artifacts не переписываются и не rekeyed.

- [x] Запушить накопленные изменения после pytest/Ruff
- [x] Переключить все штатные FAST entrypoints
- [x] Удалить legacy matcher/Tarteel loop/readers и ненужные импорты
- [x] Проверить статические зависимости: FULL не использует удалённые matcher/readers
- [x] Финальная проверка отсутствия fallback и postprocessing FAST
- [x] Проверить нормальный pipeline → transaction/manual protection/identity
- [x] Свежие локальные реальные пробы штатного пути
- [x] Полный pytest/Ruff, обновить docs/STATUS и результат задачи

Уточнение пользователя: тесты запускать только в самом конце. После этого
уточнения промежуточные suite не запускались; итоговая проверка будет одной
фазой после всех кодовых изменений, включая удаление legacy.

Локальная macOS сборка готовится в build/unified-fast-dist через текущий
surah_cutter.spec. До проверки frozen bundle прежняя dist/Ayah Studio.app
сохраняется. DMG/релиз/Windows assets этой задачей не публикуются.

## Результат

Финальная проверка выполнена на изолированном snapshot aac8f3a + собственные
изменения этой задачи: параллельные незавершённые FULL изменения не включены.
595 passed, 7 opt-in audio skips, Ruff чистый. Удалены тесты только старого
matcher; шесть проверок общих phonetics helpers сохранены, cancellation/budget
регрессии переведены на действующий LocalEvidenceReads.

Штатный process_surah(mode=fast), без experimental флага: шесть реальных
случаев108/109 × CTC/Tiny/Base,52actual dispatches с проверкой PCM/SHA/ledger
и budgets. Среди подтверждённых ошибок>500ms нет; обе ASR модели подтверждают
109:1–6. Во всех шести временных проектах edited/verified строки и provenance
сохранены, новый результат пишет current fast_run_key. Пользовательские БД
не использованы для записи;49protected assets неизменны.

Локальный macOS bundle пересобран из того же snapshot. Frozen --self-check:
114/6236, ONNX/Silero/FFmpeg/assets OK; codesign --verify --deep --strict OK.
Проверен PYZ: unified executor включён, legacy module/matcher/readers отсутствуют.
Обновлён dist/Ayah Studio.app; прежний bundle сохранён в
build/Ayah Studio.before-unified.app. Native GUI запуск в этой задаче не проверялся.
Измеренные M5/ASR timing ограничения сохраняются; legacy удаление их не решает.

Docs updated: docs/project.md, docs/STATUS.md, этот task.md,
verify_default.py, verification.json, summary.json.

Блокеры: активация авторизована; M5 и калибровка ASR timing остаются открытыми
вопросами качества, не скрываются при смене штатного entrypoint.
