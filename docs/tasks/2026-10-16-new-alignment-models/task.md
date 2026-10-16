# Новые модели выравнивания

Цель: добавить MFA Quran Hafs, Zipformer Quran v3 и Muno FastConformer,
проверить реальные аятные/пословные результаты на одинаковом аудио.

Прочитано: AGENTS/RTK, docs/project.md (pipeline, catalog, FAST), STATUS,
model-runtimes/task, asr-model-selection/report; model_store/runtime,
word_align/align и карточки upstream (форматы, лицензии, ограничения),
официальный MFA align_one и его фактические tokenizer/dictionary loaders.

Решения: отдельный MFA adapter, pinned revisions/checksums; defaults/FAST и
пользовательские БД не менять. Существующие незакоммиченные изменения сохранены.
Новый registry содержит три строки; обе gated модели недоступны для выбора
до полноценного подключения/проверки, не выдаются за рабочие.
MFA требует отдельное Kaldi/MFA окружение; установлен native MFA3.4.0,
сохранён explicit conda lock для osx-arm64. Text lookup копируется в NFKC
симметрично transcript/lexicon, canonical owners и фонемы не меняются.
MFA confidence некалиброван, поэтому все результаты имеют confidence=0/review.
Muno запланирована через существующий NeMo путь (459 МБ), компактный ONNX
не реализован; известный ml-dtypes конфликт должен быть решён до допуска.

- [x] Три pinned записи каталога, локальный HF login без сохранения токена
- [x] MFA FULL/words adapter, cancellation/timeout и настоящий probe
- [x] MFA веса/словарь/native окружение установлены на этом Mac
- [x] Шатри1/112/114 ×QPC/MFA: реальные FULL и слова первого аята
- [x] Дополнительный штатный mfa align на112: также смещения; отдельная
      fMLLR adaptation фактически не подтверждена, преимущество не заявлено
- [x] 152 целевых теста (включая ручные метки/commit contract), Ruff, diff-check
- [ ] Muno: получить доступ, запустить и проверить адаптер
- [ ] Zipformer: получить доступ, реализовать адаптер по реальным файлам
- [ ] Полное парное сравнение всех трёх кандидатов

Результаты и команды — report.md, shatri-comparison.json, final-smoke.json.
Основная модель сохранила преимущество; MFA112 max2810мс, все метки review.
Ручного эталона каждого слова нет, точность слов не заявляется. Скорость
не обобщать: на компьютере параллельно исполнялись другие задачи.

Блокеры: anonymous HEAD Zipformer/Muno →401 GatedRepo; локального HF token
на старте нет. Пользователю отправлены страницы для принятия условий и просьба
выполнить локальный HF login; ответа/доступа пока нет. Условия/передачу личных
контактов за пользователя не принимали. Без доступа эту часть не завершить.
Windows/Intel и собранная .app не проверялись. Для текущего GUI нужен перезапуск
после обновления исходников; пользовательские проекты/настройки не затронуты.

Docs updated: docs/project.md, docs/STATUS.md, task.md, report.md и артефакты рядом.
