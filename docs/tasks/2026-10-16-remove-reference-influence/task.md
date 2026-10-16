# Полностью исключить эталонные тайминги из приложения

Прямое поручение пользователя во время исследования моделей: убрать любое
влияние эталона и закрепить правило. Исследование продолжается на frozen
исходниках с отключённым prior; оно не переписывает рабочие файлы.

Прочитано: AGENTS/RTK, project/STATUS; align_tiled/reference_prior,
timing_accuracy/probability/calibration, onset_evidence, result_source;
references/eval, packaging spec/pyproject, связанные тесты и benchmark tools.

Решения: удалить reference prior; убрать reference-derived confidence artifact;
вынести reader/evaluator в tools/research; исключить БД изwheel/PyInstaller.
Ручные/пользовательские/эталонные БД и канонический текст не менять.
Новые времена исследованием в пользовательские БД не сохранять.
Правило явно распространяется на косвенные priors и калибровки.

- [x] Синхронно обновить AGENTS/CLAUDE.
- [x] Удалить runtime зависимости и package assets.
- [x] Обновить проверки, документацию и provenance.
- [x] Проверить FULL/tiled/FAST, сохранность ручных данных, сборочный пакет.

Дополнительно удалена CTC−200ms: комментарий документировал подбор по
размеченному чтецу. Env override не возвращает её. Lag0 и новый FULLpolicy
попадают в provenance; автосохранённые результаты сами не переписываются.
Старые metadata/protected manual сохраняются; writer больше не имеет
отдельного reference-режима. Архив calibration/prior сохранён вне приложения.

Проверка:96 целевых,967 общихpytest/10skip;4 реальных112/114 FULL/FAST.
Первый общий запуск выявил старое ожидание lag200 в synthetic test; исправлен
на фактический5200ms без компенсации. Исторические audio bounds относятся к
lag200: evaluator добавляет ровно200ms по triangle inequality, не подгоняя
runtime. Эти оценки не означают физическую точность нового lag0. Сохранены
первоначальные отказы и финальные логи. Ruff/sync/diff OK.
Wheel88files проверен без reference assets/readers/calibration/evaluator;
канонические данные присутствуют. Native PyInstaller не собирался, spec очищен.
95 защищённыхdata/audio/weights hashes прежние. ПользовательскиеБД не менялись.
Старое открытое приложение требует перезапуска/новой сборки.

Дополнительная проверка обнаружила чтение старых процентов в GUI. Удалены
значки/сортировка/фильтр по такой вероятности и подсказки о ней; старые данные
с другой целью не влияют на очередь. Из runtime-каталога и его переводов
вынесены reference-derived benchmark числа и сравнительные описания. Они
сохранены во внешнем historical/. Новый browser regression подаёт старые
значения0/23/98/100% и подтверждает их полное игнорирование.

Docs updated: AGENTS.md, CLAUDE.md, docs/project.md, docs/STATUS.md, task.md.

Финальная повторная проверка после GUI/catalog:975 passed/10skipped. Два
отказа первого общего запуска относились к одновременно изменявшимся playback
тестам; финальный общий запуск проходит.39 целевых также прошли. Пересобран
и заново проверен wheel88files: GUI/catalog без reference-derived данных;
95 защищённых файлов прежние, AGENTS/CLAUDE идентичны. Native build не собирался.

Аудит документированных fitted constants также нашёл recheck margin0.10:
код прямо называл его подобранным на размеченном чтеце. Порог удалён; теперь
предупреждение выдаётся при строго лучшем акустическом совпадении соседнего
начала. Это может увеличить число предупреждений; recheck по-прежнему ничего
не записывает и не двигает времена. Исправлен устаревший комментарий tiled.

Дополнительно исключены эталонные data/references, внешний reader/evaluator
tools/research и исследовательские docs/tasks (включая архив старого кода)
из sdist. uv build --offline успешно построил sdist и wheel из него. Проверены
206/88 файлов: эталонов/readers/calibration нет; единственная БД в wheel —
канонический Quran text. Native сборка не запускалась. distribution-audit.json.
Последний общий запуск после recheck:976 passed/10skip, один отказ старого
full_resume fixture без review_meta.confidence; этот read/resume код здесь не
менялся.25 целевых recheck/reference проверок прошли. Логи всех отказов сохранены.

Итоговый общий запуск после завершения параллельных правок:983 passed,10skipped
(pytest-last.log). Ранее отмеченный независимый full_resume отказ больше не
воспроизводится. JS syntax app/i18n/models и исследовательский Ruff проходят.
