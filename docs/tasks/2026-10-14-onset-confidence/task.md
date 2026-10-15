# Отображение акустической уверенности начала

Цель: устранить misleading низкие проценты возле правильно найденных начал.
Различать качество услышанного слова и поддержку звука в позиции начала;
не выдавать некалиброванное число за вероятность правильного времени.

Прочитано: AGENTS/CLAUDE/RTK; task/report FULL2:220; align/onset checker,
word_align, writer, alignment_runs, API/get_sura, UI/timingConfidence,
web-ui-ux/better-writing. Codebase-memory MCP недоступен; Serena.

Ограничения: времена, текст, число аятов, БД/ручные статусы и FAST алгоритм
не менять. Сохранить существующие acoustic guards и первичный confidence.
Не коммитить/пушить. Открытые чужие изменения сохранить.

У пользователя запрошено уточнение смысла показателя. Независимая работа:
проследить writer-reader, определить отображение evidence без повторного
прогона пользовательской БД. Шкала не калибруется по эталону.

- [x] Уточнить/выбрать смысл показателя и сформулировать UI.
- [x] Реализовать раздельные данные начала/слова без новых DB columns.
- [x] Проверить mismatched/stale/manual записи и реальные2:219/220.
- [x] Проверить UI и обновить документацию.

Решение: read-only разделение first_token_conf иword conf; новое число
называется«Звук начала», не калиброванной уверенностьюграницы. Существующие
бюджеты/пороговыеguards/БД/времена не менялись. Источникevidence проверяется
поaudio/time/source/conf; moved/manual/stale/conflicts не наследуютscore.
Результаты/ограничения: report.md. 84целевыхpassed,Nodepassed,Ruff/JS/diffclean;
286реальныхметок прочитаны, hashБД неизменен. Требуется перезапускGUI.
Docs updated: task.md,report.md,docs/STATUS.md.

Общийsuite:676passed/8skipped/1failed вFASTsampled-onsetfixture (actual==requested
24084), изолированно воспроизведён. Не исправлялся внеscopeconfidence.
