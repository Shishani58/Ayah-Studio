# Identity, search cue и время: аудит текущих путей

Общий вход — неизменённое независимое observation из одного read-ID. Качество —
минимум essential units, а не среднее obs.quality. Пороги читаются из Policy:
CTC .25 / ASR .6; posterior prefix .9, expansion .1, beam log margin log9.
Это разные шкалы, ни одна не является вероятностью ошибки времени<500ms.

| Путь | Допустимый вывод | Соперники/essential | Время и validator |
|---|---|---|---|
| Greedy | Полная запланированная фраза | Все canonical word starts; минимум всей фразы | DSP geometry midpoint или CTC positions; validate_boundaries |
| Beam | Независимая полная альтернатива | Независимые beam hypotheses и margin; минимум units | Те же ограничения и replay beam в validator |
| Posterior | First-entry prefix от начала crop | Canonical rivals, blank, unrestricted residual; prefix_path essential | Только существующие разрешённые frames; validator сравнивает сохранённые scalar comparisons |
| Contextual seam | Совместный previous-tail + shorter opening | Симметричный plan_seam, отдельное качество обеих частей | Два read-ID для pause, один совместный CTC onset для wasl |
| Noisy opening | Identity anchors с отсутствующей первой единицей | Консервативное сравнение всех владельцев/внутренних вхождений | Только пауза; не first_onset/wasl, пропущенный onset не реконструируется |
| Split ASR opening | Два целых слова с исходными буквами | Все canonical starts с симметричной segmentation; исходные selected BPE contributors | Только DSP midpoint, independent segmented replay validator |
| Body cue | Только адресная head recheck | Уникальная сильная пара внутри owner, не alias | Не порождает boundary; новый opening read обязан пройти обычный proof |
| Sparse anchor | Уже услышанный уникальный future opening | Глобальный canonical owner, пропуски остаются review | Собственная принятая пауза/read-ID; validator проверяет anchor |
| Wasl | Tail+opening в одном локальном crop | Unique seam и minimum quality всех units | Требует timing=True и сохранённые позиции; Tiny/Base timing=False |

При first_onset у ASR полный текст сам по себе не даёт boundary. Первичный отказ
от времени не равен отказу распознавания; такие случаи выделяются в cause artifacts.
Сильный previous-body rival и противоречащий alias нельзя переименовать в opening.
Внутренний prefix event отличается от first-entry; максимум по crop offsets —
диагностическая верхняя возможность, без нового null/multiple-comparison контракта
он не принимается за posterior proof.

H5: groups участвуют в lexical matching, estimates и tempo. Связь существует
структурно и исторически подтверждена, но в этой задаче representation не менялся.
Без новой причинной A/B миграция групп/tempo не нужна и не переносится.
Run-key уже содержит algorithm/adapter/policy/loaded code/PCM/vocab/runtime/DSP;
при неизменном runtime новых версий формата/БД не требуется.

JSON/forged-proof/run-key/budget/cancel/error/cache/manual persistence регрессии
существуют в общей suite и будут запущены только в M7. Они проверяют контракт,
не физическую точность модели. До измерений новые общие правила не принимаются.
