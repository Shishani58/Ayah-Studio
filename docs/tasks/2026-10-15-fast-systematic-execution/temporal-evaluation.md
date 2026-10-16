# Временная ветка M4

Результат: новый estimator не принят. Доступная работа выполнена:
[протокол](annotation-protocol.md),18WAV+SVG и пустые labels в
[пакете](annotation-package/labels.csv), original samples/SHA в
annotation-private-manifest.json. annotations.log фиксирует завершение генерации.
**Фактического прослушивания и human labels нет.**

Для нового short-context prototype время — прежний центр той же измеренной
паузы. Внутреннее слово или recoveredq не создаёт новое начало речи. После
получения predictions benchmark открывает read-only reference; это проверяет
расхождение с packagedtime, а не физический onset/конец предыдущего слова.
Семантика playback boundary и physical onset разведена до выбора estimator.

Нет нового причинного основания повторять уже неудачные VAD midpoint,
energy midpoint, CTC-gap midpoint и inherited attentionheads. Они не включены.
Надёжного метода local ASR timing на текущих данных не установлено;
Tiny/Base timing=False сохранён. Неограниченный acoustic pass запрещён.

Известный Base79:26+508ms без review воспроизведён fresh baselineA и остаётся
отдельным долгом. Ни высокая tokenprobability, ни successfulidentity не названы
доказательством точности времени. Прототип не является исправлением этого долга.
