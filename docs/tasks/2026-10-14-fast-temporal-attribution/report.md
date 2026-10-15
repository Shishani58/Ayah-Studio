# FAST: временная атрибуция и контроль широких пауз

Production code unchanged; тесты не запускались. Read-only анализ fresh sparse15
полного согласованного корпуса12сур/607строк, из которых294подтверждены.
Подтверждённые строки не описывают точность всех607меток: остальные требуют ревью.

## Что показала сверка

294proofs: median absolute73,5ms, p95 390,35ms,13outliers>500ms.
268обычных midpoint: median64,5ms, p95 271,15ms,4outliers.
Контрфактическая глобальная замена на independently observed first phoneme:
median202,5ms/p95 541,5ms/16outliers. Такой перенос отвергнут.
12experimental pause_native_onset: median996ms,9outliers.
13wasl: median149ms,p95 225,8ms,0outliers. First_onset1:262ms.
Сравнения относятся к этому чтецу/корпусу, не всему Корану.

Эталон вне DSP-интервала у16обычных midpoint. Среди7accepted пауз шириной>1000ms
триoutliers; среди261остальной паузы один. Nonstationary сам по себе не означает
ошибку:154такие паузы имеют median74,5ms и только3outliers. Нельзя просто
запретить все reverberant pauses либо калибровать границы по известному GT.

## Независимые локальные попытки

Первый seam diagnostic2:53/80:37: два свежих чтения5,22/5,06s через ledger,
без forced target; полностью прочитаны previous last word и current distinct phrase.
Второй endpoint diagnostic повторил те же два чтения, сохранив raw argmax runs.
Dispatch hashes и posterior hashes совпали с первым прогоном. Только scalar
frame endpoints сохранены; raw matrices в файлы не выгружались.

2:53: previous last argmax support end969708, current first support971314;
CTC-gap midpoint970511 vs reference970114: +397ms вместо+576ms.
80:37: corresponding155548/157154, midpoint156351 vs reference155938:
+413ms вместо+672ms. Это гипотеза, не новые подтверждённые границы.
Последний token support не равен физическому окончанию речи/эха.

Предварительно зафиксирован контроль всех5остальных широких midpoint:
2:21/26/29/44 и77:25. Ещё5fresh calls, каждый≤6s. Полный joint greedy proof
с обеих сторон получен только2:29: ошибка291→251ms. Остальные incomplete
сохраняют прежнюю границу. Это важное ограничение совместного окна.

Отдельная mixed-run counterfactual сверка fresh left endpoint с retained right proof:
2:21 639→529ms,2:26 abs172→182ms,2:44 abs316→486ms.
Она показывает риск регрессии, не является prospective результатом новойpolicy.
77:25 previous last word не прочитан полностью; изменений нет.
Статический previous connected target может отличаться приwaqf; не удалять
согласные, не подтверждать неполный хвост. Native_onset outliers остаются нерешёнными.

## Решение и следующий шаг

Глобальную замену midpoint отклонить. Глобальный запрет nonstationary пауз
не обоснован. Joint CTC-gap candidate интересен лишь при независимом полном
previous/current proof, но нуждается в prospective executor guard, alternate
pronunciation attribution и проверке всего корпуса. Public activation пока отсутствует.
Главная нерешённая группа —12experimental native_onset: нужна отдельная left-tail
проверка и явное разделение text identity от temporal assurance.

## Проверка и артефакты

analyze.py,attribution.json/csv —все294строки, GT signed, interval locations,
source/reference hashes. local_seam_probe.py иlocal_seam_endpoint_probe.py:
4fresh calls. wide_pause_control_probe.py:5fresh calls. JSON/log каждого прогона,
seam-counterfactual.json,wide-controls-counterfactual.json сохранены.
Каждый вызов проверен dispatch_spy/verify_dispatch; model worker released,
reference DB read-only. Scoped Ruff всех4скриптов clean; pytest не запускался.
AGENTS.md/CLAUDE.md идентичны. No commits/push.
Docs updated: task.md,report.md,docs/STATUS.md. No production edits this turn.
