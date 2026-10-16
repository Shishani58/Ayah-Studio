"""Render the completed Base-only change from final recorded evidence."""
import json
from pathlib import Path
P = Path(__file__).resolve().parent
def load(name):
    return json.loads((P / name).read_text())
gate, costs, gains = load('final-gate.json'), load('costs.json'), load('causal-gains.json')
assert gate['passed'] and load('production-smoke-gate.json')['passed']
assert load('final-data-guard.json')['passed']
assert load('final-pytest-completion.json')['exit'] == load('final-ruff-completion.json')['exit'] == 0
model = 'whisper-tarteel-base'
before, after = costs['wide-baseline']['models'][model], costs['wide-prototype-c']['models'][model]
direct = sum(g['kind'] == 'direct_same_window_extension' for g in gains)
lines = ['# Рабочий FAST на одной Tarteel Base', '',
    'Перенесён F: `sequential-fast-v3-sparse-13-local-prefix-review`.',
    'После прямого указания пользователя все новые акустические запуски — только',
    'Tarteel Base с одними весами. Предыдущие результаты других моделей — история.', '',
    '## Изменение алгоритма', '',
    'Сильная целая короткая ASR-единица внутри неполного канонического слова',
    '(`يا` в `يا ايها`) разрешает конечное дочитывание того же окна. Подтверждение',
    'требует полной глобально различающей фразы и прежнего minQ каждого essential',
    'token >=0,6. Новая возможность не применяется к паузе с радиусом>500мс.',
    'Неудачное дочитывание возвращает исходные кандидаты, сохраняя старый сильный',
    'prefix block. Canonical текст/индексы и validator не ослаблены.', '',
    'Текст и время разделены. Если минимальный crop уже содержал полную sparse-фразу,',
    'но её первая ASR-единица была слабой, рост score на большем crop не доказывает',
    'начало. При отсутствии стационарной тихой DSP-паузы время остаётся review/conf0,',
    'а доказанный owner сохраняется для следующего поиска. Новых численных порогов,',
    'нового timestamp или дополнительных model calls этот temporal guard не вводит.', '',
    '## Результат', '',
    f"Итог: {gate['pairs']} пар Base, **+{gate['added']} подтверждений**, "
    f"{gate['lost']} потерь прежних и {gate['new_over500']} новых confirmed-расхождений>500мс.",
    f"Прямой эффект дополнительных чтений: {direct}; последующее восстановление: {len(gains)-direct}.", '',
    '| Шатри114 / 6236 аятов | До | После F |', '|---|---:|---:|',
    f"| Подтверждено | {before['confirmed']} | {after['confirmed']} |",
    f"| Подтверждено и reference ≤500мс | {before['confirmed_reference_within500']} | {after['confirmed_reference_within500']} |", '',
    'Дополнительный перенос — свежие65/73 Гамиди и Миншауи; Хусари — контроль без потерь.',
    'Все новые границы, actual cue/proof/read IDs и индивидуальные расхождения:',
    '[causal-gains.json](causal-gains.json).', '',
    'D дал47новых на Шатри, но84:10 имел903мс расхождения; D отклонён.',
    'F оставляет эту метку для ревью и сохраняет следующие правильные границы.',
    'Прежние28:4/41:3/72:6 не демотируются: сильное первое слово либо стационарная',
    'пауза. Это консервативный отказ по времени, а не human-калибровка onset.', '',
    '## Проверка', '',
    'Исходный и D acoustic runs Base114 свежие. F исполнен на всех114реальных',
    'последовательностях D observations: каждый PCM SHA, размер, порядок и окно',
    'совпали. Различие — публикация неподдержанного времени84:10. Это **replay,',
    'не свежий acoustic pass F114**. Затем выполнены свежие Base F warm3повтора65/73,',
    'три других чтеца65/73 и7случаев рабочего кода:28/41/65/72/73/82/84.', '',
    '- Общий финальный pytest: [final-pytest.log](final-pytest.log).',
    '- Scoped ruff: [final-ruff.log](final-ruff.log).',
    '- Регрессии: неполный/чужой текст, слабый token, широкая пауза, сохранение',
    '  repetition block, JSON proof, слабое начало/тихая пауза и продолжение',
    '  поиска после identity-only.',
    '- Рабочий executor равен F кроме VERSION;7fresh live rows/PCM inputs совпали.',
    '  Whole-file guard этого запуска завершился exit1 из-за чужой правки только',
    '  FULL align_surah во время вычислений. Исходный отказ сохранён, FAST AST/bytes',
    '  отдельно сверены.2fresh Base случая65/84 на неизменяемой копии текущего',
    '  кода завершились exit0, также повторив F. [production-smoke-gate.json](production-smoke-gate.json).',
    '- Чужие FULL/каталог правки сохранены; Base entry и relevant FAST функции',
    '  сверены. [foreign-source-acceptance.json](foreign-source-acceptance.json).',
    '-230canonical/reference/audio файлов проверены SHA, target tables прежние;',
    '  AGENTS/CLAUDE идентичны. [final-data-guard.json](final-data-guard.json).', '',
    '## Цена', '',
    f"Шатри114: calls {before['actual_dispatches']}→{after['actual_dispatches']}; "
    f"model input {before['model_input_ms']/1000:.2f}→{after['model_input_ms']/1000:.2f}с.",
    'F не добавляет модельных чтений относительно D. Лимиты6с/65%/200%/32MiB соблюдены.', '',
    '| Base, тёплая медиана executor,3повтора | До,с | После F,с |', '|---|---:|---:|']
for s in ['65', '73']:
    old = costs['warm-baseline']['models'][model]['warm_repeats'][s]
    new = costs['warm-prototype-c']['models'][model]['warm_repeats'][s]
    lines.append(f"| {s} | {old['median_executor_s']:.3f} | {new['median_executor_s']:.3f} |")
lines += ['', 'Ускорение не заявлено: присутствовала другая CPU-нагрузка, wall нельзя',
    'целиком приписывать алгоритму. Loading/synthetic probe, decode/DSP, model/',
    'остальная работа, inputs/union/cache, warm ranges и одновременная сумма RSS',
    'дерева с sampling gaps — [costs.json](costs.json). Wide timers/RSS измерены D;',
    'F повторяет его входы. Wide process включал исторический Tiny-этап до выбора',
    'единственного Base; per-model поля здесь отфильтрованы по Base. Warm/новые',
    'cross/production запуски — только Base.', '',
    'Исторические имена prototype-c сохранены: effective wide=D, warm=F.',
    'Actual argv/cwd/import SHA записаны. Source архивы и SHA идентифицируют веса;',
    'новые модели не скачивались.', '',
    '## Пределы', '',
    'Прежние sparse-owner и midpoint ошибки остаются в baseline; весь FAST не',
    'объявлен точным или прошедшим общий production M5. Score текста не является',
    'вероятностью точного времени. Фактического прослушивания/human labels нет.',
    'Pre-pause Base-проба8reads отрицательна и не перенесена. Reference не выбирал',
    'окна, owner, пороги или время. Ручные edited/verified защищены прежним writer;',
    'схема БД не менялась. Коммит/пуш не выполнялись.']
(P / 'report.md').write_text('\n'.join(lines) + '\n')
print('Rendered completed Base-only implementation report')
