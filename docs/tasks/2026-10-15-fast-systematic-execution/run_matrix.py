"""Sequential corpus stages; child run.py persists each launch and completion."""
import subprocess
import sys
from pathlib import Path

P = Path(__file__).resolve().parent
stage = sys.argv[1]
models = ['--models', 'quran-phonetics', 'quran-phonetics-fp32']
commands = []
if stage == 'transfer':
    for name in ['saad_al_ghamidi', 'mohammed_siddiq_al_minshawi',
                 'mahmoud_khalil_al_hussary', 'abdul_basit_mujawwad']:
        for variant in ['baseline', 'prototype']:
            commands.append([f'{variant}-{name}', '--variant', variant, *models,
                             '--surahs', '80', '98', '112', '--reciter', name])
elif stage == 'after-ae':
    commands.append(['after-a', '--variant', 'prototype'])
    commands.append(['after-e', '--variant', 'prototype', *models,
                     '--surahs', *map(str, [9, 42, 83, 95, 109, 113, 114])])
elif stage == 'wide':
    for variant in ['baseline', 'prototype']:
        commands.append([f'wide-{variant}', '--variant', variant, *models,
                         '--surahs', *map(str, range(1, 115))])
elif stage == 'repeat':
    for variant in ['baseline', 'prototype']:
        commands.append([f'repeat-{variant}', '--variant', variant, *models,
                         '--surahs', '112', '112', '112', '114', '114', '114'])
else:
    raise ValueError(stage)
for args in commands:
    print('start', args[0], flush=True)
    result = subprocess.run([sys.executable, '-u', str(P / 'run.py'), *args])
    print('end', args[0], result.returncode, flush=True)
    if result.returncode:
        raise SystemExit(result.returncode)
