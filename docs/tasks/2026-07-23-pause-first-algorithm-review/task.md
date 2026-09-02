# Pause-first algorithm review

## Goal and scope

Improve the general fast forced-alignment algorithm on Abu Bakr Ash-Shatri
audio.  Every detected ayah start must use a pause when a pause exists; the
only exception is a proven wasl/merged pair, which must be aligned as a whole
interval.  The matcher must not silently advance past an unproven current ayah:
it must classify the cause and widen the read, resolve a merged interval, or
diagnose the rejected pause.

## Read documents

- `docs/project.md` — pipeline, output invariants, and target accuracy.
- `docs/STATUS.md` — current fast-mode contract and existing diagnostics.
- `AGENTS.md` / `CLAUDE.md` — correctness, minimal-change, task-file, and
  validation rules.

## Decisions

- Ground truth: `/Users/aabibulat1/Developer/quran/quran_flutter/server/reciter_timecodes/abu_bakr_ash_shatri.db`.
- Acceptance metric: retain the documented corpus-level guard (median absolute
  start error <150 ms and p95 <500 ms), and add a pause-contract audit: every
  non-wasl accepted start is backed by a selected pause.
- Train and diagnose on medium Shatri suras first; verify any general change on
  an independent medium control set before broad claims.
- A future ayah read is evidence of an unresolved current interval, not a
  licence to skip it.  The algorithm must choose one reason: insufficient read
  window, merged/no-pause interval, or wrong/rejected pause.

## Checklist

- [x] Inspect the current fast matcher and its trace contract.
- [x] Confirm the supplied reference schema and select medium audio fixtures.
- [x] Measure baseline and classify unresolved/current-gap cases.
- [x] Implement a general resolution path and focused regression tests.
- [x] Re-run target and control benchmarks; update status/task results.

## Verification

Confirmed root cause on Shatri 89:30: a `micro-tempo` internal valley with
`sim=0.191` was accepted as a fallback start even though the current ayah was
not recognized.  The true start had no inter-ayah pause, so its local wasl run
needed continuous forced alignment.  Rejecting every low-sim `micro-tempo`
fallback leaves that interval in `wasl-run`; the resulting start is 165896 ms
against the 165971 ms reference (75 ms error; previously 1683 ms).

The same audit exposed the general version of that mistake on Shatri 83:2:
an `intro-vad-tight` fallback with `sim=0.141` was selected at 5.80--6.36 s,
then refinement moved the final marker to 9.449 s, outside its claimed pause.
Fallback now requires the same minimum recognition score for every pause tier.
The wrong pause is therefore unresolved, then solved by pause/wasl recovery at
9.330 s (4 ms from the 9.334 s reference), with the final marker inside its
selected pause.

Verified:

- `uv run pytest tests/test_align_fast.py -q` — 98 passed.
- `uv run pytest tests/test_align_fast.py tests/test_pipeline.py -q` —
  100 passed.
- `uv run ruff check surah_cutter/core/align_fast.py tests/test_align_fast.py`
  and `git diff --check` — passed.
- `uv run python -m tools.bench_fast fast-only 080 083 087 089` — 127 ayahs:
  median 75 ms, p90 194 ms, p95 211 ms, 83% <=150 ms, 100% <=500 ms.  The
  pre-change baseline on the same set was p95 213 ms and 99% <=500 ms.
- Final control: `uv run python -m tools.bench_fast fast-only 080 083 086 087
  089` — 144 ayahs: median 70 ms, p90 184 ms, p95 210 ms, 85% <=150 ms,
  100% <=500 ms.  Manual trace audit confirms all 34 pause-backed starts in
  surah 83 lie inside their selected pauses; its single no-pause start is an
  explicit wasl interval.  The same audit on surah 89 finds 27 pause-backed
  starts inside their selected pauses and 2 explicit wasl intervals.
- Independent contract-and-error audits: surah 78 has 38 pause-backed starts
  inside their pauses and 1 wasl (p95 274 ms, max 497 ms); surah 79 has 44
  pause-backed starts inside their pauses and 1 wasl (p95 267 ms, max 301 ms).
- The audit also found state desynchronisation, rather than a new
  surah-specific alignment error.  On 82:5 refinement replaced the original
  recovered pause with a directly observed pre-word micro-pause; the trace now
  records that actual interval (21.553--21.656 s), and the output is 21.656 s
  versus 21.694 s reference (38 ms).  On 87 the shifted-run repair now records
  the real replacement pause (25.720--26.160 s) for its 26.039 s result
  (34 ms error).
- On 91:10 an earlier recovery had updated the selected pause centre to 37.420
  s but left the old interval (36.000--36.200 s) in the trace.  Final ownership
  is now synchronised once before writing the match: it records the actual
  37.320--37.520 s pause, action `post_selected_micro_recovery`, and the
  replaced pause.  The output remains 37.420 s (63 ms from the 37.357 s
  reference).  Contract audits of the current code on surahs 78, 79, and 80
  found no pause-boundary violations.
- `tools.bench_fast --audit-pauses` is now a required, machine-checked guard:
  it fails if a pause-backed output is outside its final trace interval or if a
  non-pause output is not explicitly classified as `wasl`.  Unit tests cover
  both accepted and rejected traces.  The independent current run on 83 and 87
  passes the contract (55 ayahs total; median 71 ms, p95 252 ms, 100% <=500
  ms).
- Rejected a tempting but invalid global correction after analysing medium
  surah 84.  Its worst case, 84:10, is assigned to the correct long pause but
  uses its centre (54.320 s) rather than the early reference point (53.417 s),
  producing 903 ms error.  Moving every low-sim `recovered` result to the
  pause start improves 84 p95 from 435 to 165 ms, but regresses independent
  83, 87, and 89 substantially.  Therefore no per-sura/threshold patch was
  added.  The remaining algorithm question is how to distinguish a weakly
  read but correctly ordered long pause from a centre-timed weak pause; it
  needs evidence beyond confidence alone.
- `tools.bench_fast --diagnose-outliers` now writes the machine-readable cause
  together with the final pause interval and matcher action.  Its current 84
  run classifies only 84:10 above 500 ms as
  `weak-recovered-pause-needs-edge-analysis`; the pause contract still passes.
  A full local CTC read over ayahs 9--11 placed that start even later (54.631
  s), confirming that this is an acoustic leading-edge problem, not an
  insufficient-read or wasl problem.
- Accepted a constrained leading-edge recovery after the above diagnosis.  A
  `recovered` DSP pause uses its start (rather than centre) only when its text
  read is weak, the pause is at least 600 ms, and the candidate remains within
  3 s of the duration prior.  This is a property of the evidence, not surah
  84.  On 84 it changes p95 435 -> 246 ms and max 903 -> 423 ms, reaching
  100% <=500 ms while retaining the pause contract.  Independent 83, 87, 89,
  and 91 retain their prior metrics and each passes the pause contract.
- Additional independent medium-sura audit after the change: 80/82 (61 ayahs)
  has median 75 ms, p95 211 ms, 100% <=500 ms; 85/86 (39 ayahs) has median 56
  ms, p95 147 ms, 100% <=500 ms.  All four suras pass the pause contract and
  produce no diagnostic outlier above 500 ms.
- The complete current medium fast-alignment set (78--92, 96, and 98) has now
  been audited in independent batches.  Every result passes the pause contract
  and is <=500 ms from Shatri reference; the highest observed error is 497 ms
  in 78.  The local Shatri directory contains all 114 recordings, so this set
  is a fast training/control boundary, not a claim of full-corpus completion.
- `tools.bench_fast corpus` now explicitly selects 1--114; the pre-existing
  `all` remains the 78--114 Juz control.  This prevents a future full-corpus
  claim from accidentally exercising only the latter subset.
- Full-corpus smoke test found a separate compact-pause class in 1:2: the
  pre-existing generic weak-pause fallback used the start of an 840 ms pause
  despite a weak following span, yielding 728 ms error.  The inverse general
  rule keeps only such compact, weak, recovered DSP pauses at centre when the
  following span is unclear.  It improves 1:2 to 308 ms and retains the 84
  leading-edge result; independent 83/87/89/91 remain unchanged and pass the
  pause contract.
- The final pause-match trace now records `timing_strategy` for the two new
  weak-recovery timing decisions.  `--diagnose-outliers` includes this field,
  so a future error can be attributed to the exact edge/centre policy rather
  than inferred from `timing_adjusted`.
- Long independent surah 55 exposed two pause-contract violations despite good
  reference error: timing logic had moved starts to local micro-valleys but
  left the original pause in trace.  The arbitrary `micro-tempo` forward shift
  now requires a real detected micro-valley; the other valley recoveries carry
  their own final pause interval into the match trace.  The rerun has zero
  contract violations (78 ayahs, median 50 ms, p95 289 ms, max 483 ms,
  100% <=500 ms).  Independent 83/84 retain their metrics and pass contract.
- Surah 62:2 exposed the complementary wasl failure: it was correctly marked
  as no-pause but the 6 s local repair remained 1.616 s late.  A low-confidence
  wasl repair now reads the complete interval from the previous reliable ayah
  through the next anchor and accepts that anchored result even below the
  ordinary pause confidence threshold.  It finds 15.490 s vs 15.491 s
  reference.  The trace keeps this as `wasl` with `wasl_full_repair`, never as
  a synthetic pause.  Surah 62 now has max 153 ms and passes the contract;
  wasl-heavy 89 remains unchanged and passes.
- Surah 64:12 exercised a multi-ayah missing interval.  A future pause with
  sim=0.186 had been claimed for the missing current ayah; rejecting that
  unsupported pause kept it wasl but exposed that no reliable future anchor
  was available for the full read.  Wasl repair now uses the next placed pause
  as its right bracket when necessary and accepts an anchored full-span result
  with absolute confidence >=0.10 rather than comparing it to an unreliable
  local window.  Surah 64 max improves 4071 -> 431 ms, all 18 ayahs are
  <=500 ms, and the pause contract passes.  Independent 62 and 89 retain their
  metrics and pass.
- Surah 61:14 exposed the reverse failure: a real but `weak-dsp` pause at
  220.600--220.820 s was excluded from Phase 1, so a low-sim match claimed the
  unrelated final pause at 239.660--240.100 s.  The repair read the correct
  opening at 220.694 s with confidence 0.688 but previously rejected it because
  the weak pause was absent from the active grid.  A high-confidence local read
  that lands inside an exact weak DSP interval now promotes that interval to the
  final pause owner.  This also reclassified 61:13 from provisional wasl to its
  detected weak pause at 210.220--210.480 s; a full wasl-span repair remains
  exempt.  Surah 61 now has 13 pause-backed boundaries, zero wasl boundaries,
  max 653 ms (the remaining separate leading-edge cases are 10/11), and passes
  the pause contract.  Independent medium controls 62 and 64 retain their
  prior results and pass.
- The remaining 61:10/11 errors are a different long-pause edge class, not an
  ownership failure: both retain their correct pause and pass the contract, but
  the fast centre marker is 653/507 ms early while the full alignment finds
  12/53 ms error.  A trial of a one-boundary contextual repair (from the
  preceding anchor through the next) still returned the pause centre and also
  changed an unrelated weak-pause classification, so it was rejected and
  removed.  The next general investigation must use a broader, cost-bounded
  context read for this class; it must not change long-pause timing globally.
- The required baseline
  `uv run python -m tools.bench_fast fast-only --audit-pauses
  --diagnose-outliers 061 062 064` reproduced those two outliers and no pause
  contract violations: 43 ayahs, median 62 ms, p95 431 ms, 95% <=500 ms.
  Exact trace showed two 1.580 s DSP pauses.  For 61:10, the best non-edge local
  CTC probe moved from in-pause candidates near 156.5 s to 155.454 s
  (`conf=0.391`), 326 ms before the selected 155.780--157.360 s pause; the
  probe onsets spanned 1.241 s.  For 61:11, `sim=0.564` skipped refine and kept
  the centre of 172.460--174.040 s.  Neighbouring pause-backed anchors were
  61:9 at 140.208 s and 61:12 at 190.350 s.
- Accepted a reference-free uncertainty guard: only a long DSP pause whose
  sufficiently confident local onset is outside the selected interval, while
  another window remains inside and the probe spread is at least half the pause
  duration, requests the expensive read.  A scan of 61/62/64 and independent
  80/83/84/87/89 triggered only on 61:10.  The repair forced-aligns the complete
  known text in one block from the previous pause-backed ayah through the anchor
  two ayahs later (bounded by the existing 120 s section limit).  It accepts
  each block onset only with `conf>=0.45` and only inside that ayah's already
  selected pause, so ownership and the pause/wasl contract are preserved.
- An initial implementation ran the block before ownership repairs.  Although
  it fixed 61:10/11, making 61:10 a new reliable anchor prevented the later
  weak-DSP repair for 61:14 and regressed it by 19.334 s.  That ordering was
  rejected.  The block is now a final timing-only phase after all
  classification/ownership repair, so it cannot influence later ayahs.
- Final 61 trace: 61:10 is 157.221 s versus 157.223 s (2 ms,
  block confidence 0.590); 61:11 is 173.816 s versus 173.757 s (59 ms,
  confidence 0.930).  Both remain inside their original pauses with action
  `context_block_edge_recovery`; 61:14 retains its 220.600--220.820 s weak-DSP
  owner and is 84 ms from reference.  The final 61/62/64 audit has 43 ayahs,
  median 57 ms, p95 166 ms, 88% <=150 ms, 100% <=500 ms; every sura passes the
  pause contract.  Independent 80/83/84/87/89 has 152 ayahs, median 73 ms,
  p95 213 ms, max 423 ms, 100% <=500 ms, and all five contracts pass.
- Focused verification: `109 passed` for `tests/test_align_fast.py`,
  `tests/test_pipeline.py`, and `tests/test_bench_fast.py`; Ruff and
  `git diff --check` pass.  Full-corpus proof has not been run, so the overall
  all-sura goal remains open.

An additional broad run was deliberately interrupted after surah 36 (94%
<=150 ms, 100% <=500 ms) because the next file was too long for the
medium-sura resource budget; it is not used as control evidence.

Independent surah 86 also completed without regression (61 ayahs: median
52 ms, p90 96 ms, max 99 ms, 100% <=150 ms and <=500 ms).  It took 43.2 s for
77.8 s of audio, so the remaining batch was deliberately interrupted before a
result; this is a speed signal for a separate algorithm review, not a timing
regression and not evidence for the present guard.

Docs updated: this task file.  `docs/project.md` and `docs/STATUS.md` are
unchanged because the public pipeline contract did not change.

## Continuation 2026-08-07: full-span ownership repairs

- A segmented 1--114 diagnostic run found failures that a reference-error-only
  bench hid: final starts could retain a stale pause interval after repair,
  reversed neighbouring owners could be made merely monotonic, and a long
  provisional wasl run could be labelled without actually reading all of its
  text.  Owner replacement is now atomic (`start`, `place`, matcher action and
  trace interval change together), reversed-owner blocks are reread between
  reliable anchors, and wasl reads no longer have the former 20 s truncation.
  A non-pause result is recorded as `wasl` only after that full read; otherwise
  the trace exposes `insufficient_audio` and the pause audit fails honestly.
- Broad long-pause correction was rejected again.  Context edge recovery is
  limited to pauses no longer than 1.8 s; final reconciliation may snap a
  pause-backed start into its existing owner by at most 700 ms.  This preserves
  the 61:10/11 result while avoiding the previously observed multi-second
  movements on independent long DSP pauses.
- Surahs 5 and 9 exposed low-sim owners whose adjacent text-span geometry was
  impossible even though a one-word probe was confident.  Only
  `weak_pause_recovery`/`first_opening_repair` owners with both neighbour
  qualities <=0.10 and an absolute prior disagreement >=80 s are removed from
  the reliable-anchor set and given a full-span read.  This reduces 5:111 from
  45.744 s error to about 10.9 s maximum for the sura and corrects 9:121; the
  pause contract remains `OK`.  Wider action/30 s variants were rejected after
  they regressed surah 37.
- Exact trace of the surah-37 regression showed a different ambiguity:
  37:71 had confidence 0.987 on the one-word opening `وَلَقَدۡ`, but 37:72 has
  the identical opening, the pause similarity was only 0.380, and the previous
  text-span quality was 0.042.  It was therefore an invalid left anchor for an
  atomic block read.  Such a repeated weak opening is no longer admitted as a
  reliable pause anchor.  Surah 37 returned from median/p90/max
  62/208/8964 ms to 60/177/8389 ms with contract `OK`.
- A first attempt to accept every confident unique opening inside a long pause
  corrected 9:122 but regressed 37:66 by 25.027 s: exact Arabic uniqueness did
  not represent the acoustic competition between `فَإِنَّهُمۡ` and
  `إِنَّهُمۡ`.  That rule was removed.  The accepted rule additionally requires
  the full read to select the next trusted pause after the local read, with no
  intervening trusted pause.  For 9:122 the local onset is 2877.063 s, the next
  trusted pause is 2880.740--2883.240 s, and the full onset is 2883.072 s with
  confidence 0.899.  For 37:66 a trusted 499.220--499.540 s pause intervenes;
  for 37:67 the target opening is repeated.  Current rerun: surah 9 max
  30.774 s -> 7.575 s; surah 37 retains 8.389 s; both contracts pass.
- Required medium/control audit after the repeated-anchor rule:
  61/62/64/80/83/84/87/89, 195 boundaries, median 70 ms, p90 171 ms,
  p95 213 ms, 85% <=150 ms and 100% <=500 ms.  Every sura passes
  `--audit-pauses`.  The primary 61/62/64 set remains at max 214/153/431 ms.
- Surah 26 exposed a shifted-section failure rather than 36 genuine wasl
  boundaries.  Up to 26:190 the pause/prior displacement stayed near
  +20--23 s; the low-trust `observed_current` claim for the ambiguous
  `وَإِنَّ`/`وَإِنَّهُۥ` opening at 26:191 jumped to +44.244 s and every later
  boundary inherited +44--56 s.  A section repair now starts only on this
  observable combination: pause-backed `observed_current`, similarity below
  trust, opening shared prefix >=6, and a >=15 s displacement jump from the
  previous boundary.  It reads atomically to the first reliable pause anchor
  that returns to the previous displacement; if none exists, it uses the final
  audio anchor.  No reference timestamp participates in that decision.
- The 26 rerun removes the complete 191--223 cascade and the two downstream
  221/223 trace-owner violations: median 104 -> 77 ms, p90 26449 -> 219 ms,
  max 48429 -> 1560 ms, <=500 ms 83% -> 97%, and provisional wasl 36 -> 12.
  Pause contract is `OK`; the remaining six >500 ms cases are independent
  edge/short-wasl errors, not a shifted section.
- The first fresh corpus attempt was stopped after surah 2 rather than treating
  a known bad result as useful proof.  Surah 1 passed its contract with max
  308 ms.  Surah 2 passed the contract but had median/p90/max 82/480/14853 ms
  and took 939 s; 2:113/124/145/221/267 exposed several general owner and
  full-span acceptance failures.  Diagnostic full reads (reference timestamps
  used only to crop this investigation, never by the algorithm) found the
  correct openings with confidence 0.910/0.223/0.770/0.977/0.588.
- The accepted geometry rules use only runtime evidence.  A low-trust
  `weak_pause_recovery` with a repeated opening and a >=10 s displacement jump,
  or a low-trust `observed_current` with both neighbour qualities <=0.10 and a
  >=15 s jump, is not a reliable anchor.  The unique-next-pause full-read
  threshold is the ordinary acceptance threshold (`0.25`), not the stronger
  reliable-anchor threshold (`0.45`).  These changes remove the 2:113/124/145
  failures.  A `post_selected_micro_recovery` unsupported by either neighbour
  is also not a reliable anchor, which repairs 2:221 without changing its owner
  directly.
- An attempted absolute replacement guard for every unsupported
  `post_selected_micro_recovery` was rejected: it changed the useful 37:67
  result from 962 ms to 2962 ms.  The final rule only excludes that boundary
  from use as a context anchor; it does not force a replacement owner.
- Exact 2:267 trace proved why the first geometry patch did not change the
  result.  The false `micro-tempo` pause at 6445.559--6445.663 s was correctly
  marked geometry-unreliable (`sim=0.256`, both neighbour qualities zero), but
  the drifting interpolation predicted 6445.941 s, only 330 ms from the same
  false owner.  The generic 1200 ms "already consistent" shortcut therefore
  skipped the full read.  A geometry-invalidated owner can no longer validate
  itself through that same prior or through a one-word retry; it requires the
  anchored full-span read.  `post_selected_micro_recovery` remains explicitly
  excluded from this forced path.
- The final 2:267 full read spans the reliable neighbouring anchors, returns
  6440.338 s with confidence 0.629, and atomically replaces the false micro
  owner with the trusted DSP pause 6440.180--6440.500 s.  Surah 2 improves to
  median/p90/max 80/382/4094 ms, 92% <=500 ms, in 525.7 s; its pause contract
  is `OK`.  The remaining max is the independent 2:18 edge/read error rather
  than an owner cascade.
- Final mandatory and independent audits preserve their baselines and all pass
  the pause contract: 61 `72/164/214`, 62 `57/134/153`, 64 `57/166/431`,
  37 `60/177/8389`, 26 `77/219/1560`, 83 `108/213/265`, and 87
  `47/96/154` (median/p90/max milliseconds).  The combined seven-sura audit has
  507 boundaries, median 66 ms, p90 201 ms, p95 280 ms, 98% <=500 ms.
- A fresh corpus run then passed surahs 1 and 2 but was stopped after surah 3
  exposed a 13.571 s error at 3:153.  The selected low-trust `recovered` owner
  was 2959.660--2960.260 s, while reliable anchors 152/154 predicted
  2971.775 s and a detected trusted pause existed at 2973.080--2976.020 s.
  Both neighbouring span qualities were zero.  The first attempted rule still
  did not run the full read because the old owner could not prove a local
  onset; gate tracing now records this distinction explicitly.  The accepted
  rule makes a low-trust `recovered` owner with both qualities <=0.10 and a
  >=10 s conflict with anchored geometry require the full read even when the
  local retry is unresolved.
- The 3:153 full read placed the short `۞ إِذۡ` opening inside the trusted pause
  but its own three-token confidence was zero; the following word of the same
  ayah started 240 ms later with confidence 0.470.  Only for an already
  geometry-invalidated owner, inside an existing trusted pause, may a following
  word from the same joint read and same ayah support the opening within 1 s.
  This changes 3:153 to the correct owner with 944 ms edge error and preserves
  the pause contract.
- The next surah-3 maximum, 3:100, was a different `preword_micro_recovery`
  error.  It replaced an activated weak-DSP pause at 1950.780--1951.000 s with
  a synthetic later micro interval at 1956.637--1956.790 s.  A full 99--101
  read found 3:100 at 1951.020 s with confidence 0.577.  A low-trust preword
  owner with both span qualities zero and a >=5 s displacement jump is now
  geometry-invalidated; a reliable full read may confirm a weak-DSP pause only
  inside it or within 80 ms of its acoustic edge.  Surah 3 improves from
  median/p90/max 78/491/13571 ms to 75/402/4902 ms, <=500 ms 90% -> 92%, with
  contract `OK`; 3:100 is no longer an outlier.
- The final preword/full-read rule was rechecked on 61/64/37/26.  All retain
  their prior median/p90/max values and pass `--audit-pauses`; in particular
  the post-selected 37:67 counterexample remains 962 ms and the 26:191--223
  section repair remains intact.
- The next segmented corpus pass stopped on surah 4 rather than continuing past
  a 68.816 s owner error at 4:24.  The low-trust `first_opening_repair` owner
  had both span qualities zero and was 60.528 s from the reliable-anchor
  interpolation.  Its already-existing full 23--26 read found 740.099 s with
  confidence 0.593 inside the detected 737.960--740.280 s trusted pause, but
  the result was rejected because the older absolute displacement guard was
  80 s.  A separate >=30 s anchor-geometry conflict rule for this exact action
  and two unsupported spans accepts the full result.  The 37:66 counterexample
  is excluded by its previous span quality 0.959.  4:24 improves to 704 ms.
- After 4:24, 4:130 remained 7.883 s early.  Five existing growing-window
  repairs around the reliable-anchor prediction all landed in the detected
  3377.240--3377.560 s trusted pause; their onset spread was 21 ms and peak
  confidence 0.153.  A broad multiscale-consensus rule was rejected after it
  moved the correct 37:66 boundary by 17.202 s.  The accepted rule is limited
  to low-trust `observed_current`, an owner/prediction shift >=3 s, and an
  opening below the existing ambiguity threshold (shared prefix <6).  It needs
  at least three non-edge onsets in one trusted pause, <=80 ms spread, and peak
  confidence >=0.10.  This excludes 37:66 by action and 26:191 by its six-char
  ambiguous prefix.  It uses already-computed repair windows and does not add
  a broad model read.
- Surah 4 now has median/p90/max 58/226/1304 ms instead of 58/266/68816 ms;
  4:24 and 4:130 keep real detected owners, the pause contract is `OK`, and
  coverage/runtime remain 2.026/681.7 s.  The remaining max is a separate
  short wasl error at 4:163, not an owner cascade.
- The next corpus segment stopped at 5:65: the low-trust `observed_current`
  owner was the trusted 1958.740--1959.280 s pause, 8.655 s after the
  reliable-anchor prediction and 10.933 s after reference.  The opening
  `وَلَوۡ أَنَّ` shares nine collapsed characters with 5:66.  Three existing
  non-edge repair windows formed an 11 ms cluster near 1950.656 s, close to
  the anchor prediction but far from the owner; because the repeated opening
  cannot identify the ayah locally, that cluster is evidence to request a
  complete read, not evidence for a pause or wasl classification by itself.
- The accepted rule is reference-free and cost-bounded: a low-trust direct
  pause claim with an ambiguous opening, an owner/prediction conflict >=3 s,
  and at least three non-edge local onsets within 80 ms near the prediction
  must be read from the previous reliable anchor through the next one.  The
  full 5:64--67 read found 5:65 at 1948.665 s.  Its short first word had
  confidence 0.073, but the next word of the same ayah started 400 ms later
  with confidence 0.639; the existing same-read support rule therefore proves
  the opening.  No detected pause owns that onset, so the result is atomically
  classified as `wasl` rather than merely moved off its old pause.
- Surah 5 improves from median/p90/max 88/569/10933 ms to 88/534/2219 ms;
  5:65 changes from 10.933 s error to 519 ms, pause/wasl counts change 100/19
  to 99/20, and the pause contract remains `OK`.  Independent controls preserve
  61 `72/164/214`, 64 `57/166/431`, 37 `60/177/8389`, and 26 `77/219/1560`
  (median/p90/max milliseconds), all with contract `OK`.  In particular the
  former 37:66 broad-multiscale regression and the ambiguous 26 section repair
  do not recur.
- The next fresh corpus segment stopped on surah 6: 6:152 kept a low-trust
  `first_opening_repair` owner at 3155.920--3156.280 s although reliable
  neighbours 151/153 bracketed the target near 3184 s; both adjacent span
  qualities were zero.  Its structural displacement changed from +11.389 s on
  the previous boundary to -24.059 s, a 35.447 s jump.  A first attempt to
  lower only the later anchor-conflict threshold was ineffective: confidence
  0.988 on the wrong local word made the owner reliable before that later
  phase.  The accepted rule invalidates this exact evidence class early:
  low-trust `first_opening_repair`, both span qualities <=0.10, and displacement
  jump >=20 s.  The required full 151--153 read returns 3184.904 s with
  confidence 0.883 inside the trusted 3183.940--3185.100 s DSP pause.
- 6:160 was independent, not a cascade from 6:152.  Its rejected provisional
  owner was 3393.880--3394.080 s, while the local repair returned 3384.288 s
  (`conf=0.375`) and the anchored full read returned 3385.013 s.  Those two
  independent reads straddle the same weak-DSP pause 3384.540--3385.000 s;
  the short first word had confidence 0.001 but its next same-ayah word within
  540 ms raised full-read support to 0.169.  A weak pause may now be promoted
  by this joint evidence only when full support is >=0.10, local confidence is
  >=0.25, the reads agree within 3 s, and both lie inside or within 80/350 ms
  of that same weak pause.  The result remains pause-backed, not wasl.
- Surah 6 improves from median/p90/max 61/190/28167 ms to 61/184/3108 ms and
  95% -> 96% <=500 ms; 6:152 and 6:160 disappear from outliers and the pause
  contract remains `OK`.  Post-change controls preserve 61 `72/164/214`,
  64 `57/166/431`, 37 `60/177/8389`, and 26 `77/219/1560`, all with contract
  `OK` and no new outlier.
- Surah 7 then exposed another early reliability gap at 7:38.  A low-trust
  `weak_pause_recovery` owner at 698.500--699.060 s had both adjacent span
  qualities zero and a structural displacement jump from +3.372 s to
  +34.915 s.  The older weak-recovery guard only handled a repeated opening or
  an 80 s absolute displacement, so a confident local probe (0.965) incorrectly
  made this owner reliable.  The opening is not repeated; the general missing
  condition is instead two unsupported spans plus a >=20 s displacement jump.
- Invalidating that owner early triggers the bounded read from reliable 7:37
  through 7:39.  It finds 7:38 at 670.520 s with confidence 0.947 inside the
  trusted 670.220--670.740 s DSP pause.  Surah 7 improves from median/p90/max
  95/412/28384 ms to 94/388/2430 ms; the previous outliers do not worsen and
  the pause contract remains `OK`.  Independent controls again preserve
  61 `72/164/214`, 64 `57/166/431`, 37 `60/177/8389`, and 26 `77/219/1560`,
  all with contract `OK`.
- Surah 8 first exposed a lifecycle error across 8:71--72.  The full wasl read
  moved 8:71 by 11.181 s, but its low-trust `pause_fallback` right anchor 8:72
  remained marked reliable even though that geometry depended on the old 71
  position.  The first read contained only the first ambiguous `إِنَّ` of 72
  and placed it late.  A complete 70--73 read instead found 8:72 at 1388.678 s
  with confidence 0.907.  The accepted lifecycle rule invalidates only a
  pause-backed, low-trust `pause_fallback` right anchor after an accepted full
  wasl revision >=3 s; the next repair then includes its full text and a later
  anchor.  8:72 changes from 23.554 s error to 202 ms and remains pause-backed.
- The next maximum, 8:33, was a direct low-trust owner whose one-word confidence
  was 0.994 although both span qualities were <=0.043.  Its nearest other
  reliable anchors 32/34 predicted 568.604 s, 7.556 s from the selected owner.
  A 32--34 full read found 569.724 s with confidence 0.941 inside the weak-DSP
  pause 569.680--569.920 s.  A leave-one-out pass now invalidates only
  low-trust `observed_current` owners with both qualities <=0.10 and >=5 s
  conflict with two other reliable anchors.
- The first leave-one-out implementation was rejected after it invalidated
  26:190 using 26:191 as its right anchor.  But 191 itself satisfies the
  existing shifted-section signature; this bypassed the atomic 191--223 repair,
  regressed max to 48.429 s and produced two pause-contract violations.  The
  accepted rule reserves both a shifted-section start and its immediate
  predecessor for the later atomic repair, so a known-bad section anchor can
  never serve as leave-one-out evidence.  Surah 26 returns to `77/219/1560`,
  12 wasl and contract `OK`.
- Final surah 8 median/p90/max is 92/348/2375 ms versus the initial
  86/446/23554 ms; <=500 ms improves 91% -> 93%, 8:33 and 8:72 disappear from
  outliers, and the pause contract is `OK`.  Post-scope controls preserve
  61 `72/164/214`, 64 `57/166/431`, 37 `60/177/8389`, and 26 `77/219/1560`,
  all with contract `OK`.
- The fresh surah-9 run reduced its older 7.575 s maximum to 3.014 s but exposed
  a contract-classification gap at 9:15.  Its selected 291.040--291.460 s DSP
  pause was inside the ayah before `وَيَتُوبُ`; the matcher action was empty and
  similarity 0.240, below `_MIN_MATCH_SIM`, but a local forced alignment still
  made that unclassified owner look usable.  The full 14--16 read found the
  actual 9:15 opening at 288.285 s with confidence 0.383 and no pause owner,
  proving wasl.
- A pause-backed owner with an actual positive similarity below the minimum
  match threshold and no classification action must now receive the anchored
  full-span read; a local one-word confidence cannot classify it.  Surah 9
  changes pause/wasl 109/19 -> 108/20 and improves median/p90/max
  67/324/3014 ms -> 66/301/1077 ms, with contract `OK`.
- Controls 61/64/37 retain their exact baselines and contracts.  Surah 26 keeps
  the same p90/max 219/1560 ms, all prior outliers and contract `OK`, while two
  formerly unclassified low-sim owners are now proven wasl with 54/154 ms
  errors.  Its pause/wasl changes 214/12 -> 213/13, median 77 -> 78 ms, and
  coverage 1.471 -> 1.599 (about 108 -> 124 s in the measured runs).  This is
  accepted as a bounded cost for explicit classification, not an accuracy
  regression; no >500 ms result or pause owner changed.
- Surah 10 exposed two non-ambiguous `weak_pause_recovery` owners that the
  initial >=20 s unsupported-span jump guard missed.  At 10:23/25 both adjacent
  span qualities were <=0.014 and displacement changed by 10.959/13.096 s.
  Full reads 22--24 and 24--26 found 556.897 s (`conf=0.675`) and 654.292 s
  (`conf=0.520`) inside trusted DSP pauses 556.620--557.100 s and
  654.040--654.480 s.  The general threshold is now 10 s when *both* span
  qualities are <=0.10; the existing repeated-opening rule remains separate.
- Surah 10 improves median/p90/max 55/230/8410 ms -> 54/150/1190 ms;
  coverage rises 1.159 -> 1.251 and the pause contract remains `OK`.  Controls
  61/64/37 preserve their exact results; surah 26 preserves its post-9 baseline
  `78/219/1560`, pause/wasl 213/13, coverage 1.599 and contract `OK`.
- The fresh surah-11 baseline exposed two independent lifecycle failures.
  First, low-trust `opening_run_repair` assigned 11:37 to the pause of 11:38.
  Its displacement jumped from +24.918 s to +41.589 s while the previous span
  quality was zero.  The anchored 36--39 read found 11:37 at 780.350 s (13 ms
  from the reference), but its short first word had confidence 0.188; the next
  same-ayah word 620 ms later had confidence 0.565.  The accepted rule marks
  only a low-trust forward `opening_run_repair` skip with >=30 s displacement,
  >=15 s displacement jump, and at least one unsupported adjacent span as
  geometry-invalid.  Mandatory full-span reads may use the immediately
  following same-ayah word even when the recovered onset has no pause owner;
  11:37 is therefore proven wasl and 11:38 recovers its 793.660--794.020 s
  pause.
- That repair exposed why 11:39 violated the pause contract.  Before Phase 2,
  provisional 11:38 was at 822.160 s, so the sequential monotonic clamp pushed
  the strong 11:39 pause owner from 812.780 s to 822.190 s.  After 11:38 moved
  back to 794.030 s, the later ayah was never restored and its wrong timestamp
  then poisoned 11:40.  After any accepted predecessor repair, the algorithm
  now restores the immediately following pause owner only when its similarity
  is already trusted, its preserved owner timing lies inside the claimed pause,
  and it is monotonic after the repaired predecessor.  This costs no model read
  and fixes both 11:39 and 11:40.
- 11:30--31 were a separate insufficient-audio case.  The first wasl-run was
  cut off by low-trust 11:31 at 635.525 s although the real repeated opening of
  11:30 was later; CTC forced `وَيَٰقَوۡمِ` onto 626.376 s with confidence
  0.168.  A weak non-zero confidence is no proof of wasl when the current ayah
  repeats the previous opening and the right closer itself is low-trust.  That
  combination is now classified provisionally as `insufficient_audio`, the
  closer is invalidated, and the complete span is reread through the next
  trusted anchor.  Unique openings or a trusted right closer still prove that
  the full run contained enough audio, so their low-confidence wasl results are
  preserved.
- Two broader variants were rejected.  Treating every full-wasl confidence
  below 0.25 as insufficient created pause-contract violations on 37:37/76 and
  26:144.  Triggering an owner-order block for every final owner-interval
  mismatch doubled surah-11 runtime from about 164 to 303 s and still failed to
  repair 11:39 because one weak intermediate word rejected the atomic block.
  Enabling following-word support for every provisional insufficient result
  also did not close those controls; the actual discriminator was repeated
  opening plus a weak closer.
- Final surah 11 improves median/p90/max 63/361/17725 ms -> 59/218/1124 ms,
  <=500 ms improves 91% -> 95%, coverage changes 1.173 -> 1.197, and the pause
  contract is `OK`.  Mandatory controls are `OK`: 61 `72/164/214`, coverage
  1.358; 62 `57/134/153`, coverage 1.182; 64 `57/166/431`, coverage 1.038;
  independent 37 `60/177/8389`, coverage 1.656; and 26 `78/219/1560`,
  pause/wasl 213/13, coverage 1.599.  The rejected broad variants' control
  violations do not remain.
- Surah 12 exposed the same insufficient-bracket lifecycle failure over a
  longer run.  The initial 12:77--78 wasl read ended at low-trust closer 12:79
  (`sim=0.314`) while repeated openings and minimum confidence 0.216 made that
  closer insufficient evidence.  Its false geometry then displaced pause
  owners through 12:80 by 18--23 s and left 12:80 outside its claimed pause.
  Reading only to that weak closer cannot distinguish the repeated text.
- The accepted rule detects only a run of at least two provisional wasl
  boundaries with a repeated opening, confidence below the normal acceptance
  threshold, and a pause-backed closer below `_PAUSE_TRUST_SIM`.  Before the
  run is classified, the atomic read is extended past that provisional closer
  to the next reliable pause anchor.  Every intermediate ayah is then replaced
  together: an onset inside a supported pause inherits that owner; a positive
  non-pause onset is wasl because the complete audio bracket has now been
  proven.  A zero-confidence onset still remains insufficient audio.
- The first atomic-block variant was rejected: it performed the same complete
  read but required every first word to meet the generic 0.10 confidence gate,
  so one short opening rejected the entire block.  Letting the following word
  raise that first-word confidence was also rejected after producing exactly
  the same 12:77--80 errors and contract violation.  The problem was not local
  word support but classification before a reliable right bracket.
- Surah 12 improves median/p90/max from `84/1509/22909` ms to
  `80/618/14683` ms, <=500 ms from 85% to 87%, while 12:77--80 all leave the
  outlier list and the pause contract changes from one violation at 12:80 to
  `OK`.  Coverage remains 1.480 and the measured runtime remains about 183 s,
  so the accepted classification does not add cost beyond the required full
  bracket read.
- Independent controls preserve their exact preceding metrics and all have
  contract `OK`: 61 `72/164/214`, coverage 1.358; 62 `57/134/153`, coverage
  1.182; 64 `57/166/431`, coverage 1.038; 37 `60/177/8389`, coverage 1.656;
  and 26 `78/219/1560`, pause/wasl 213/13, coverage 1.599.
- Surah 13 found a distinct false `post_selected_micro_recovery` at 13:14.
  The originally selected weak-DSP pause 306.020--306.300 s agreed with both
  the neighbour-anchor prediction (306.247 s) and the local read (306.308 s),
  but the post-selection pass replaced it with 308.860--309.160 s.  The
  replacement had similarity 0.186 and both neighbouring span qualities were
  zero.  Its 480 ms preceding speech was an internal stop, not a new ayah.
- A previous absolute guard for every unsupported post-selected recovery was
  already rejected because it worsened useful 37:67 from 962 to 2962 ms.  The
  accepted discriminator is acoustic rather than surah-specific: the existing
  exception remains when the replacement reaches `_MIN_MATCH_SIM`, but an
  unsupported replacement below that threshold must receive the anchored
  full-span read.  Thus 37:67 (`sim=0.335`) is unchanged while the false 13:14
  owner (`sim=0.186`) cannot revalidate itself through a local retry.
- Surah 13 improves median/p90/max from `56/173/2893` ms to `56/172/647` ms
  and <=500 ms from 95% to 98%, with unchanged coverage 0.934, runtime about
  40 s, and contract `OK`.  Controls again preserve exact metrics and contract:
  61 `72/164/214`, 62 `57/134/153`, 64 `57/166/431`, 37 `60/177/8389`, and
  26 `78/219/1560`; 37:67 remains 962 ms.
- Surah 14 exposed a high-similarity internal-repeat false owner.  The opening
  of 14:33, `وَسَخَّرَ لَكُمُ`, also occurs repeatedly inside 14:32, so the
  matcher confidently (`sim=0.553`) selected the internal DSP pause
  703.860--704.320 s.  The existing `_opening_internal_repeats` signal already
  described that acoustic ambiguity but was used only to lengthen the decode.
  With both span qualities <=0.022 and a 16.247 s displacement jump, similarity
  to a repeated internal phrase cannot make `observed_current` a reliable
  owner.  The anchored 32--34 read found the real trusted pause
  712.560--712.940 s with confidence 0.891; 14:33 changes from 8.637 s error to
  23 ms.
- That repair changed interpolation enough to expose an existing final-ayah
  ownership gap at 14:52.  Its local Phase-2 onset 1013.659 s was outside the
  retained 1025.260--1026.460 s owner, but no following ayah existed to bracket
  the required full read.  The end of the recording is now the reliable right
  bracket for a trailing interval.  The full read can therefore classify the
  last boundary as a real pause or wasl instead of accepting an unowned local
  retry; 14:52 returns to its real pause and the contract is `OK`.
- The first trailing-bracket implementation was rejected as unnecessarily
  expensive: it reread 61:14 through the recording end even though its local
  confidence 0.609 already landed inside the exact weak-DSP pause
  220.600--220.820 s.  Coverage rose 1.358 -> 1.586 without changing output.
  The accepted cost guard reuses the existing weak-DSP promotion proof: such a
  locally classified pause needs no full read, while an off-grid final onset
  like 14:52 still must be read to the recording end.  Surah 61 returns exactly
  to coverage 1.358 and its prior runtime/metrics.
- Surah 14 improves median/p90/max `71/409/8637` -> `67/253/1930` ms and
  <=500 ms 90% -> 94%; coverage is 1.097, runtime about 60 s, and contract is
  `OK`.  Controls preserve 61 `72/164/214`, 62 `57/134/153`, 64 `57/166/431`,
  37 `60/177/8389`, and 26 `78/219/1560`; all contracts are `OK`.  Surah-26
  coverage slightly improves 1.599 -> 1.591 with unchanged pause/wasl 213/13.
- Surah 15 exposed a timing-only edge case beyond the existing 1.3--1.8 s
  long-pause guard.  The 15:86 owner was the correct 7.640 s DSP pause
  777.140--784.780 s, but its center was 3.185 s from the reference.  A Phase-2
  retry independently found 784.984 s with confidence 0.340: just outside the
  opposite edge and 4.024 s from the selected center.  The initial center
  probes did not meet the old spread ratio, so no context request was made.
- A DSP pause >=1.8 s now requests the existing final timing-only context block
  when the independent retry is confident, at least 120 ms outside the pause,
  and displaced by at least half the pause duration from the selected timing
  point.  The block still accepts only confidence >=0.45 inside the already
  selected owner, so it cannot alter pause/wasl classification.  15:86 leaves
  the outlier list; surah 15 max improves 3185 -> 1559 ms and <=500 ms 91% ->
  92%, with coverage 1.108 -> 1.168, runtime about 39 -> 43 s, and contract
  `OK`.
- The first implementation was rejected after surah 37 coverage rose
  1.656 -> 1.709 and median/p90 worsened 60/177 -> 62/197.  At 37:102 the very
  long provisional owner created a request, but normal Phase 2 later replaced
  it with the correct short 1.140 s pause.  The stale timing request still ran
  and moved a near-exact center to a 349 ms error.  A very-long request is now
  revalidated against the final owner immediately before its block read; if
  ownership repair made the pause shorter than 1.8 s, the request is cancelled.
  Surah 37 returns exactly to `60/177/8389`, coverage 1.656.
- Controls preserve 61 `72/164/214`, 62 `57/134/153`, 64 `57/166/431`, 37
  `60/177/8389`, and 26 `78/219/1560`; all contracts are `OK`, and no control
  coverage increases.
- Surah 16 first exposed a four-boundary displaced section at 16:93--96.  The
  center boundary 16:94 had no classification, both span qualities were zero,
  its opening shared six collapsed characters with the next ayah, and both
  neighbouring pause owners were below `_PAUSE_TRUST_SIM`.  A local repair
  cannot decide which repeated opening owns which pause.  This exact observable
  state now requests one atomic read through the next reliable pause anchor.
  The immediate left owner is also excluded from the bracket when it is
  low-trust, unsupported on its preceding span, and acoustically shares at
  least three opening characters with the run.  Thus the complete 16:92--97
  bracket classifies 16:93--96 together instead of preserving a false left
  anchor.  The previous 9.064/15.650/24.058 s errors at 16:93--95 disappear.
- The remaining false owner at 16:27 was a different ambiguity.  The selected
  trusted-DSP pause 367.180--367.660 s had `sim=0.489`; an unused earlier
  weak-DSP pause 356.000--356.460 s decoded the same opening.  A one-word local
  probe incorrectly confirmed the late pause (`conf=0.951`), and weighted-text
  leave-one-out geometry also predicted 365.764 s, so geometry is not evidence
  here.  The general trigger is now a low-trust direct DSP owner with an unused
  DSP/weak-DSP pause 2.5--15 s earlier inside the left-anchor interval whose
  short decode has `sim>=0.40` and loses no more than 0.10 to the selected
  owner.
- Several approaches were rejected.  Extending the geometry exception for a
  short first word was invalid because the actual leave-one-out conflict was
  only 1.716 s.  One full 26--28 Viterbi still chose 366.469 s and falsely
  classified 16:27 as wasl.  Adding another left context ayah changed it only
  to 366.481 s while increasing runtime to about 171 s.  Comparing two
  separately constrained full-ayah Viterbi paths found the correct pause but
  increased runtime to about 287 s; a six-word variant was still
  pathologically slower and was stopped after nine minutes.  None of these
  rejected paths remains.
- The accepted bounded discriminator reuses greedy pause decoding with an
  8-second window only after the competing-pause trigger.  It moves ownership
  only when the earlier pause reaches extended `sim>=0.48` and beats the
  selected pause by at least 0.02.  At 16:27 the wider scores are 0.488 versus
  0.463, so the final owner becomes the real weak-DSP interval at 356.000--
  356.460 s.  This classified pause is deliberately not promoted to a reliable
  neighbouring anchor.  Surah 16 improves from `75/212/11274` to
  `75/211/6815` ms, <=150 ms from 81% to 82%, pause/wasl remains 121/6,
  coverage 1.083 -> 1.091, runtime about 110 -> 125 s, and contract remains
  `OK`.
- Independent controls retain exact metrics and contracts: 61 `72/164/214`,
  coverage 1.358; 62 `57/134/153`, coverage 1.182; 64 `57/166/431`, coverage
  1.038; 37 `60/177/8389`, coverage 1.656; and 26 `78/219/1560`, coverage
  1.591.
- The remaining 6.815 s error at 16:120 was not a context-read problem.  Phase
  1 had already identified the correct unused DSP pause 2004.440--2005.000 s
  as `best_pause` (`sim=0.389`, opening prefix 5), but retained an earlier
  low-sim `recovered` owner 1997.700--1998.080 s (`sim=0.340`, prefix 3).
  The accepted zero-read lifecycle rule applies only to a DSP-backed
  `recovered` owner below 0.40: a free later DSP pause 2.5--9 s ahead and still
  2.5 s before the next owner may replace it only with `sim>=0.38`, a gain of
  at least 0.04, prefix length at least 5, and prefix gain at least 2.  This
  moves 16:120 to its real pause without inventing a geometric prior.
- Final surah 16 median/p90/max is `75/189/980` ms versus the pre-surah-16-rule
  `75/252/24058` baseline; <=500 ms improves 92% -> 96%, pause/wasl is 121/6,
  coverage is 1.113, measured runtime about 127 s, and contract is `OK`.
  The second mandatory control pass again preserves exact 61/62/64/37/26
  metrics, coverage, outlier sets, and contracts listed above.
- Fresh surah 17 is already stable at `45/127/654` ms, 99% <=500 ms,
  pause/wasl 104/6, coverage 1.102, runtime about 121 s, and contract `OK`.
  Its only >500 ms result, 17:84, is a fully bracketed wasl read with confidence
  0.767.  A weak-DSP interval 1370.660--1370.940 s sits 389 ms before the full
  onset and contains the reference marker, but the algorithm has no independent
  local onset to distinguish a real short stop from a weak internal valley.
- A proposed general promotion of any free weak-DSP pause within 450 ms before
  a confident full-wasl onset was rejected.  It improved surah 17 to
  `44/125/378` and found real short pauses, but independently changed three
  boundaries in surah 37: p90 worsened 177 -> 190 ms and <=150 ms 85% -> 84%.
  It also changed 26 pause/wasl 213/13 -> 214/12.  Pause duration did not provide
  a principled separator: the affected correct short pauses overlapped in size.
  The promotion code and test were removed; all controls return to their exact
  preceding metrics.  Surah 17 therefore advances without an algorithm change,
  while its small wasl-onset error remains evidence for a future corpus-level
  rule rather than a surah-specific threshold.
- Surah 18 initially measured `121/317/4550` ms with contract `OK`.  Its large
  18:49 error was another pair of acoustically competing owners: the real
  weak-DSP pause 888.540--888.800 s decoded the opening at sim 0.396, while an
  internal DSP pause 892.960--893.260 s decoded the same phrase at sim 0.494
  and received a confident local one-word probe.  Lowering the first-stage
  competition gate from 0.40 to 0.39 is safe only because it still does not
  select an owner: the existing 8-second comparison rejected the earlier pause
  (0.352 versus 0.379).
- A complete 18:48--50 Viterbi was rejected: it again chose the late internal
  pause, improved the error by only 38 ms, and raised coverage 1.151 -> 1.196.
  The useful discriminator was the opposite side of each pause.  Four seconds
  before the real candidate, greedy decode matches the tail of 18:48 at 0.634;
  before the selected pause it matches that tail at only 0.216 and already
  resembles the opening of 18:49 more strongly (0.281 versus 0.190).  This
  proves the selected stop is internal without consulting reference timings.
- The accepted fallback runs only after the post-pause extended decode remains
  unresolved.  It selects the earlier pause when its pre-context matches the
  previous ayah tail at >=0.55, gains >=0.25 over the selected pause, and the
  selected pre-context conversely gains >=0.05 on the current opening.  Surah
  18 improves to `110/303/1438` ms, <=500 ms 95% -> 96%, coverage 1.163,
  runtime about 115 s, and contract `OK`.  Remaining >500 ms errors are timing
  choices inside already correct long pause owners and are retained as corpus
  evidence instead of receiving surah-specific edge rules.
- Mandatory controls remain exact: 61 `72/164/214`, coverage 1.358; 62
  `57/134/153`, coverage 1.182; 64 `57/166/431`, coverage 1.038; 37
  `60/177/8389`, coverage 1.656; and 26 `78/219/1560`, coverage 1.591.  All
  pause contracts are `OK` and the new pre-context read does not run there.
- Fresh surah 19 measures `79/215/1851` ms, 95% <=500 ms, coverage 1.353,
  runtime about 80 s, and contract `OK`.  All five >500 ms results are already
  attached to the correct 1.2--3.38 s pause intervals; they add independent
  evidence that no global center/start/end timing choice is valid.  Local reads
  point toward the right edge for 19:7/42, while an existing heuristic chose
  the left edge for 19:87 even though its reference lies near the right edge.
- One lifecycle experiment retained the confident full-span onset when it was
  already inside the same owner and at least 500 ms from the selected timing.
  It fixed 19:42 and improved surah 19 to `76/213/1851` without extra coverage,
  but was rejected after controls: 62 p90 worsened 134 -> 141, 37 worsened
  `60/177` -> `62/197` with <=150 ms 85% -> 84%, and 26 median changed 78 ->
  77.  Thus even an in-owner full CTC onset is not a universal pause-edge
  oracle.  The code and test were removed; surah 19 retains its baseline while
  long-pause timing remains a cross-sura investigation.
- Fresh surah 20 initially measured `82/243/6192` ms, 96% <=500 ms,
  coverage 1.084, and contract `OK`.  The two ownership errors had independent
  observable causes.  For 20:83 Phase 1 had already selected the real compact
  long DSP pause 845.100--847.240 s at sim 0.327, but the geometry-only
  internal-stop rule replaced it with a prior pause at sim 0.116.  Requiring
  acoustic support before such a replacement only for compact long pauses
  (<=2.5 s) retains the real owner; genuinely very long stops remain ambiguous
  and keep the existing geometry recovery.
- The first unrestricted acoustic-support guard was rejected after it regressed
  37:100: the selected stop there is 7.26 s long, so its poor prior decode is
  not evidence against the existing geometry recovery.  The bounded rule
  restores the exact surah-37 baseline `60/177/8389`, coverage 1.656.
- For 20:73 the late selected `micro-tempo` pause decoded no opening prefix.
  A maximum-width repair read found the current opening confidently (0.579) at
  the left edge, 213 ms after a free trusted DSP pause 688.360--688.860 s; that
  pause's greedy decode has an exact three-character opening prefix.  The
  accepted post-pause recovery therefore runs only for an untrusted
  `micro-tempo` owner, a confident left-edge onset within 350 ms after a free
  trusted DSP pause, a candidate prefix >=3, and selected prefix zero.  It
  promotes the pause as final owner instead of treating the edge read as an
  off-grid onset.  No reference timing participates in the decision.
- Surah 20 now measures `81/240/1956` ms, <=150 ms improves 72% -> 73%,
  coverage stays 1.084, runtime about 84 s, and contract is `OK`; both 20:73
  (`5101` ms) and the wrong-owner part of 20:83 (`6192` ms) are removed.  The
  remaining 20:83 error is an edge choice inside the correct pause and is not
  conflated with ownership recovery.
- Mandatory controls remain exact after both accepted surah-20 rules: 61
  `72/164/214`, coverage 1.358; 62 `57/134/153`, coverage 1.182; 64
  `57/166/431`, coverage 1.038; 37 `60/177/8389`, coverage 1.656; and 26
  `78/219/1560`, coverage 1.591.  All pause contracts are `OK`.
- Fresh surah 21 initially measured `93/263/3576` ms, 93% <=500 ms,
  pause/wasl 108/3, coverage 0.896, and contract `OK`.  Its 21:107 owner was a
  late `micro-tempo` stop inside the ayah: its greedy decode has zero opening
  prefix, while an unused trusted DSP pause 3.62 s earlier has a two-character
  opening prefix and sim 0.237.  The prior pause also improves previous-anchor
  span quality 0.580 -> 0.962 and the two-sided pair quality 0.778 -> 0.894.
- The accepted micro-owner rule requires all of those independent signals:
  action `weak_pause_recovery`, selected prefix zero, a free prior DSP pause no
  more than 5 s earlier with sim >=0.22 and prefix >=2, previous-span quality
  >=0.85 with gain >=0.30, next-span quality >=0.75, and two-sided pair gain
  >=0.10.  It cannot turn an arbitrary geometrically convenient pause into an
  owner and uses no reference timing.  Surah 21 improves to `91/258/2706` ms,
  94% <=500 ms, pause/wasl 109/2, coverage 0.900, and contract `OK`.
- Mandatory controls again remain exact after the surah-21 rule: 61
  `72/164/214`, 62 `57/134/153`, 64 `57/166/431`, 37 `60/177/8389`, and 26
  `78/219/1560`, with unchanged coverage and pause/wasl classification.  All
  contracts are `OK`.
- Fresh surah 22 initially measured `129/440/7977` ms, 92% <=500 ms,
  pause/wasl 72/5, coverage 1.288, and contract `OK`.  The 22:62 wasl read was
  confidently wrong because adjacent 22:61 and 22:62 share a 19-character
  opening (`dhalika bi-anna Allah...`): the full 67.6-second CTC span stretched
  the first occurrence across the second and placed the boundary 7.86 s late.
  A free trusted DSP pause 1251.110 s decoded the distinguishing continuation
  of 22:62 better than 22:61 (0.413 versus 0.338), while its left-anchor span
  quality was 0.973 versus 0.587 for the full-read onset.
- The accepted repeated-wasl recovery runs only for a single-boundary wasl with
  a shared opening >=6 characters and a reliable complete-span read.  It reuses
  or performs a minimum 4.5-second candidate decode, then requires a free DSP
  pause 2.5--12 s before the full onset, current sim >=0.38, acoustic gain over
  the previous opening >=0.05, span quality >=0.85, and span-quality gain
  >=0.30.  The selected pause becomes the final owner; it is not emitted as a
  non-pause onset.  This is the mandatory wider-context path for a proved
  repeated-opening ambiguity and does not use reference timings.
- Surah 22 improves to `120/422/1852` ms, 94% <=500 ms, pause/wasl 73/4,
  coverage 1.340, runtime about 122 s, and contract `OK`.  Mandatory controls
  remain exact: 61 `72/164/214`, 62 `57/134/153`, 64 `57/166/431`, 37
  `60/177/8389`, and 26 `78/219/1560`, with unchanged coverage and contracts.
- Fresh surah 23 is stable without a rule change at `61/184/437` ms, 100%
  <=500 ms, pause/wasl 112/5, coverage 1.197, runtime about 71 s, and contract
  `OK`.
- Fresh surah 24 initially measured `95/562/1713` ms, 89% <=500 ms,
  pause/wasl 61/2, coverage 1.260, and contract `OK`.  For 24:2 the provisional
  weak-DSP owner 20.140--20.340 s had sim 0.367, but the already-required
  six-second refine found the opening at 22.020 s with confidence 0.706,
  directly inside a different free trusted DSP pause 21.700--22.140 s.  The
  accepted lifecycle invariant lets such an existing confident refine replace
  the low-trust owner; it performs no additional model read and keeps the final
  marker inside the newly proved owner.
- Surah 24 improves to `85/485/1112` ms, 91% <=500 ms, unchanged pause/wasl and
  coverage, runtime about 115 s, and contract `OK`.  Mandatory controls remain
  exact: 61 `72/164/214`, 62 `57/134/153`, 64 `57/166/431`, 37
  `60/177/8389`, and 26 `78/219/1560`, with unchanged coverage and contracts.
- Fresh surah 25 initially measured `71/279/8475` ms and failed the pause
  contract at 25:26: Phase 1 had selected the correct 437.220--439.220 s owner,
  but Phase 2 provisionally pushed its start to 445.950 s behind a predecessor
  that was itself still wrong.  Repair later moved the predecessor back and a
  confident local read (0.560) landed at 439.180 s inside the original owner,
  yet the successor retained the obsolete monotonic clamp because its Phase-1
  similarity 0.402 was below the trusted-owner threshold.
- The accepted lifecycle rule restores the owner's original timing only when
  the current pause-backed start is outside that owner, a reliable repair lands
  inside the exact same interval, the owner timing is itself inside it, and the
  repaired predecessor leaves a valid monotonic gap.  It neither changes pause
  ownership nor treats a CTC point inside silence as a universal edge oracle.
  Surah 25 now measures `71/279/878` ms, 94% <=500 ms, unchanged pause/wasl and
  coverage 1.191, runtime about 70 s, and contract `OK`; 25:26 improves from
  8475 to 745 ms.  Mandatory controls remain exact: 61 `72/164/214`, 62
  `57/134/153`, 64 `57/166/431`, 37 `60/177/8389`, and 26 `78/219/1560`, with
  unchanged coverage and contracts.
- Fresh surah 27 initially measured `74/256/11786` ms, 94% <=500 ms,
  coverage 1.133, runtime about 84 s, and contract `OK`, but two pause-backed
  starts were false internal owners.  For 27:11 a low-trust recovered pause
  (sim 0.242) had previous/next span quality 0.199/0.0 and was 5.64 s from its
  prior; a short probe nevertheless stretched the unique opening onto that
  pause at confidence 0.498.  A complete read between reliable neighbours
  moved it to the real 151.200--152.240 s pause.  For 27:80 the previous ayah
  already contains the same `innaka` opening: the selected owner was 9.44 s
  early at sim 0.384 with zero quality on both spans, while the matcher had
  already observed a later pause at sim 0.517.  Anchored repair found the real
  weak-DSP pause 1191.120--1191.540 s at confidence 0.831.
- The accepted uncertainty trigger requires either a recovered owner with sim
  <0.25, both spans effectively unsupported and >=5 s prior displacement, or a
  direct owner at sim <0.40 with zero span support, >=5 s early displacement,
  and an already-observed later pause at sim >=0.50 with >=0.10 gain.  A
  geometry-invalidated owner's stale one-word confidence can no longer block a
  reliable anchored repair, but the repair must still land inside a different
  trusted/weak DSP owner and pass the existing pause classification gates.
  Reference timings are used only by the benchmark, not these decisions.
- An experiment that treated any five-character internal opening repeat as the
  trigger was rejected: although it fixed 27, it changed 61 median 72 -> 80,
  raised coverage 1.358 -> 1.670, raised 62/64 coverage, and changed surah 26.
  That path was removed.  The accepted narrower rule yields surah 27
  `71/202/1066` ms, 95% <=500 ms, coverage 1.155, runtime about 85 s, and
  contract `OK`; 27:11 improves 11786 -> 574 ms and 27:80 falls below 500 ms.
  Mandatory controls remain exact: 61 `72/164/214` coverage 1.358, 62
  `57/134/153` coverage 1.182, 64 `57/166/431` coverage 1.038, 37
  `60/177/8389` coverage 1.656, and 26 `78/219/1560` coverage 1.591.  All
  contracts are `OK`.
- Fresh surah 28 initially measured `76/534/21090` ms, 89% <=500 ms,
  pause/wasl 81/6, coverage 1.625, runtime about 179 s, and contract `OK`, but
  contained a shifted repeated-opening run.  Ayahs 16 and 17 both start with
  `qala rabbi`; their adjacent direct owners were low-trust, both had impossible
  previous spans, and their expected delta grew by 5.94 s.  The first owner's
  confident one-word probe had locked onto the next occurrence.  The accepted
  trigger reads the complete block from the reliable predecessor through the
  first reliable anchor after this adjacent repeated pair, classifying 16--18
  atomically instead of repairing one timestamp.
- The first implementation reused the long shifted-section right-anchor search
  and was rejected: it read through ayah 45, raised coverage to 2.050/runtime to
  about 244 s, and created a false 28:23 wasl error of 3962 ms.  Bounding this
  specific repeated-pair block to the first reliable following anchor removes
  that regression; coverage for the block alone is 1.667.
- Two independent owner errors required narrower context triggers.  For 28:45
  the opening `walakinna` repeats at character 99 inside the same ayah; a short
  probe chose the later occurrence at 891.5 s.  A self-repeat may invalidate a
  direct owner only when sim <0.50, the right span quality is <=0.10, and the
  owner is at least 15 s early relative to the prior.  Anchored context then
  recovers the real 872.660 s pause.  A broader absolute-delta self-repeat rule
  was rejected after it changed 26:183--185 and pause/wasl 213/13 -> 214/12;
  requiring a negative delta and impossible right span restores the exact 26
  baseline.
- For 28:33 the selected 88 ms `micro-tempo` owner had sim 0.406, zero quality
  on both spans, and a 21.8 s prior disagreement.  Local repair remained
  under-bracketed, while the required full context located the opening inside
  the already-detected DSP pause 671.220--672.660 s.  The accepted trigger is
  limited to direct `micro-tempo` owners with both qualities <=0.10 and absolute
  displacement >=15 s.
- Surah 28 now measures `62/308/7362` ms, 94% <=500 ms, pause/wasl 82/5,
  coverage 1.710, observed runtime 174--238 s, and contract `OK`.  The remaining
  maximum is not an ownership failure: fast full-span and the independent full
  aligner both place the wasl start of 28:66 at about 1342.84 s, and greedy
  audio decodes `faamiyat` there, while the reference timestamp 1350.200 s lies
  inside the ayah.  It is recorded as a reference anomaly and no algorithm rule
  is fitted to it.  Other >500 ms results remain inside their correct pause
  owners.
- Mandatory controls remain exact after the final three rules: 61
  `72/164/214` coverage 1.358, 62 `57/134/153` coverage 1.182, 64
  `57/166/431` coverage 1.038, 37 `60/177/8389` coverage 1.656, and 26
  `78/219/1560` coverage 1.591 with pause/wasl 213/13.  All contracts are `OK`.
- Fresh surahs 29--31 need no ownership rule.  They measure respectively
  `64/331/1038` ms (coverage 1.005), `71/406/1685` ms (coverage 1.162), and
  `63/529/984` ms (coverage 0.971); all contracts are `OK`.  Every >500 ms
  result remains inside the correct pause owner and is retained as independent
  pause-edge evidence.
- Fresh surah 32 initially measured `95/456/6253` ms, 93% <=500 ms,
  pause/wasl 28/1, coverage 0.797, and contract `OK`.  Both outliers were false
  owners for which the matcher had already seen a stronger later pause.  For
  32:24 the direct owner had sim 0.401, previous/next quality 0.0/0.018 and was
  7.78 s early, while the later DSP had sim 0.502.  For 32:4 the recovered
  owner had sim 0.274, zero right-span quality and 7.45 s displacement; the
  full read found the unique opening at 45.742 s strictly inside weak-DSP
  45.620--45.880 s.
- The accepted future-evidence trigger extends the surah-27 direct case only to
  sim <0.42 with both qualities <=0.10, and adds a recovered case at sim <0.30
  with right quality <=0.10; both require >=5 s displacement, a later pause,
  candidate sim >=0.50/0.40 respectively, and >=0.10 gain.  A geometry-
  invalidated full span may promote an exact weak-DSP owner below the usual
  reliable threshold only when its target occurs once in the complete span,
  confidence is >=the normal acceptance threshold, and onset lies strictly
  inside that weak interval.  Thus 32:4 remains pause-backed rather than being
  mislabeled wasl.
- Surah 32 now measures `80/376/456` ms, 100% <=500 ms, pause/wasl 28/1,
  coverage 1.038, runtime about 25 s, and contract `OK`.  Mandatory controls
  retain exact timings and coverage: 61 `72/164/214`, 62 `57/134/153`, 64
  `57/166/431`, 37 `60/177/8389`, and 26 `78/219/1560`.  On 26 the new proof
  intentionally corrects pause/wasl 213/13 -> 214/12 without changing any
  timestamp: the unique anchored full read for 26:126 has confidence 0.438 and
  falls inside weak-DSP 920.980--921.260 s, so the old wasl was only a threshold
  artefact.  All contracts remain `OK`.
- Fresh surah 33 initially measured `78/367/5018` ms, 93% <=500 ms,
  coverage 1.613, and contract `OK`.  For 33:30 the direct weak-DSP owner had
  sim 0.311, zero support on both neighbouring spans, and a 38.1 s expected-
  position disagreement.  Its local probe remained trapped on the false owner,
  while the complete reliable-anchor span placed the opening at 567.757 s.
  The accepted uncertainty trigger is limited to direct weak-DSP owners with
  sim <0.34, both span qualities <=0.10, and absolute displacement >=20 s.
  Surah 33 improves to `78/308/1224` ms, 95% <=500 ms, coverage 1.642, and
  contract `OK`.
- Fresh surah 34 initially measured `71/675/3272` ms, 87% <=500 ms,
  pause/wasl 49/4, coverage 1.200, and contract `OK`.  For 34:34 the recovered
  88 ms `micro-tempo` owner had sim 0.423, zero support on both spans, and a
  16.2 s displacement.  The complete read between reliable anchors found the
  opening at 659.724 s with confidence 0.924 and found no pause candidate in
  the surrounding 656--664 s interval, proving wasl rather than a false pause.
  The existing impossible-micro-owner trigger therefore applies equally to a
  recovered owner: reason `micro-tempo`, both qualities <=0.10, and absolute
  displacement >=15 s.  It remains independent of surah and reference timing.
- Surah 34 now measures `71/501/1040` ms, 89% <=500 ms, pause/wasl 48/5,
  unchanged coverage 1.200, runtime about 62 s, and contract `OK`.  The
  remaining maximum at 34:20 is not an ownership failure: fast and the full
  aligner agree around 382.22 s while the reference is 1.04 s later inside the
  same wasl ayah, so no rule is fitted to it.
- Mandatory controls after the surah-33 and surah-34 rules remain exact: 61
  `72/164/214` coverage 1.358, 62 `57/134/153` coverage 1.182, 64
  `57/166/431` coverage 1.038, 37 `60/177/8389` coverage 1.656, and 26
  `78/219/1560` coverage 1.591 with pause/wasl 214/12.  All contracts are
  `OK`.
- Current focused verification: `150 passed` for `tests/test_align_fast.py`
  and `tests/test_bench_fast.py`; Ruff and `git diff --check` pass.  A fresh
  complete 1--114 proof is still pending, so the all-sura goal remains open.

Docs updated: this task file.  `docs/project.md` and `docs/STATUS.md` remain
unchanged because no public CLI, DB schema, or review workflow changed.

## Blockers

None yet.
