# Fast alignment lessons from Shatri goal thread

## Context

The goal is still active: Shatri fast-only timings must become correct for all
114 suras without using full mode. The bundled Shatri reference DB is only an
oracle for verification, not a source to copy from.

The strongest user constraint from the thread is algorithmic:

- Pauses are the primary source of ayah starts.
- Non-pauses may be used only after the algorithm has identified a true wasl
  transition, where the reciter joined two ayahs without a pause.
- If a detected pause exists near a plausible ayah boundary, the algorithm must
  explain why it did not use that pause before falling back to a non-pause or a
  later text match.

## Useful conclusions

1. Repeated fixes were treating symptoms.

   Many iterations adjusted thresholds, slack windows, weak-match floors, or
   one-off recovery branches. These sometimes improved one sura, but repeatedly
   caused regressions elsewhere. This is a sign that the model of the problem is
   wrong: the matcher is still making local greedy decisions where it needs a
   local sequence decision.

2. Greedy textual confirmation is unsafe in repeated openings.

   The clearest case is sura 37 around ayahs 66-70. A later pause can decode as
   a stronger fuzzy match for the current opening than the first real
   pause-backed start. In the observed failure, the algorithm preferred a later
   repeated `fa-innahum...` around 514.73s instead of the real pause-backed
   start around 480.15s, then cascaded into wrong assignments for following
   `thumma inna...` openings.

3. Fuzzy similarity is not enough for shared openings.

   In repeated text, short noisy decodes like `ta-in...` can score close to
   real `thumma-in...` openings. For shared openings, prefix evidence must be
   stronger than fuzzy similarity. Fuzzy score should be secondary, not the
   deciding signal.

4. The algorithm needs a local block assignment, not another guard.

   Once a repeated-opening region is detected, the correct unit of work is a
   block: several neighboring ayahs plus the available pause-backed candidates.
   The block should be solved with an order-preserving local DP/cost model.
   Scoring should include:

   - monotonic pause order;
   - prefix strength for shared openings;
   - fuzzy similarity as a secondary signal;
   - penalty for using internal repeats inside the previous ayah;
   - penalty for skipping plausible pause-backed starts;
   - explicit wasl only when no real pause-backed boundary is plausible.

5. `first_opening_repair` and `opening_run_repair` were useful diagnostics, but
   not a complete solution.

   They proved that the 37:66 cascade can be improved by treating the first
   pause-backed occurrence as important. However, greedy repair of the next
   ayahs either failed to close the block or caused regressions. The successful
   idea is "repair the block"; the unsuccessful implementation detail is
   "repair one boundary at a time."

6. Regression checks must include medium and previously green suras.

   Medium suras are the best iteration target: they finish quickly enough, but
   still include varied ayah lengths and repeated openings. Useful control set
   from the thread:

   - `036 037 056 075 076 080 083 089 098` for medium coverage;
   - `023 067 078` as regression/control suras;
   - `080 083` for local >500ms timing errors;
   - `089 098 075 076` as mostly green controls.

7. Sura 23 exposed false-positive early repair.

   A broad "pick earlier similar pause" rule made 23 regress badly. A proper
   local-DP solution must not move a boundary away from a strong current match
   just because one earlier pause has a plausible fuzzy similarity. It needs
   block-level evidence that a repeated-opening region is actually present.

8. Sura 67 exposed regression from overly broad repair windows.

   Repair logic that helps 37 can still break 67 when it is not gated by a real
   repeated-opening block. Any repair layer must be conservative outside the
   detected block.

9. Bench results must be read as evidence, not as proof of general success.

   A narrow green result can hide regressions. A useful loop is:

   - run one target sura;
   - inspect exact bad ayahs and chosen pause actions;
   - make the smallest algorithmic change;
   - run the target again;
   - immediately run regression suras that represent already-fixed classes;
   - only then expand toward all 114 suras.

10. The current completion criterion remains strict.

    The goal is not met until fast-only output is verified against Shatri
    reference for all 114 suras with no unacceptable errors, while preserving
    the invariant that pauses are the source of starts except for confirmed
    wasl.

## Anti-patterns to avoid

- Do not keep lowering global similarity thresholds to make one sura pass.
- Do not accept non-pause starts just because forced alignment found a confident
  word onset inside an ayah.
- Do not allow a later high-similarity repeated opening to skip over earlier
  plausible pauses without block-level justification.
- Do not judge a repair by one sura only; immediately check known regression
  controls.
- Do not use the Shatri reference to generate output timings; use it only to
  audit the fast algorithm.

## Next algorithmic step

Replace the current greedy repeated-opening repair with a local DP assignment:

1. Detect a suspicious repeated-opening block:
   late confirmed/current match, skipped plausible pauses, or repeated/shared
   openings in neighboring ayahs.
2. Collect candidate pauses in the local audio interval, including currently
   assigned pauses that belong to the block.
3. Score each ayah/pause pair with prefix-first logic for shared openings.
4. Solve an order-preserving assignment for the block.
5. Mark unmatched boundaries as wasl only when no plausible pause-backed
   assignment exists.
6. Run targeted and regression benches before expanding to all suras.

Docs unchanged outside this note.
