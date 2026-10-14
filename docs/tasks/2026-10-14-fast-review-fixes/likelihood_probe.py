"""Prospective CTC prefix posterior study on bounded PCM, never a production gate."""
import json
import math
from pathlib import Path

import numpy as np
from surah_cutter.core import align, decode
from surah_cutter.core.fast_adapters import CTCAdapter
from surah_cutter.core.fast_reads import LocalEvidenceReads
from surah_cutter.data import quran_text
from tools.bench_fast_models import dispatch_spy, verify_dispatch


def prefix_score(emissions, targets, blank):
    # Probability that independently collapsed CTC starts with this prefix.
    # Reaching the final letter immediately enters an absorbing accept state;
    # future audio is free. No duplicate post/last-letter completion paths.
    symbols = [blank]
    for token in targets[:-1]:
        symbols.extend([token, blank])
    states = np.full(len(symbols), -math.inf)
    states[0] = 0.
    done = -math.inf
    for row in emissions:
        next_states = np.full(len(symbols), -math.inf)
        for i, token in enumerate(symbols):
            source = states[i]
            if i:
                source = np.logaddexp(source, states[i-1])
            if i > 1 and token != blank and token != symbols[i-2]:
                source = np.logaddexp(source, states[i-2])
            next_states[i] = source + row[token]
        source = states[-1]
        if len(targets) > 1 and targets[-1] != targets[-2]:
            source = np.logaddexp(source, states[-2])
        done = np.logaddexp(done, source + row[targets[-1]])
        states = next_states
    return float(done)


def main():
    bundle = align.default_bundle()
    adapter = CTCAdapter(bundle)
    output = []
    for surah, start in [(78,6320),(78,11980),(108,10640),(109,10720),(77,6180)]:
        pcm = decode.decode(f"/Users/ayub/Desktop/Abu_Bakr_Ash-Shatri/{surah:03d}.mp3")
        words = quran_text.surah_words(surah)
        rendered = adapter.represent(words)
        hypotheses = [(w.ayah,w.index,units) for w, units in zip(words, rendered)]
        reads = LocalEvidenceReads(pcm,16000,adapter)
        for duration in (1500,2700,4200):
            end = start + duration
            with dispatch_spy(bundle,pcm) as calls:
                obs = reads.read(start,end,reason="prospective_competing_prefix_study")
            # Reuse the same charged PCM call for marginal scores below. This
            # separate temporary instrument captures frames, not a second pass.
            # The capture is installed for this observation in main wrapper.
            frames = CAPTURE[-1]
            verify_dispatch(calls, pcm, {"acoustic_reads": [reads.calls[-1]]})
            rankings = []
            for ayah,index,units in hypotheses:
                target = [bundle.vocab[u] for u in units if u in bundle.vocab]
                if target:
                    rankings.append((prefix_score(frames,target,bundle.blank_id),ayah,index,"".join(units)))
            rankings.sort(reverse=True)
            output.append({"surah":surah,"window_ms":[start,end],"greedy":obs.text,
                           "rivals":rankings[:8],"null":float(frames[:,bundle.blank_id].sum()),
                           "actual_dispatches":calls,"read":reads.calls[-1]})
            print(surah,start,duration,obs.text,rankings[:3],flush=True)
    Path(__file__).with_name("likelihood-probe.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    CAPTURE=[]
    original=align._emissions
    def capture(bundle,pcm,np_module):
        result=original(bundle,pcm,np_module)
        CAPTURE[:]=[result]
        return result
    align._emissions=capture
    main()
