"""Prospective QPC boundary rule. No GT/ayah-specific exceptions."""
import numpy as np


def skeleton(text):
    letters=[c for c in text if c.isalpha() and c not in 'aeiouoāīū']
    return ''.join(c for i,c in enumerate(letters) if i==0 or c!=letters[i-1])


def recheck(bundle, pcm, sr, words, word_spans, spans, stride, checks, pauses,
            *, frame_scores, base_ms=0, reads=None):
    from surah_cutter.core import align,candidates
    from surah_cutter.core._ctc import forced_align,aligned_segments
    from surah_cutter.data import transliteration
    reads=[] if reads is None else reads
    trs=transliteration.surah_word_tr(words[0].surah)
    inverse={v:k for k,v in bundle.vocab.items()}
    seen=set()
    for wi,word in enumerate(words):
        if word.ayah in seen:
            continue
        seen.add(word.ayah)
        check=checks[word.ayah]
        if check['repaired'] or wi+1>=len(words) or words[wi+1].ayah!=word.ayah:
            continue
        ts,te=word_spans[wi]
        first=base_ms+spans[ts][0]*stride
        last=base_ms+spans[te-1][0]*stride
        split=[p for p in pauses if first<p.start_ms<p.end_ms<=last and p.depth_ms>=300]
        if len(split)!=1:
            continue
        p=split[0]
        if not candidates._dsp_accepts_pause(candidates._dsp_pause_metrics(pcm,sr,p)):
            continue
        # A confident consonant on the left may be a genuine interrupted word.
        # Only vowels/formatting or unsupported consonants may be discarded.
        first_targets,_=align._build_targets([word],[trs[word.index]],bundle,None)
        if any(skeleton(inverse[token]) and align._span_confidence(
                frame_scores, sf, ef)>=.25 for token,(sf,ef) in
               zip(first_targets,spans[ts:te]) if base_ms+sf*stride<p.end_ms):
            continue
        count=2
        while count<3 and wi+count<len(words) and words[wi+count].ayah==word.ayah and len(
                ''.join(skeleton(trs[w.index]) for w in words[wi:wi+count]))<6:
            count+=1
        selected_words=words[wi:wi+count]
        selected_trs=[trs[w.index] for w in selected_words]
        consonants=skeleton(selected_trs[0])
        opening=''.join(skeleton(tr) for tr in selected_trs)
        if not consonants or len(opening)<6:
            continue
        context_end=base_ms+(spans[word_spans[wi+count-1][1]-1][1]+1)*stride+100
        lo=max(p.center_ms,p.end_ms-200)
        hi=min(len(pcm)*1000/sr,context_end)
        lo=round(lo/20)*20
        hi=round(hi/20)*20
        if not 500<=hi-lo<=6000:
            continue
        crop=pcm[round(lo*sr/1000):round(hi*sr/1000)]
        matrix=bundle.emitter.emissions(crop)
        ids=matrix.argmax(axis=1)
        decoded=''.join(inverse[int(token)] for i,token in enumerate(ids)
                        if token!=bundle.blank_id and (i==0 or token!=ids[i-1]))
        record={'ayah':word.ayah,'start_ms':lo,'end_ms':hi,'frames':len(matrix),
                'decoded':decoded,'target':selected_trs,'accepted':False}
        reads.append(record)
        if not skeleton(decoded).startswith(opening):
            record['reason']='greedy-opening-mismatch'
            continue
        targets,local_words=align._build_targets(selected_words,selected_trs,bundle,None)
        path,scores=forced_align(matrix[None],np.asarray(targets)[None],blank=bundle.blank_id)
        segments=aligned_segments(path.reshape(-1),targets,blank=bundle.blank_id)
        local_spans=[(s.start,s.end) for s in segments]
        local_stride=align._stride_ms(len(crop),sr,len(matrix))
        local=align._check_ayah_onsets(selected_words,local_words,local_spans,
                    scores.reshape(-1),local_stride,pauses=[],base_ms=lo,lag_ms=bundle.lag_ms)[word.ayah]
        token_conf=local['token_confs']
        chars=[inverse[targets[i]] for i in range(*local_words[0])]
        consonant_conf=[c for char,c in zip(chars,token_conf) if skeleton(char)]
        first_consonant=next(i for i,char in enumerate(chars) if skeleton(char))
        body_conf=[c for char,c in zip(chars[first_consonant:],token_conf[first_consonant:]) if char.isalpha()]
        record.update(local_check=local,consonant_conf=consonant_conf,body_token_conf=body_conf)
        if (min(consonant_conf,default=0)<.45 or local['conf']<.25 or
                min(body_conf,default=0)<.25 or
                not p.center_ms<=local['onset_ms']<=p.end_ms+200):
            record['reason']='insufficient-local-word-evidence'
            continue
        old_ms=check['onset_ms']
        check.update(onset_ms=local['onset_ms'],conf=local['conf'],
                    first_token_conf=local['first_token_conf'],token_confs=token_conf,
                    reliable=check['reliable'] and local['reliable'],repaired=True,
                    reason='split-word-local-recheck',original_conf=check['conf'],
                    original_first_token_conf=check['first_token_conf'],
                    pause_start_ms=p.start_ms,pause_end_ms=p.end_ms)
        record.update(accepted=True,reason='split-word-local-recheck',before_ms=old_ms,after_ms=check['onset_ms'])
    return checks
