// The audio render thread enforces the end even if UI timers arrive late.
const audioPlaybackGuards = new WeakMap();

function setAudioPlaybackEnd(audio, endMs) {
  if (!audio) return;
  let setEnd = audioPlaybackGuards.get(audio);
  if (!setEnd) {
    if (endMs == null) return;
    const Context = window.AudioContext || window.webkitAudioContext;
    if (!Context) return;
    const context = new Context();
    const source = context.createMediaElementSource(audio);
    const gain = context.createGain();
    source.connect(gain); gain.connect(context.destination);
    let end = null, waiting = false;
    const arm = () => {
      const now = context.currentTime;
      gain.gain.cancelScheduledValues(now);
      if (audio.paused || audio.seeking || waiting) {
        gain.gain.setValueAtTime(0, now);
        return;
      }
      if (end == null) { gain.gain.setValueAtTime(1, now); return; }
      const remaining = (end - audio.currentTime * 1000) / (1000 * audio.playbackRate);
      gain.gain.setValueAtTime(remaining > 0 ? 1 : 0, now);
      if (remaining > 0) gain.gain.setValueAtTime(0, now + remaining);
    };
    for (const event of ["play", "playing", "seeked", "ratechange", "pause", "seeking"]) {
      audio.addEventListener(event, arm);
    }
    audio.addEventListener("waiting", () => { waiting = true; arm(); });
    audio.addEventListener("playing", () => { waiting = false; arm(); });
    audio.addEventListener("emptied", () => { end = null; waiting = false; arm(); });
    setEnd = value => {
      end = value;
      arm();
      if (context.state === "suspended") void context.resume().then(arm);
    };
    audioPlaybackGuards.set(audio, setEnd);
  }
  setEnd(endMs);
}
