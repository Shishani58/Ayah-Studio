// Bounded previews physically end at the cut; UI timers only update selection.
function mediaTimeMs(audio) {
  return audio ? audio._pendingTimeMs ?? ((audio._clipStartMs || 0) + audio.currentTime * 1000) : 0;
}

function mediaDurationMs(audio) {
  return audio._clipEndMs != null || audio._previewReady === false ? audio._fullDurationMs : audio.duration * 1000;
}

function replaceAudioSource(audio, url, ms) {
  if (!audio._previewEvents) {
    audio._previewEvents = true;
    audio.addEventListener("loadedmetadata", () => {
      audio._previewReady = true;
      const target = audio._pendingTimeMs;
      if (target != null) {
        const relative = Math.max(0, target - (audio._clipStartMs || 0)) / 1000;
        audio.currentTime = relative;
        if (relative === 0) audio._pendingTimeMs = null;
      }
    });
    audio.addEventListener("seeked", () => { if (audio._previewReady) audio._pendingTimeMs = null; });
  }
  audio._previewReady = false; audio._pendingTimeMs = ms;
  audio.src = url;
  audio.currentTime = Math.max(0, ms - (audio._clipStartMs || 0)) / 1000;
}

function setAudioSource(audio, url) {
  audio._clipStartMs = 0; audio._clipEndMs = null;
  audio._fullDurationMs = null; audio._fullSrc = url;
  replaceAudioSource(audio, url, 0);
}

function setAudioPlaybackEnd(audio, endMs) {
  if (!audio) return;
  const time = mediaTimeMs(audio);
  if (endMs === audio._clipEndMs || (endMs == null && audio._clipEndMs == null)) return;
  const playing = !audio.paused, rate = audio.playbackRate;
  if (endMs == null) {
    audio._clipStartMs = 0; audio._clipEndMs = null;
    replaceAudioSource(audio, audio._fullSrc, time);
  } else {
    const start = Math.round(time), end = Math.round(endMs);
    if (!(end > start)) return;
    if (audio._clipEndMs == null) {
      audio._fullSrc ||= audio.getAttribute("src");
      if (audio._previewReady && Number.isFinite(audio.duration)) audio._fullDurationMs = audio.duration * 1000;
    }
    audio._clipStartMs = start; audio._clipEndMs = end;
    replaceAudioSource(audio, `${audio._fullSrc}/clip?start_ms=${start}&end_ms=${end}`, start);
  }
  audio.playbackRate = rate;
  if (playing) void audio.play().catch(() => {});
}

function seekAudioMs(audio, ms) {
  if (audio._clipEndMs != null && (ms < audio._clipStartMs || ms >= audio._clipEndMs)) {
    setAudioPlaybackEnd(audio, null);
  }
  audio._pendingTimeMs = ms;
  audio.currentTime = Math.max(0, ms - (audio._clipStartMs || 0)) / 1000;
}
