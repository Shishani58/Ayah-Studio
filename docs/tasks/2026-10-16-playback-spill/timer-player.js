// Historical pre-fix playback functions for reproducible baseline probes.
function ayahEndMs(i) {
  for (let j = i + 1; j < R.ayahs.length; j++) if (R.ayahs[j].time != null) return R.ayahs[j].time;
  return R.duration;
}
function scheduleAyahStop(reanchor = false) {
  clearTimeout(R.playbackTimer);
  if (W.mode || R.playbackMode === "continuous" || !R.audio || R.audio.paused || !R.dataReady) return;
  if (reanchor || R.playbackAyah < 0) R.playbackAyah = currentAyahByTime(R.audio.currentTime * 1000);
  const i = R.playbackAyah;
  if (i < 0 || R.ayahs[i]?.time == null || ayahEndMs(i) <= R.ayahs[i].time) return;
  const delay = Math.max(0, (ayahEndMs(i) - R.audio.currentTime * 1000) / R.audio.playbackRate);
  R.playbackTimer = setTimeout(() => {
    if (!finishAyahPlayback()) scheduleAyahStop();
  }, delay);
}
function finishAyahPlayback() {
  const i = R.playbackAyah;
  if (W.mode || R.playbackMode === "continuous" || !R.audio || R.audio.paused || i < 0) return false;
  const end = ayahEndMs(i);
  if (R.audio.currentTime * 1000 < end) return false;
  clearTimeout(R.playbackTimer);
  if (R.loop) seekAyah(i, true);
  else {
    R.audio.pause();
    R.audio.currentTime = Math.max(R.ayahs[i].time, end - 2) / 1000;
    setCurrent(i);
    drawWave();
  }
  return true;
}
