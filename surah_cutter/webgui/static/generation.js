/* One reconnectable subscription; DOM state belongs to the caller. */
function watchGeneration({ jobId, after = 0, onEvent, onError, getStatus, onReconnect, onSnapshot }) {
  const query = new URLSearchParams({ job_id: jobId, after: String(after) });
  const stream = new EventSource(`/api/generate/stream?${query}`);
  let closed = false, checking = false, cursor = after, delivery = Promise.resolve();
  const nativeClose = stream.close.bind(stream);
  stream.close = () => { closed = true; nativeClose(); };
  const terminal = event => ["done", "cancelled", "error"].includes(event.type);
  const deliver = event => {
    if (terminal(event)) stream.close();
    delivery = delivery.then(() => onEvent(event)).catch(error => {
      stream.close(); onError(error);
    });
  };
  stream.onmessage = message => {
    if (closed) return;
    try {
      const event = JSON.parse(message.data);
      const sequence = Number(message.lastEventId);
      if (Number.isSafeInteger(sequence) && sequence > 0) {
        if (sequence <= cursor && !terminal(event)) return;
        cursor = Math.max(cursor, sequence);
      }
      deliver(event);
    } catch (error) { stream.close(); onError(error); }
  };
  stream.onerror = async () => {
    if (closed || checking) return;
    checking = true;
    onReconnect?.();
    try {
      const status = await getStatus();
      if (closed) return;
      if (status.job_id !== jobId || status.same_project === false) {
        stream.close(); onError(new Error("Расчёт изменился. Обновите состояние задания."));
      } else if (status.terminal) {
        delivery = delivery.then(() => onSnapshot?.(status));
        deliver(status.terminal);
      }
      // EventSource retries with Last-Event-ID; keep calculation controls active.
    } catch {
      // The server may temporarily be unreachable. A disconnect is not a job failure.
    } finally { checking = false; }
  };
  return stream;
}
