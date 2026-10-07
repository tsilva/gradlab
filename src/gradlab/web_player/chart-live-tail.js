// Recorded history supplies older, downsampled points. Streamed history fills only
// its growing tail, without replacing exact calibration annotations or backfilling
// points deliberately omitted by the chart sampler.
export function chartWithLiveTail(recorded, live, { episode, range = null, throughStep = Infinity }) {
  const base = recorded || [];
  const last = base.at(-1)?.step ?? -Infinity;
  const tail = live.filter((point) => point.episode === episode
    && point.step > last && point.step <= throughStep
    && (!range || (point.step >= range.first && point.step <= range.last)));
  return tail.length ? base.concat(tail) : base;
}
