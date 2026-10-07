// One workspace reference survives panel remounts and is shared by detached windows.
export function rewardReferenceEpisode(snapshot, epoch) {
  return JSON.stringify([epoch, snapshot?.trajectory?.episode_id,
    snapshot?.transition?.episode ?? snapshot?.session?.episode]);
}

export function createRewardReference(snapshot, epoch) {
  const transition = snapshot?.transition;
  if (!Number.isInteger(transition?.step)) return null;
  return {
    episode: rewardReferenceEpisode(snapshot, epoch),
    step: transition.step,
    sample: {
      step: transition.step,
      sequence: transition.sequence,
      reward_provider: transition.reward?.provider,
      reward_shaped: transition.reward?.shaped,
      ...(Number.isFinite(transition.decision?.value) ? { value: transition.decision.value } : {}),
    },
  };
}

export function rewardReferenceStore(storage, workspaceId) {
  const key = `gradlab-reward-reference-${workspaceId}`;
  const read = () => {
    try {
      const entries = JSON.parse(storage.getItem(key) || "[]");
      return Array.isArray(entries) ? entries : [];
    } catch { return []; }
  };
  const save = reference => {
    if (!reference) return null;
    const entries = read().filter(item => item?.episode !== reference.episode);
    storage.setItem(key, JSON.stringify([...entries.slice(-7), reference]));
    return reference;
  };
  return {
    get(snapshot, epoch) {
      const episode = rewardReferenceEpisode(snapshot, epoch);
      const existing = read().find(item => item?.episode === episode && Number.isInteger(item.step));
      if (existing) return existing;
      const reference = createRewardReference(snapshot, epoch);
      if (!reference) return null;
      // A stats window may first receive an episode after it has already advanced.
      // Its arrival timing must not choose the discount origin or invent step-zero facts.
      return save({ ...reference, step: 0, sample: reference.step === 0 ? reference.sample : null });
    },
    set(snapshot, epoch) { return save(createRewardReference(snapshot, epoch)); },
  };
}
