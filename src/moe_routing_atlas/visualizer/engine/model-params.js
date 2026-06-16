/** Parse nominal total/active parameter counts from model name strings. */
export function parseModelParams(modelName = '') {
    const name = modelName || '';
    const upper = name.toUpperCase();

    // e.g. Qwen3.6-35B-A3B, Qwen3.5-35B-A3B
    const a3b = upper.match(/(\d+(?:\.\d+)?)\s*B\s*[-_]?\s*A\s*(\d+(?:\.\d+)?)\s*B/);
    if (a3b) {
        return {
            totalB: parseFloat(a3b[1]),
            activeB: parseFloat(a3b[2]),
            source: 'name',
        };
    }

    // e.g. Qwen1.5-MoE-A2.7B — total only; active estimated later from arch
    const totalOnly = upper.match(/A\s*(\d+(?:\.\d+)?)\s*B/);
    if (totalOnly) {
        return { totalB: parseFloat(totalOnly[1]), activeB: null, source: 'name-partial' };
    }

    const plainB = upper.match(/(\d+(?:\.\d+)?)\s*B/);
    if (plainB) {
        return { totalB: parseFloat(plainB[1]), activeB: null, source: 'name-partial' };
    }

    return { totalB: null, activeB: null, source: 'unknown' };
}

/** Estimate active B given MoE geometry when not in model name. */
export function estimateActiveB(totalB, arch) {
    if (!totalB || !arch?.L || !arch?.totalExperts || !arch?.topK) return null;
    const moeFrac = arch.L / Math.max(arch.modelDepth || arch.L, 1);
    const expertFrac = arch.topK / arch.totalExperts;
    // MoE FFN block is ~65-75% of layer params; use 0.7 as display heuristic
    return totalB * moeFrac * expertFrac * 0.7 + totalB * (1 - moeFrac) * 0.25;
}

export function formatB(n) {
    if (n == null || Number.isNaN(n)) return '?';
    if (n >= 1) return `${n % 1 === 0 ? n.toFixed(0) : n.toFixed(1)}B`;
    return `${(n * 1000).toFixed(0)}M`;
}

/** Per-token MoE sparsity stats for UI + 3D encoding. */
export function computeMoESparsity(acts, arch, params) {
    const L = arch.L;
    const K = arch.totalExperts;
    const topK = arch.topK || 1;
    const slotsPerLayer = K;
    const maxSlots = L * slotsPerLayer;
    const nominalRouted = L * topK;

    const perLayer = {};
    let gateSum = 0;
    for (const a of acts) {
        perLayer[a.layer] = (perLayer[a.layer] ?? 0) + 1;
        gateSum += a.gate_weight;
    }

    const layersActive = Object.keys(perLayer).length;
    const expertsRouted = acts.length;
    const expertUtilPct = maxSlots > 0 ? (expertsRouted / maxSlots) * 100 : 0;
    const routeFillPct = nominalRouted > 0 ? (expertsRouted / nominalRouted) * 100 : 0;
    const perLayerPct = (topK / K) * 100;

    let totalB = params?.totalB;
    let activeB = params?.activeB;
    if (totalB && !activeB) activeB = estimateActiveB(totalB, arch);

    const nominalParamPct = totalB && activeB ? (activeB / totalB) * 100 : null;

    // Token-weighted active estimate: scale nominal active by mean gate mass
    const meanGate = expertsRouted > 0 ? gateSum / expertsRouted : 0;
    const tokenActiveB = activeB != null
        ? activeB * (routeFillPct / 100) * (0.85 + meanGate * 0.15)
        : null;

    return {
        perLayer,
        layersActive,
        expertsRouted,
        maxSlots,
        nominalRouted,
        expertUtilPct,
        routeFillPct,
        perLayerPct,
        totalB,
        activeB,
        tokenActiveB,
        nominalParamPct,
        meanGate,
        perLayerLabel: `${topK}/${K} experts per MoE layer`,
    };
}