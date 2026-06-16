import { gridLayout, MAX_VIZ_EXPERTS, Y0 } from './layout.js';
import { estimateActiveB, parseModelParams } from './model-params.js';

/** Build runtime layout from trace metadata + observed MoE layer ids. */
export function moeLayerIdsFromActivations(activations, numLayers = 24, moeLayers = null) {
    if (moeLayers?.length) return moeLayers;
    const ids = [...new Set((activations || []).map(a => a.layer))].sort((a, b) => a - b);
    if (ids.length) return ids;
    return [...Array(numLayers)].map((_, i) => i);
}

export function buildArchConfig(trace) {
    const totalExperts = trace.num_experts || 60;
    const topK = trace.top_k || 0;
    const modelDepth = trace.num_layers || 24;
    const moeLayerIds = moeLayerIdsFromActivations(
        trace.activations, modelDepth, trace.moe_layers,
    );
    const L = moeLayerIds.length;
    const layerToSlot = Object.fromEntries(moeLayerIds.map((id, i) => [id, i]));

    const layout = gridLayout(Math.min(totalExperts, MAX_VIZ_EXPERTS));
    const eK = Math.min(totalExperts, layout.cols * layout.rows, MAX_VIZ_EXPERTS);

    const gridW = layout.cols * layout.gap;
    const gridH = layout.rows * layout.gap;
    const layerGap = Math.max(2.0, Math.min(5.5, 150 / Math.max(L, 1)));
    const depth = Math.max(0, (L - 1) * layerGap);
    const centerZ = depth / 2;
    const routerX = -(gridW / 2 + 3.2);

    const moeStride = moeLayerIds.length >= 2
        ? moeLayerIds[1] - moeLayerIds[0]
        : 1;

    const modelName = trace.model_name || trace.model_id || '';
    const params = parseModelParams(modelName);
    if (params.totalB && !params.activeB) {
        params.activeB = estimateActiveB(params.totalB, {
            L, totalExperts, topK, modelDepth,
        });
    }

    return {
        L,
        moeLayerIds,
        layerToSlot,
        layerGap,
        routerX,
        centerZ,
        depth,
        gridW,
        gridH,
        K: eK,
        totalExperts,
        topK,
        modelDepth,
        moeStride,
        expertSize: totalExperts >= 256 ? 0.62 : (totalExperts >= 128 ? 0.48 : 0.5),
        modelName,
        params,
        ...layout,
    };
}

export function fitCameraToArch(camera, controls, scene, arch) {
    // Frame one expert lattice — not full Z depth (40 MoE layers would shrink grids to dots).
    const lattice = Math.max(arch.gridW, arch.gridH, 14);
    const dist = lattice * 1.22 + 10;
    const cy = Y0 + arch.gridH * 0.38 + 5;

    controls.target.set(0, cy, arch.centerZ);
    camera.position.set(dist * 0.58, dist * 0.5, arch.centerZ - dist * 0.48);
    controls.minDistance = lattice * 0.45;
    controls.maxDistance = Math.max(arch.depth * 2.2, lattice * 5);
    camera.near = 0.25;
    camera.far = Math.max(1200, arch.depth * 4 + lattice * 6);
    if (camera.fov > 34) camera.fov = 34;
    camera.updateProjectionMatrix();

    if (scene?.fog) {
        scene.fog.density = 0.00085 / Math.max(1, arch.L / 24);
    }
}