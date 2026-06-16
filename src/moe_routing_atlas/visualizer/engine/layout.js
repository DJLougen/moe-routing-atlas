export const Y0 = 2;
export const ROUT_Y = Y0 + 8;
export const MAX_VIZ_EXPERTS = 256;

export function gridLayout(k) {
    if (k === 60) return { cols: 10, rows: 6, gap: 1.6 };
    // 256E: larger cells so each MoE layer reads clearly on screen (not pinpoints).
    if (k === 256) return { cols: 16, rows: 16, gap: 1.38 };
    const cols = Math.max(4, Math.ceil(Math.sqrt(k)));
    const rows = Math.ceil(k / cols);
    const gap = Math.max(0.5, Math.min(1.6, 13 / cols));
    return { cols, rows, gap };
}

export function layerZ(slot, arch) {
    const gap = arch?.layerGap ?? 5.5;
    return slot * gap;
}

export function layerSlot(arch, modelLayer) {
    return arch.layerToSlot?.[modelLayer] ?? modelLayer;
}

export function expertPos(arch, modelLayer, expert) {
    const { cols, rows, gap } = arch;
    const slot = layerSlot(arch, modelLayer);
    const row = Math.floor(expert / cols);
    const col = expert % cols;
    return [
        (col - cols / 2 + 0.5) * gap,
        Y0 + 3 + (row - rows / 2 + 0.5) * gap,
        layerZ(slot, arch),
    ];
}

export function routerPosForLayer(arch, modelLayer) {
    const slot = layerSlot(arch, modelLayer);
    const rx = arch?.routerX ?? -8;
    return [rx, ROUT_Y, layerZ(slot, arch)];
}