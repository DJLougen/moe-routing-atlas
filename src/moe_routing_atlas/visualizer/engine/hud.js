/**
 * Lightweight performance + backend HUD (no deps).
 */
export class Hud {
    constructor(rootId = 'hud') {
        let el = document.getElementById(rootId);
        if (!el) {
            el = document.createElement('div');
            el.id = rootId;
            el.style.cssText = [
                'position:fixed', 'bottom:12px', 'right:12px', 'z-index:1001',
                'font:10px/1.45 ui-monospace,Consolas,monospace', 'color:#6ab',
                'background:rgba(8,8,20,0.88)', 'border:1px solid #2a3a5a',
                'border-radius:6px', 'padding:8px 10px', 'pointer-events:none',
                'min-width:140px',
            ].join(';');
            document.body.appendChild(el);
        }
        this.el = el;
        this.backend = '?';
        this.instances = 0;
        this._frames = 0;
        this._last = performance.now();
        this.fps = 0;
    }

    setBackend(name) {
        this.backend = name;
    }

    setInstances(n) {
        this.instances = n;
    }

    tick() {
        this._frames++;
        const now = performance.now();
        if (now - this._last >= 500) {
            this.fps = Math.round((this._frames * 1000) / (now - this._last));
            this._frames = 0;
            this._last = now;
            this._draw();
        }
    }

    _draw() {
        const badge = this.backend === 'webgpu'
            ? '<span style="color:#0ff;font-weight:700">WebGPU</span>'
            : '<span style="color:#fa0">WebGL</span>';
        this.el.innerHTML = `${badge} · ${this.fps} fps<br>`
            + `<span style="color:#556">${this.instances.toLocaleString()} instances</span>`;
    }
}