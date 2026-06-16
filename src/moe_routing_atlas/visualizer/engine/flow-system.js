import * as THREE from 'three';

/**
 * GPU-instanced particles that travel along routing segments (router→expert, layer→layer).
 */
export class FlowSystem {
    constructor(scene, capacity = 640) {
        this.capacity = capacity;
        this.count = 0;
        this.segments = [];
        this.dummy = new THREE.Object3D();

        const geo = new THREE.SphereGeometry(0.08, 6, 6);
        const mat = new THREE.MeshBasicMaterial({
            color: 0x44ffcc,
            transparent: true,
            opacity: 0.85,
            blending: THREE.AdditiveBlending,
            depthWrite: false,
        });
        this.mesh = new THREE.InstancedMesh(geo, mat, capacity);
        this.mesh.frustumCulled = false;
        this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        scene.add(this.mesh);

        this.progress = new Float32Array(capacity);
        this.speed = new Float32Array(capacity);
        this._hideAll();
    }

    _hideAll() {
        this.dummy.position.set(0, -9999, 0);
        this.dummy.scale.set(0, 0, 0);
        this.dummy.updateMatrix();
        for (let i = 0; i < this.capacity; i++) {
            this.mesh.setMatrixAt(i, this.dummy.matrix);
        }
        this.mesh.instanceMatrix.needsUpdate = true;
    }

    /**
     * @param {Array<{from: number[], to: number[], weight: number, kind?: string}>} segments
     */
    setSegments(segments) {
        this.segments = segments.slice(0, this.capacity);
        this.count = this.segments.length;
        for (let i = 0; i < this.count; i++) {
            this.progress[i] = Math.random();
            this.speed[i] = 0.35 + Math.min(1, this.segments[i].weight ?? 0.5) * 1.4;
        }
        for (let i = this.count; i < this.capacity; i++) {
            this.progress[i] = 0;
            this.speed[i] = 0;
        }
    }

    update(dt, speedMul = 1) {
        if (!this.count) return;

        const color = new THREE.Color();
        for (let i = 0; i < this.count; i++) {
            const seg = this.segments[i];
            this.progress[i] = (this.progress[i] + this.speed[i] * dt * 0.5 * speedMul) % 1;
            const t = this.progress[i];
            const [fx, fy, fz] = seg.from;
            const [tx, ty, tz] = seg.to;
            const w = Math.min(1, seg.weight ?? 0.3);

            this.dummy.position.set(
                fx + (tx - fx) * t,
                fy + (ty - fy) * t,
                fz + (tz - fz) * t,
            );
            const pulse = 0.7 + 0.5 * Math.sin(t * Math.PI);
            const s = (0.12 + w * 0.22) * pulse;
            this.dummy.scale.set(s, s, s);
            this.dummy.updateMatrix();
            this.mesh.setMatrixAt(i, this.dummy.matrix);

            color.setHSL(0.48 + w * 0.08, 0.9, 0.45 + w * 0.25);
            this.mesh.setColorAt(i, color);
        }
        this.mesh.instanceMatrix.needsUpdate = true;
        if (this.mesh.instanceColor) this.mesh.instanceColor.needsUpdate = true;
    }

    dispose() {
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
    }
}