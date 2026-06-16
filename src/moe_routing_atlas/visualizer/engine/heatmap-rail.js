import * as THREE from 'three';

/**
 * Vertical heat strip — one bar per layer, height/color = routing intensity for current token.
 */
export class HeatmapRail {
    constructor(scene) {
        this.mesh = null;
        this.scene = scene;
        this.maxLayers = 0;
    }

    rebuild(layerCount, layerZFn) {
        if (this.mesh) {
            this.scene.remove(this.mesh);
            this.mesh.geometry.dispose();
            this.mesh.material.dispose();
        }
        this.maxLayers = layerCount;
        this._layerZ = layerZFn;
        const geo = new THREE.BoxGeometry(0.35, 1, 0.35);
        const mat = new THREE.MeshStandardMaterial({
            color: 0xffffff,
            metalness: 0.1,
            roughness: 0.3,
            emissive: new THREE.Color(0x001122),
            emissiveIntensity: 0.6,
            transparent: true,
            opacity: 0.9,
        });
        this.mesh = new THREE.InstancedMesh(geo, mat, layerCount);
        this.mesh.frustumCulled = false;
        const dummy = new THREE.Object3D();
        for (let l = 0; l < layerCount; l++) {
            dummy.position.set(-11.5, 6, layerZFn(l));
            dummy.scale.set(1, 0.15, 1);
            dummy.updateMatrix();
            this.mesh.setMatrixAt(l, dummy.matrix);
            this.mesh.setColorAt(l, new THREE.Color(0x112233));
        }
        this.scene.add(this.mesh);
    }

    /**
     * @param {Record<number, {sumGate: number, maxGate: number, count: number}>} layerStats
     */
    update(layerStats) {
        if (!this.mesh) return;
        const dummy = new THREE.Object3D();
        const cold = new THREE.Color(0x1a2840);
        const hot = new THREE.Color(0xff6600);

        for (let l = 0; l < this.maxLayers; l++) {
            const st = layerStats[l];
            dummy.position.set(-11.5, 6, this._layerZ ? this._layerZ(l) : l * 5.5);
            if (st) {
                const intensity = Math.min(1, st.maxGate * 1.5);
                const h = 0.2 + intensity * 2.2;
                dummy.scale.set(1, h, 1);
                this.mesh.setColorAt(l, new THREE.Color().lerpColors(cold, hot, intensity));
            } else {
                dummy.scale.set(1, 0.08, 1);
                this.mesh.setColorAt(l, cold);
            }
            dummy.updateMatrix();
            this.mesh.setMatrixAt(l, dummy.matrix);
        }
        this.mesh.instanceMatrix.needsUpdate = true;
        this.mesh.instanceColor.needsUpdate = true;
    }

    setLayerZFn(fn) {
        this._layerZ = fn;
    }

    dispose() {
        if (!this.mesh) return;
        this.scene.remove(this.mesh);
        this.mesh.geometry.dispose();
        this.mesh.material.dispose();
    }
}