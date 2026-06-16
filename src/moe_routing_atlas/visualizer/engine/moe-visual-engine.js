import * as THREE from 'three';
import {
    EXPERT_VERT, EXPERT_FRAG,
    HUB_VERT, HUB_FRAG, LINK_VERT, LINK_FRAG,
    SPARK_VERT, SPARK_FRAG,
} from './shaders.js';
import { expertPos, routerPosForLayer } from './layout.js';

/**
 * Neural MoE lattice: glowing neurons, electric synapses, depth path.
 */
export class MoEVisualEngine {
    constructor(scene) {
        this.scene = scene;
        this.arch = { L: 0, K: 0, cols: 10, rows: 6, gap: 1.6, totalExperts: 0 };
        this.meshes = [];
        this.uTime = 0;
        this.uSpeed = 1;

        this.experts = null;
        this.hubs = null;
        this.links = null;
        this.flow = null;
        this.gateAttr = null;
        this.selAttr = null;
        this._dummy = new THREE.Object3D();
        this._flowSegs = [];
        this._flowProg = null;
        this._flowSpeed = null;
    }

    build(archIn) {
        this.dispose();
        this.arch = archIn;
        const L = archIn.L;
        const eK = archIn.K;

        this._buildExperts(L, eK);
        this._buildHubs(L);
        this._buildPathLinks();
        this._buildFlow(96);
    }

    _track(mesh) {
        this.scene.add(mesh);
        this.meshes.push(mesh);
        return mesh;
    }

    _buildExperts(L, eK) {
        const count = L * eK;
        const s = (this.arch.expertSize ?? 0.5) * 0.55;
        const geo = new THREE.SphereGeometry(s, 10, 10);

        this.gateAttr = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
        this.selAttr = new THREE.InstancedBufferAttribute(new Float32Array(count), 1);
        geo.setAttribute('instanceGate', this.gateAttr);
        geo.setAttribute('instanceSel', this.selAttr);

        const mat = new THREE.ShaderMaterial({
            uniforms: { uTime: { value: 0 } },
            vertexShader: EXPERT_VERT,
            fragmentShader: EXPERT_FRAG,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        });
        mat.defines.USE_INSTANCING = '';

        this.experts = new THREE.InstancedMesh(geo, mat, count);
        this.experts.frustumCulled = false;

        for (let slot = 0; slot < L; slot++) {
            const modelLayer = this.arch.moeLayerIds[slot];
            for (let e = 0; e < eK; e++) {
                const [x, y, z] = expertPos(this.arch, modelLayer, e);
                this._dummy.position.set(x, y, z);
                this._dummy.updateMatrix();
                const i = slot * eK + e;
                this.experts.setMatrixAt(i, this._dummy.matrix);
            }
        }
        this.experts.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this._track(this.experts);
    }

    _buildHubs(L) {
        const geo = new THREE.SphereGeometry(0.35, 10, 10);
        this.hubWeightAttr = new THREE.InstancedBufferAttribute(new Float32Array(L), 1);
        geo.setAttribute('instanceWeight', this.hubWeightAttr);

        const mat = new THREE.ShaderMaterial({
            uniforms: { uTime: { value: 0 } },
            vertexShader: HUB_VERT,
            fragmentShader: HUB_FRAG,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        });
        mat.defines.USE_INSTANCING = '';

        this.hubs = new THREE.InstancedMesh(geo, mat, L);
        this._dummy.scale.set(0, 0, 0);
        this._dummy.updateMatrix();
        for (let i = 0; i < L; i++) {
            this.hubs.setMatrixAt(i, this._dummy.matrix);
        }
        this._track(this.hubs);
    }

    _buildPathLinks() {
        const maxVerts = 4096;
        const pos = new Float32Array(maxVerts * 3);
        const kinds = new Float32Array(maxVerts);
        const weights = new Float32Array(maxVerts);
        const along = new Float32Array(maxVerts);
        const geo = new THREE.BufferGeometry();
        geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
        geo.setAttribute('aKind', new THREE.BufferAttribute(kinds, 1));
        geo.setAttribute('aWeight', new THREE.BufferAttribute(weights, 1));
        geo.setAttribute('aAlong', new THREE.BufferAttribute(along, 1));
        geo.setDrawRange(0, 0);

        const mat = new THREE.ShaderMaterial({
            uniforms: { uTime: { value: 0 } },
            vertexShader: LINK_VERT,
            fragmentShader: LINK_FRAG,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        });

        this.links = this._track(new THREE.LineSegments(geo, mat));
        this.links.frustumCulled = false;
        this._linkPos = pos;
        this._linkKind = kinds;
        this._linkWeight = weights;
        this._linkAlong = along;
    }

    _buildFlow(capacity) {
        const r = this.arch.totalExperts >= 256 ? 0.14 : 0.1;
        const geo = new THREE.SphereGeometry(r, 6, 6);
        const mat = new THREE.ShaderMaterial({
            uniforms: { uTime: { value: 0 } },
            vertexShader: SPARK_VERT,
            fragmentShader: SPARK_FRAG,
            transparent: true,
            depthWrite: false,
            blending: THREE.AdditiveBlending,
        });
        mat.defines.USE_INSTANCING = '';
        this.flow = new THREE.InstancedMesh(geo, mat, capacity);
        this.flow.frustumCulled = false;
        this._flowCap = capacity;
        this._flowProg = new Float32Array(capacity);
        this._flowSpeed = new Float32Array(capacity);
        this._track(this.flow);
        this._hideFlow();
    }

    _hideFlow() {
        this._dummy.position.set(0, -999, 0);
        this._dummy.scale.set(0, 0, 0);
        this._dummy.updateMatrix();
        for (let i = 0; i < this._flowCap; i++) {
            this.flow.setMatrixAt(i, this._dummy.matrix);
        }
        this.flow.instanceMatrix.needsUpdate = true;
    }

    _centroid(experts) {
        let sx = 0; let sy = 0; let sz = 0; let sw = 0;
        for (const e of experts) {
            sx += e.pos[0] * e.weight;
            sy += e.pos[1] * e.weight;
            sz += e.pos[2] * e.weight;
            sw += e.weight;
        }
        if (sw <= 0) return null;
        return { pos: [sx / sw, sy / sw, sz / sw], w: sw / experts.length };
    }

    /**
     * @param {number|null} detailLayer - when set, show router→expert lines for that layer only
     */
    applyToken(acts, focusAct = null, detailLayer = null) {
        const { L, K } = this.arch;
        const gates = this.gateAttr.array;
        const sels = this.selAttr.array;

        gates.fill(0);
        sels.fill(0);

        const layerStats = {};
        const flowSegs = [];
        const linkVerts = [];
        const linkKinds = [];
        const linkWeights = [];
        const linkAlong = [];
        const activeByLayer = {};
        const hubBySlot = {};
        const pushLink = (from, to, kind, w) => {
            linkVerts.push(...from, ...to);
            linkKinds.push(kind, kind);
            linkWeights.push(w, w);
            linkAlong.push(0, 1);
        };

        for (const a of acts) {
            const slot = this.arch.layerToSlot[a.layer];
            if (slot === undefined || a.expert_idx >= K) continue;
            const g = Math.min(1, a.gate_weight);
            const idx = slot * K + a.expert_idx;
            gates[idx] = Math.max(gates[idx], g);

            if (focusAct?.layer === a.layer && focusAct?.expert_idx === a.expert_idx) {
                sels[idx] = 1;
            }

            const ls = layerStats[a.layer] ??= { maxGate: 0, sumGate: 0, count: 0 };
            ls.maxGate = Math.max(ls.maxGate, g);
            ls.sumGate += g;
            ls.count += 1;

            const [ex, ey, ez] = expertPos(this.arch, a.layer, a.expert_idx);
            (activeByLayer[a.layer] ??= []).push({ pos: [ex, ey, ez], weight: g });

            if (detailLayer != null && a.layer === detailLayer) {
                const router = routerPosForLayer(this.arch, a.layer);
                pushLink(router, [ex, ey, ez], 1, g);
            }
        }

        const layers = Object.keys(activeByLayer).map(Number).sort((a, b) => a - b);
        const centroids = layers
            .map((layer) => ({ layer, ...this._centroid(activeByLayer[layer]) }))
            .filter((c) => c.pos);

        for (const c of centroids) {
            const slot = this.arch.layerToSlot[c.layer];
            if (slot !== undefined) hubBySlot[slot] = c;
        }

        for (let i = 0; i < centroids.length - 1; i++) {
            const a = centroids[i];
            const b = centroids[i + 1];
            const w = (a.w + b.w) * 0.5;
            pushLink(a.pos, b.pos, 0, w);
            for (let f = 0; f < 2 && flowSegs.length < this._flowCap; f++) {
                flowSegs.push({ from: a.pos, to: b.pos, weight: w, phase: Math.random() });
            }
        }

        this.gateAttr.needsUpdate = true;
        this.selAttr.needsUpdate = true;

        const lv = Math.min(linkVerts.length / 3, this._linkPos.length / 3);
        this._linkPos.set(linkVerts.slice(0, lv * 3));
        this._linkKind.set(linkKinds.slice(0, lv));
        this._linkWeight.set(linkWeights.slice(0, lv));
        this._linkAlong.set(linkAlong.slice(0, lv));
        this.links.geometry.attributes.position.needsUpdate = true;
        this.links.geometry.attributes.aKind.needsUpdate = true;
        this.links.geometry.attributes.aWeight.needsUpdate = true;
        this.links.geometry.attributes.aAlong.needsUpdate = true;
        this.links.geometry.setDrawRange(0, lv);

        for (let slot = 0; slot < L; slot++) {
            const hub = hubBySlot[slot];
            if (hub?.pos) {
                this.hubWeightAttr.array[slot] = Math.min(1, hub.w * 2.5);
                this._dummy.position.set(hub.pos[0], hub.pos[1], hub.pos[2]);
                this._dummy.scale.set(1, 1, 1);
            } else {
                this.hubWeightAttr.array[slot] = 0;
                this._dummy.scale.set(0, 0, 0);
            }
            this._dummy.updateMatrix();
            this.hubs.setMatrixAt(slot, this._dummy.matrix);
        }
        this.hubWeightAttr.needsUpdate = true;
        this.hubs.instanceMatrix.needsUpdate = true;

        this._flowSegs = flowSegs.slice(0, this._flowCap);
        for (let i = 0; i < this._flowSegs.length; i++) {
            this._flowProg[i] = this._flowSegs[i].phase ?? Math.random();
            this._flowSpeed[i] = 0.35 + this._flowSegs[i].weight * 0.65;
        }

        return {
            layerStats,
            connections: detailLayer != null ? Math.floor(linkKinds.filter((k) => k > 0.5).length / 2) : 0,
            crossLinks: Math.max(0, Math.floor(lv / 2) - Math.floor(linkKinds.filter((k) => k > 0.5).length / 2)),
        };
    }

    setTime(t, speed = 1) {
        if (!this.experts || !this.hubs || !this.links || !this.flow) return;
        this.uTime = t;
        this.uSpeed = speed;
        this.experts.material.uniforms.uTime.value = t;
        this.hubs.material.uniforms.uTime.value = t;
        this.links.material.uniforms.uTime.value = t;
        this.flow.material.uniforms.uTime.value = t;

        const n = this._flowSegs.length;
        for (let i = 0; i < n; i++) {
            const seg = this._flowSegs[i];
            this._flowProg[i] = (this._flowProg[i] + this._flowSpeed[i] * 0.018 * speed) % 1;
            const p = this._flowProg[i];
            const [fx, fy, fz] = seg.from;
            const [tx, ty, tz] = seg.to;
            this._dummy.position.set(
                fx + (tx - fx) * p,
                fy + (ty - fy) * p,
                fz + (tz - fz) * p,
            );
            const s = 0.75 + seg.weight * 0.65;
            this._dummy.scale.set(s, s, s);
            this._dummy.updateMatrix();
            this.flow.setMatrixAt(i, this._dummy.matrix);
        }
        for (let i = n; i < this._flowCap; i++) {
            this._dummy.scale.set(0, 0, 0);
            this._dummy.updateMatrix();
            this.flow.setMatrixAt(i, this._dummy.matrix);
        }
        if (this._flowCap) this.flow.instanceMatrix.needsUpdate = true;
    }

    getPickTarget() {
        return this.experts;
    }

    instanceToLayerExpert(id) {
        const slot = Math.floor(id / this.arch.K);
        const expert = id % this.arch.K;
        const layer = this.arch.moeLayerIds[slot] ?? slot;
        return { layer, expert };
    }

    dispose() {
        for (const m of this.meshes) {
            this.scene.remove(m);
            m.geometry?.dispose?.();
            if (Array.isArray(m.material)) m.material.forEach(x => x.dispose());
            else m.material?.dispose?.();
        }
        this.meshes = [];
    }
}