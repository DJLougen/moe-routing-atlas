import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { createRenderer } from './engine/create-renderer.js';
import { MoEVisualEngine } from './engine/moe-visual-engine.js';
import { Hud } from './engine/hud.js';
import { buildArchConfig, fitCameraToArch } from './engine/arch-config.js';
import { computeMoESparsity, formatB } from './engine/model-params.js';

let traceData = null;
let sceneReady = false;
let currentToken = 0;
let selectedLayer = 'all';
let selectedActivation = null;
let isPlaying = false;
let activationSpeed = 1;
let animTime = 0;
const BASE_PLAY_INTERVAL = 0.35;
const SPARSE_TOKEN_THRESHOLD = 32;
let arch = { L: 0, K: 0, cols: 10, rows: 6, gap: 1.6, totalExperts: 0, topK: 0 };
let tokenCache = new Map();

let canvas, renderApi, scene, camera, controls;
let engine = null;
let hud = null;
let clock = new THREE.Clock();

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();

function setStatus(msg, kind = '') {
    const el = document.getElementById('status');
    if (!el) return;
    el.textContent = msg;
    el.className = kind || '';
}

function normalizeTrace(data) {
    data.token_strs = data.token_strs || data.tokens || [];
    data.token_ids = data.token_ids || [];
    data.id = data.id ?? data.trace_id;
    data.activations = data.activations || [];
    data.moe_layers = data.moe_layers || null;
    return data;
}

function addStarfield() {
    const n = 2000;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
        pos[i * 3] = (Math.random() - 0.5) * 500;
        pos[i * 3 + 1] = Math.random() * 140 - 15;
        pos[i * 3 + 2] = Math.random() * 280 - 30;
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    scene.add(new THREE.Points(geo, new THREE.PointsMaterial({
        color: 0x6a4a9a, size: 0.55, transparent: true, opacity: 0.5, depthWrite: false,
    })));
}

async function fetchTraceJson(id, tokenIdx = null) {
    const q = tokenIdx != null ? `?token_idx=${tokenIdx}` : '';
    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 120000);
    try {
        const r = await fetch(`${apiBase()}/trace/${id}${q}`, { signal: ctrl.signal });
        if (!r.ok) throw new Error(`Trace ${id} not found`);
        return normalizeTrace(await r.json());
    } finally {
        clearTimeout(timer);
    }
}

function prefetchToken(tidx) {
    if (!traceData?._sparse || tidx < 0 || tidx >= traceData.token_strs.length) return;
    if (tokenCache.has(tidx) || prefetchToken._pending?.has(tidx)) return;
    prefetchToken._pending ??= new Set();
    prefetchToken._pending.add(tidx);
    fetchTraceJson(traceData.id, tidx)
        .then((data) => tokenCache.set(tidx, data.activations))
        .catch(() => {})
        .finally(() => prefetchToken._pending.delete(tidx));
}

async function activationsForToken(tidx) {
    if (!traceData) return [];
    if (!traceData._sparse) {
        return traceData.activations.filter(a => a.token_idx === tidx);
    }
    if (tokenCache.has(tidx)) {
        prefetchToken(tidx + 1);
        return tokenCache.get(tidx);
    }
    setStatus(`Loading token ${tidx}…`);
    const data = await fetchTraceJson(traceData.id, tidx);
    tokenCache.set(tidx, data.activations);
    setStatus(`Trace #${traceData.id} · token ${tidx}`, 'ok');
    prefetchToken(tidx + 1);
    prefetchToken(tidx + 2);
    return data.activations;
}

function getFilteredActivations(tidx, acts) {
    if (!traceData) return [];
    const src = acts ?? (traceData._sparse
        ? tokenCache.get(tidx) ?? []
        : traceData.activations.filter(a => a.token_idx === tidx));
    return src.filter(a => {
        if (selectedLayer !== 'all' && a.layer !== Number(selectedLayer)) return false;
        return a.expert_idx < arch.totalExperts && arch.moeLayerIds?.includes(a.layer);
    });
}

function renderActivationList(acts) {
    const list = document.getElementById('activation-list');
    if (!list) return;
    list.innerHTML = '';
    const header = document.createElement('div');
    header.className = 'act-row act-header';
    header.innerHTML = '<span>L</span><span>E</span><span>T</span><span>—</span><span>gate</span>';
    list.appendChild(header);

    [...acts].sort((a, b) => b.gate_weight - a.gate_weight).forEach((a) => {
        const row = document.createElement('div');
        row.className = 'act-row' + (
            selectedActivation?.layer === a.layer && selectedActivation?.expert_idx === a.expert_idx
                ? ' active' : ''
        );
        row.innerHTML = `<span>L${a.layer}</span><span>E${a.expert_idx}</span><span>T${a.token_idx}</span>`
            + `<span></span><span class="gate">${a.gate_weight.toFixed(3)}</span>`;
        row.onclick = () => {
            selectedActivation = a;
            currentToken = a.token_idx;
            document.getElementById('tok-slider').value = currentToken;
            highlightToken(currentToken, a);
            renderActivationList(getFilteredActivations(currentToken));
        };
        list.appendChild(row);
    });
    document.getElementById('act-count').textContent = `(${acts.length})`;
}

function populateLayerFilter(layerIds) {
    const sel = document.getElementById('layer-filter');
    if (!sel) return;
    sel.innerHTML = '<option value="all">All MoE layers</option>';
    for (const l of layerIds) {
        const opt = document.createElement('option');
        opt.value = String(l);
        opt.textContent = `Layer ${l}`;
        sel.appendChild(opt);
    }
}

function routeEntropy(acts) {
    if (!acts.length) return '—';
    const sum = acts.reduce((s, a) => s + a.gate_weight, 0);
    if (sum <= 0) return '—';
    let h = 0;
    for (const a of acts) {
        const p = a.gate_weight / sum;
        if (p > 0) h -= p * Math.log2(p);
    }
    return h.toFixed(2) + ' bits';
}

function updateArchBanner(archCfg, modelLabel) {
    const el = document.getElementById('arch-banner');
    if (!el) return;
    const p = archCfg.params;
    const paramStr = p?.totalB
        ? `${formatB(p.activeB)} active / ${formatB(p.totalB)} total`
        : 'params unknown';
    el.textContent = `${modelLabel} · ${paramStr} · ${archCfg.L} MoE layers · `
        + `${archCfg.totalExperts}E top-${archCfg.topK} · depth ${archCfg.modelDepth}`;
}

function updateSparsityMeter(sparse) {
    const nominalBar = document.getElementById('bar-nominal');
    const tokenBar = document.getElementById('bar-token');
    const nominalLbl = document.getElementById('nominal-label');
    const tokenLbl = document.getElementById('token-sparse-label');
    const expertLbl = document.getElementById('expert-util-label');
    if (!nominalBar || !sparse) return;

    const nomPct = sparse.nominalParamPct ?? 0;
    nominalBar.style.width = `${Math.min(100, nomPct)}%`;
    nominalLbl.textContent = sparse.totalB
        ? `${formatB(sparse.activeB)} / ${formatB(sparse.totalB)} (${nomPct.toFixed(1)}%)`
        : '—';

    const tokPct = sparse.expertUtilPct;
    tokenBar.style.width = `${Math.min(100, tokPct * 3)}%`;
    tokenLbl.textContent = sparse.tokenActiveB
        ? `~${formatB(sparse.tokenActiveB)} this token`
        : '—';
    if (expertLbl) {
        expertLbl.textContent = `${sparse.expertsRouted} routed · ${sparse.perLayerLabel} · `
            + `${sparse.perLayerPct.toFixed(1)}% params/layer`;
    }
}

function setInfoPanel(tidx, tok, acts, activeLayers, maxGate, connections, crossLinks, focus, entropy, sparse) {
    const info = document.getElementById('info');
    if (!info) return;
    info.replaceChildren();
    const addLine = (label, value, hl) => {
        const line = document.createElement('div');
        const l = document.createElement('span');
        l.className = 'label';
        l.textContent = label + ': ';
        const v = document.createElement('span');
        v.className = hl ? 'hl' : '';
        v.textContent = value;
        line.append(l, v);
        info.appendChild(line);
    };
    addLine('Token', `"${tok}" (#${tidx})`, true);
    addLine('Architecture', `${arch.L} MoE · ${arch.totalExperts}E · top-${arch.topK}`, true);
    if (sparse?.totalB) {
        addLine('Nominal params', `${formatB(sparse.activeB)} active / ${formatB(sparse.totalB)} total`, true);
        addLine('This token', `~${formatB(sparse.tokenActiveB)} · ${sparse.expertsRouted} expert routes`, true);
        addLine('Expert sparsity', `${sparse.expertUtilPct.toFixed(2)}% of ${sparse.maxSlots} slots`, false);
    }
    addLine('Activations', `${acts.length} · ${activeLayers}/${arch.L} MoE layers`, true);
    addLine('Max gate', maxGate, true);
    addLine('Route entropy', entropy, false);
    if (focus) addLine('Selected', `L${focus.layer} E${focus.expert_idx} · ${focus.gate_weight.toFixed(3)}`, true);
    addLine('Routers active', `${activeLayers} layer(s) routing`, true);
    if (connections) addLine('Layer detail', `${connections} router→expert (filtered)`, false);
    addLine('Depth path', `${crossLinks} layer segments`, false);
    if (traceData?._sparse) addLine('Load mode', 'per-token (large trace)', false);
}

async function highlightToken(tidx, focusAct = null) {
    if (!traceData) return;
    const raw = await activationsForToken(tidx);
    const acts = getFilteredActivations(tidx, raw);
    renderActivationList(acts);

    if (!sceneReady || !engine) return;

    const detailLayer = selectedLayer !== 'all' ? Number(selectedLayer) : null;
    const { layerStats, connections, crossLinks } = engine.applyToken(acts, focusAct, detailLayer);
    window.layerRouteStats = layerStats;

    const activeLayers = Object.keys(layerStats).length;
    const tok = traceData.token_strs[tidx] || '?';
    const maxGate = acts.length ? Math.max(...acts.map(a => a.gate_weight)).toFixed(3) : '—';
    const sparse = computeMoESparsity(acts, arch, arch.params);
    updateSparsityMeter(sparse);
    setInfoPanel(tidx, tok, acts, activeLayers, maxGate, connections, crossLinks, focusAct, routeEntropy(acts), sparse);
    document.querySelectorAll('.tok').forEach((el, i) => el.classList.toggle('active', i === tidx));
}

async function setTokenIndex(idx) {
    if (!traceData) return;
    currentToken = Math.max(0, Math.min(idx, traceData.token_strs.length - 1));
    selectedActivation = null;
    document.getElementById('tok-slider').value = currentToken;
    await highlightToken(currentToken);
}

async function onCanvasClick(event) {
    const pick = engine?.getPickTarget();
    if (!pick || !traceData) return;
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    const hits = raycaster.intersectObject(pick);
    if (!hits.length) return;
    const { layer, expert } = engine.instanceToLayerExpert(hits[0].instanceId);
    const raw = await activationsForToken(currentToken);
    const act = raw.find(
        a => a.token_idx === currentToken && a.layer === layer && a.expert_idx === expert
    );
    if (act) {
        selectedActivation = act;
        await highlightToken(currentToken, act);
    }
}

function apiBase() {
    if (window.location.protocol === 'file:') return 'http://127.0.0.1:8777';
    return window.location.origin;
}

async function loadTrace(id) {
    setStatus('Loading trace #' + id + '…');
    tokenCache.clear();
    if (prefetchToken._pending) prefetchToken._pending.clear();

    const peek = await fetchTraceJson(id, 0);
    const useSparse = (peek.num_tokens ?? peek.token_strs.length) > SPARSE_TOKEN_THRESHOLD;

    if (useSparse) {
        traceData = peek;
        traceData._sparse = true;
        tokenCache.set(0, peek.activations);
    } else {
        traceData = await fetchTraceJson(id);
        traceData._sparse = false;
    }

    arch = buildArchConfig(traceData);
    const { L, moeLayerIds, K: eK, totalExperts: K } = arch;

    if (sceneReady) {
        engine?.dispose();
        engine = new MoEVisualEngine(scene);
        engine.build(arch);
        hud?.setInstances(L * eK + L + 2048);
        fitCameraToArch(camera, controls, scene, arch);
    } else {
        window.__atlasPendingTraceId = id;
    }
    populateLayerFilter(moeLayerIds);

    const modelLabel = traceData.model_name || traceData.model_id || 'unknown model';
    updateArchBanner(arch, modelLabel);
    updateSparsityMeter(computeMoESparsity([], arch, arch.params));

    const slider = document.getElementById('tok-slider');
    slider.max = Math.max(0, traceData.token_strs.length - 1);
    slider.value = 0;

    const bar = document.getElementById('tokens');
    bar.replaceChildren();
    traceData.token_strs.forEach((tok, i) => {
        const el = document.createElement('div');
        el.className = 'tok';
        el.textContent = tok || '.';
        el.title = `Token ${i}`;
        el.onclick = () => setTokenIndex(i);
        bar.appendChild(el);
    });

    const warn = document.getElementById('expert-warn');
    const maxExpert = traceData.activations.length
        ? Math.max(...traceData.activations.map(a => a.expert_idx), 0) : 0;
    if (maxExpert >= K) {
        warn.style.display = 'block';
        warn.textContent = `Activations reference expert ${maxExpert} but model reports ${K}`;
    } else if (useSparse) {
        warn.style.display = 'block';
        warn.textContent = `Large trace (${traceData.num_tokens} tok) — loading activations per token`;
    } else {
        warn.style.display = 'none';
    }

    currentToken = 0;
    selectedLayer = 'all';
    selectedActivation = null;
    await highlightToken(0);
    setStatus(`Trace #${id} · ${modelLabel} · ${traceData.num_tokens} tok`, 'ok');
    document.querySelectorAll('.trace-chip').forEach(c =>
        c.classList.toggle('active', parseInt(c.dataset.id, 10) === id)
    );
}

function setActivationSpeed(speed) {
    activationSpeed = Math.max(0.25, Math.min(4, speed));
    const label = document.getElementById('speed-label');
    if (label) label.textContent = activationSpeed.toFixed(2).replace(/\.?0+$/, '') + '×';
}

function animate() {
    requestAnimationFrame(animate);
    const dt = clock.getDelta();
    animTime += dt * activationSpeed;
    controls.update();

    engine?.setTime(animTime, activationSpeed);
    hud?.tick();

    if (isPlaying && traceData) {
        animate._playT = (animate._playT ?? 0) + dt;
        if (animate._playT >= BASE_PLAY_INTERVAL / activationSpeed) {
            animate._playT = 0;
            setTokenIndex((currentToken + 1) % traceData.token_strs.length);
        }
    }

    renderApi.render();
}

async function initScene() {
    canvas = document.getElementById('canvas');
    hud = new Hud();

    window.addEventListener('atlas:load-trace', (e) => {
        loadTrace(e.detail).catch(err => setStatus(err.message, 'error'));
    });

    setStatus('Initializing GPU renderer…');
    renderApi = await createRenderer(canvas);
    hud.setBackend(renderApi.backend);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x05020e);
    scene.fog = new THREE.FogExp2(0x08041a, 0.0028);
    addStarfield();

    camera = new THREE.PerspectiveCamera(38, innerWidth / innerHeight, 0.5, 2000);
    camera.position.set(58, 42, -38);

    controls = new OrbitControls(camera, canvas);
    controls.enableDamping = true;
    controls.target.set(0, 10, 58);

    scene.add(new THREE.HemisphereLight(0x5533aa, 0x0a0618, 0.55));
    const key = new THREE.DirectionalLight(0x99ccff, 0.85);
    key.position.set(40, 70, 20);
    scene.add(key);

    renderApi.attach(scene, camera);
    engine = new MoEVisualEngine(scene);
    sceneReady = true;
    setStatus(`GPU ${renderApi.backend.toUpperCase()} ready — waiting for trace…`, 'ok');

    document.getElementById('tok-prev').onclick = () => setTokenIndex(currentToken - 1);
    document.getElementById('tok-next').onclick = () => setTokenIndex(currentToken + 1);
    document.getElementById('tok-slider').oninput = (e) => setTokenIndex(Number(e.target.value));
    document.getElementById('layer-filter').onchange = (e) => {
        selectedLayer = e.target.value;
        highlightToken(currentToken, selectedActivation);
    };
    document.getElementById('play').onclick = () => { isPlaying = true; animate._playT = 0; };
    document.getElementById('pause').onclick = () => { isPlaying = false; };
    document.getElementById('reset').onclick = () => { isPlaying = false; animate._playT = 0; setTokenIndex(0); };
    document.getElementById('rotate').onclick = () => { controls.autoRotate = !controls.autoRotate; };
    document.getElementById('speed-slider').oninput = (e) => setActivationSpeed(Number(e.target.value));
    setActivationSpeed(1);
    canvas.addEventListener('click', onCanvasClick);

    window.addEventListener('resize', () => {
        camera.aspect = innerWidth / innerHeight;
        camera.updateProjectionMatrix();
        renderApi.resize(innerWidth, innerHeight);
    });

    window.__atlasVizReady = true;
    window.dispatchEvent(new Event('atlas:viz-ready'));

    const pending = window.__atlasPendingTraceId;
    if (pending != null) {
        await loadTrace(pending).catch(err => setStatus(err.message, 'error'));
    } else if (traceData?.id) {
        engine.build(arch);
        fitCameraToArch(camera, controls, scene, arch);
        await highlightToken(currentToken);
    }

    animate();
}

initScene().catch((err) => {
    console.error('[atlas] init failed:', err);
    setStatus(`GPU init failed: ${err.message}`, 'error');
});