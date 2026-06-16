import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';

const WEBGPU_URL = 'https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.webgpu.js';
const TSL_URL = 'https://cdn.jsdelivr.net/npm/three@0.172.0/build/three.tsl.js';
const INIT_TIMEOUT_MS = 4000;

function withTimeout(promise, ms, label) {
    return Promise.race([
        promise,
        new Promise((_, reject) =>
            setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)
        ),
    ]);
}

async function tryWebGPU(canvas) {
    if (!navigator.gpu) return null;

    const [THREE_GPU, { pass }, { bloom }] = await Promise.all([
        import(WEBGPU_URL),
        import(TSL_URL),
        import('https://cdn.jsdelivr.net/npm/three@0.172.0/examples/jsm/tsl/display/BloomNode.js'),
    ]);

    const renderer = new THREE_GPU.WebGPURenderer({
        canvas,
        antialias: true,
        powerPreference: 'high-performance',
    });
    await withTimeout(renderer.init(), INIT_TIMEOUT_MS, 'WebGPU init');

    const dpr = Math.min(globalThis.devicePixelRatio ?? 1, 2);
    renderer.setPixelRatio(dpr);
    renderer.setSize(globalThis.innerWidth, globalThis.innerHeight);
    renderer.setClearColor(0x050510);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    const post = new THREE_GPU.PostProcessing(renderer);
    let scenePass = null;
    let bloomPass = null;

    return {
        renderer,
        backend: 'webgpu',
        attach(scene, camera) {
            scenePass = pass(scene, camera);
            const color = scenePass.getTextureNode('output');
            bloomPass = bloom(color, 0.35, 0.5, 0.82);
            post.outputNode = color.add(bloomPass);
        },
        render() { post.render(); },
        resize(w, h) { renderer.setSize(w, h); },
        dispose() { renderer.dispose(); },
    };
}

function createWebGL(canvas) {
    const renderer = new THREE.WebGLRenderer({
        canvas,
        antialias: true,
        powerPreference: 'high-performance',
    });
    const dpr = Math.min(globalThis.devicePixelRatio ?? 1, 2);
    const w = globalThis.innerWidth;
    const h = globalThis.innerHeight;
    renderer.setPixelRatio(dpr);
    renderer.setSize(w, h);
    renderer.setClearColor(0x050510);
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    const composer = new EffectComposer(renderer);
    let renderPass = null;

    return {
        renderer,
        backend: 'webgl',
        attach(scene, camera) {
            if (renderPass) composer.removePass(renderPass);
            renderPass = new RenderPass(scene, camera);
            composer.addPass(renderPass);
            composer.addPass(new UnrealBloomPass(new THREE.Vector2(w, h), 0.9, 0.45, 0.82));
        },
        render() { composer.render(); },
        resize(nw, nh) {
            renderer.setSize(nw, nh);
            composer.setSize(nw, nh);
        },
        dispose() { renderer.dispose(); },
    };
}

/** WebGPU-first with timeout; reliable WebGL fallback. */
export async function createRenderer(canvas) {
    try {
        const gpu = await tryWebGPU(canvas);
        if (gpu) return gpu;
    } catch (err) {
        console.warn('[atlas] WebGPU unavailable, using WebGL:', err);
    }
    return createWebGL(canvas);
}