export const EXPERT_VERT = /* glsl */`
attribute float instanceGate;
attribute float instanceSel;

uniform float uTime;

varying float vGate;
varying float vSel;
varying vec3 vN;
varying vec3 vV;

void main() {
    vGate = instanceGate;
    vSel = instanceSel;

    vec3 p = position;
    float spark = sin(uTime * 6.0 + position.x * 3.0 + position.z * 2.0) * 0.04;
    float s = instanceGate < 0.01
        ? 0.42 + spark * 0.02
        : (0.62 + instanceGate * 0.95 + spark * instanceGate);
    p *= s;

    #ifdef USE_INSTANCING
        vec4 mv = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
        mat3 im = mat3(instanceMatrix);
    #else
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        mat3 im = mat3(1.0);
    #endif
    gl_Position = projectionMatrix * mv;
    vN = normalize(normalMatrix * im * normal);
    vV = -mv.xyz;
}
`;

export const EXPERT_FRAG = /* glsl */`
uniform float uTime;

varying float vGate;
varying float vSel;
varying vec3 vN;
varying vec3 vV;

void main() {
    float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 2.2);
    float g = vGate;
    float pulse = 0.88 + 0.12 * sin(uTime * 7.0 + g * 18.0);

    if (g < 0.01) {
        vec3 dormant = vec3(0.18, 0.08, 0.32);
        vec3 glow = dormant + vec3(0.15, 0.25, 0.55) * fres;
        gl_FragColor = vec4(glow, 0.38 + fres * 0.2);
        return;
    }

    vec3 cool = vec3(0.25, 0.55, 1.0);
    vec3 hot = vec3(1.0, 0.92, 0.55);
    vec3 col = mix(cool, hot, smoothstep(0.25, 1.0, g));
    vec3 emit = col * (0.55 + g * 1.35) * pulse;
    emit += vec3(0.7, 0.9, 1.0) * fres * (0.25 + g * 0.55);
    emit += vec3(0.5, 0.3, 1.0) * sin(uTime * 12.0 + g * 25.0) * 0.08 * g;

    if (vSel > 0.5) {
        emit += vec3(1.0, 0.4, 1.0) * 0.75;
    }

    gl_FragColor = vec4(emit, 0.72 + g * 0.28);
}
`;

export const ROUTER_VERT = /* glsl */`
attribute float instanceGate;
uniform float uTime;
varying float vGate;
varying vec3 vN;
varying vec3 vV;

void main() {
    vGate = max(instanceGate, 0.08);
    float pulse = 1.0 + sin(uTime * 4.0 + instanceGate * 16.0) * 0.08;
    float s = 0.72 + instanceGate * 1.05;
    vec3 p = position * s * pulse;
    #ifdef USE_INSTANCING
        vec4 mv = modelViewMatrix * instanceMatrix * vec4(p, 1.0);
        mat3 im = mat3(instanceMatrix);
    #else
        vec4 mv = modelViewMatrix * vec4(p, 1.0);
        mat3 im = mat3(1.0);
    #endif
    gl_Position = projectionMatrix * mv;
    vN = normalize(normalMatrix * im * normal);
    vV = -mv.xyz;
}
`;

export const ROUTER_FRAG = /* glsl */`
uniform float uTime;
varying float vGate;
varying vec3 vN;
varying vec3 vV;

void main() {
    float fres = pow(1.0 - abs(dot(normalize(vN), normalize(vV))), 1.4);
    float core = 1.0 - fres * 0.65;
    float g = vGate;
    float crackle = sin(uTime * 14.0 + g * 30.0) * 0.5 + 0.5;

    vec3 dormant = vec3(0.35, 0.15, 0.55);
    vec3 firing = vec3(0.75, 0.92, 1.0);
    vec3 col = mix(dormant, firing, smoothstep(0.1, 0.85, g));
    vec3 emit = col * core * (0.7 + g * 1.1);
    emit += vec3(0.45, 0.65, 1.0) * fres * (0.55 + g * 0.9);
    emit += vec3(1.0, 1.0, 1.0) * crackle * 0.12 * g;

    float alpha = 0.55 + g * 0.4 + fres * 0.35;
    gl_FragColor = vec4(emit, alpha);
}
`;

export const ROUTER_HALO_VERT = /* glsl */`
attribute float instanceGate;
uniform float uTime;
varying float vGate;

void main() {
    vGate = max(instanceGate, 0.06);
    float s = 1.55 + instanceGate * 1.35 + sin(uTime * 3.0) * 0.06;
    vec3 p = position * s;
    #ifdef USE_INSTANCING
        gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(p, 1.0);
    #else
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    #endif
}
`;

export const ROUTER_HALO_FRAG = /* glsl */`
uniform float uTime;
varying float vGate;

void main() {
    float g = vGate;
    float pulse = 0.5 + 0.5 * sin(uTime * 5.0 + g * 20.0);
    vec3 col = mix(vec3(0.3, 0.1, 0.5), vec3(0.4, 0.85, 1.0), g);
    float alpha = (0.12 + g * 0.28) * pulse;
    gl_FragColor = vec4(col, alpha);
}
`;

export const HUB_VERT = /* glsl */`
attribute float instanceWeight;
uniform float uTime;
varying float vW;

void main() {
    vW = instanceWeight;
    float s = 0.35 + instanceWeight * 0.85 + sin(uTime * 5.0) * 0.05 * instanceWeight;
    vec3 p = position * s;
    #ifdef USE_INSTANCING
        gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(p, 1.0);
    #else
        gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    #endif
}
`;

export const HUB_FRAG = /* glsl */`
uniform float uTime;
varying float vW;

void main() {
    if (vW < 0.02) discard;
    float pulse = 0.85 + 0.15 * sin(uTime * 8.0 + vW * 22.0);
    vec3 col = mix(vec3(0.4, 0.2, 0.7), vec3(0.6, 0.95, 1.0), vW);
    gl_FragColor = vec4(col * pulse * 1.4, 0.65 + vW * 0.35);
}
`;

export const LINK_VERT = /* glsl */`
attribute float aKind;
attribute float aWeight;
attribute float aAlong;

varying float vKind;
varying float vWeight;
varying float vAlong;

void main() {
    vKind = aKind;
    vWeight = aWeight;
    vAlong = aAlong;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const LINK_FRAG = /* glsl */`
uniform float uTime;
varying float vKind;
varying float vWeight;
varying float vAlong;

void main() {
    float w = clamp(vWeight, 0.1, 1.0);
    float bolt = smoothstep(0.0, 0.12, fract(vAlong * 2.5 - uTime * 1.8));
    bolt *= smoothstep(0.35, 0.05, fract(vAlong * 2.5 - uTime * 1.8));

    vec3 synapse = mix(vec3(0.25, 0.12, 0.45), vec3(0.35, 0.75, 1.0), w);
    vec3 spark = vec3(0.85, 0.95, 1.0) * bolt;
    vec3 col = synapse * (0.35 + w * 0.45) + spark;

    float alpha = vKind < 0.5
        ? 0.28 + w * 0.35 + bolt * 0.55
        : 0.22 + w * 0.3 + bolt * 0.4;
    gl_FragColor = vec4(col, alpha);
}
`;

export const SPARK_VERT = /* glsl */`
uniform float uTime;
varying vec3 vPos;

void main() {
    vPos = position;
    #ifdef USE_INSTANCING
        vec4 mv = modelViewMatrix * instanceMatrix * vec4(position, 1.0);
    #else
        vec4 mv = modelViewMatrix * vec4(position, 1.0);
    #endif
    gl_Position = projectionMatrix * mv;
}
`;

export const SPARK_FRAG = /* glsl */`
uniform float uTime;
varying vec3 vPos;

void main() {
    float flicker = 0.7 + 0.3 * sin(uTime * 20.0 + vPos.x * 10.0);
    vec3 col = vec3(0.75, 0.92, 1.0) * flicker;
    gl_FragColor = vec4(col, 0.92);
}
`;