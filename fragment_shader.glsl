#version 330 core

in vec3 fragNormal;
out vec4 outColor;

// 簡単な平行光源(ライト方向は固定)を想定
uniform vec3 uLightDir;   // ライト方向(正規化済み想定)
uniform vec3 uColor;      // 点の基本色など (例: 白色)

void main() {
    // 拡散反射の簡単な計算 (N・L)
    float diff = max(dot(normalize(fragNormal), -uLightDir), 0.0);

    // ライト強度(ここでは適当に拡散反射のみ)
    vec3 color = uColor * diff;

    outColor = vec4(color, 1.0);
}
