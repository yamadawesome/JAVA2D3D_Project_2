#version 330 core

layout (location = 0) in vec3 inPosition; // 点の座標
layout (location = 1) in vec3 inNormal;   // 法線ベクトル

out vec3 fragNormal;

uniform mat4 uMVP;  // MVP行列 (今はほぼ固定でも良い)

void main() {
    // 座標変換
    gl_Position = uMVP * vec4(inPosition, 1.0);
    // 法線をそのままパススルー
    fragNormal = inNormal;
}
