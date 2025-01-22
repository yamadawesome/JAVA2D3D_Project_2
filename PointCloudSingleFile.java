import org.lwjgl.*;
import org.lwjgl.glfw.*;
import org.lwjgl.opengl.*;
import org.lwjgl.system.*;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.nio.*;
import java.nio.file.*;
import java.util.ArrayList;
import java.util.List;

import static org.lwjgl.glfw.Callbacks.glfwFreeCallbacks;
import static org.lwjgl.glfw.GLFW.*;
import static org.lwjgl.opengl.GL41.*;
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

public class PointCloudViewer {
    // ウィンドウハンドル
    private long window;

    // シェーダープログラムID
    private int shaderProgram;

    // VAO, VBO
    private int vaoId;
    private int vboId;

    // 読み込んだ点群データ
    private FloatBuffer pointCloudBuffer;
    private int pointCount;

    // バウンディングボックス
    private float minX = Float.POSITIVE_INFINITY, maxX = Float.NEGATIVE_INFINITY;
    private float minY = Float.POSITIVE_INFINITY, maxY = Float.NEGATIVE_INFINITY;
    private float minZ = Float.POSITIVE_INFINITY, maxZ = Float.NEGATIVE_INFINITY;

    // 画面サイズ
    private static final int WIDTH = 800;
    private static final int HEIGHT = 800;

    // シェーダーファイルのパス
    private static final String VERTEX_SHADER_PATH   = "vertex_shader.glsl";
    private static final String FRAGMENT_SHADER_PATH = "fragment_shader.glsl";

    // 点群ファイルのパス (xyz フォーマット: x, y, z, nx, ny, nz)
    private static final String DATA_PATH = "../xyz/armadillo.xyz";

    public static void main(String[] args) {
        new PointCloudViewer().run();
    }

    public void run() {
        initWindow();          // GLFW ウィンドウと OpenGL コンテキストの初期化
        initOpenGLResources(); // VBO/VAO, シェーダーなどの初期化
        loop();                // メインループ
        cleanup();             // リソースの解放
    }

    /**
     * GLFW ウィンドウと OpenGL コンテキストの初期化
     */
    private void initWindow() {
        if (!glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }

        // OpenGL バージョン指定 (3.3 Core 例)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        // macOS の場合は必要
        // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

        // ウィンドウ作成
        window = glfwCreateWindow(WIDTH, HEIGHT, "LWJGL PointCloud", NULL, NULL);
        if (window == NULL) {
            throw new RuntimeException("Failed to create the GLFW window");
        }

        glfwMakeContextCurrent(window);
        glfwSwapInterval(1);
        glfwShowWindow(window);
    }

    /**
     * VBO/VAO, シェーダー等 OpenGL 資源の初期化
     */
    private void initOpenGLResources() {
        // LWJGL で OpenGL 関数を利用できるように
        GL.createCapabilities();

        // 点群ファイル読み込み (バウンディングボックスも算出)
        loadPointCloud(DATA_PATH);

        // == VAO と VBO ==
        vaoId = glGenVertexArrays();
        glBindVertexArray(vaoId);

        vboId = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboId);
        glBufferData(GL_ARRAY_BUFFER, pointCloudBuffer, GL_STATIC_DRAW);

        // 頂点属性: 位置 (location=0, vec3)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 6 * Float.BYTES, 0L);

        // 頂点属性: 法線 (location=1, vec3)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 6 * Float.BYTES, 3L * Float.BYTES);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // == シェーダープログラム ==
        int vs = createShader(loadTextFile(VERTEX_SHADER_PATH), GL_VERTEX_SHADER);
        int fs = createShader(loadTextFile(FRAGMENT_SHADER_PATH), GL_FRAGMENT_SHADER);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vs);
        glAttachShader(shaderProgram, fs);
        glLinkProgram(shaderProgram);
        if (glGetProgrami(shaderProgram, GL_LINK_STATUS) == GL_FALSE) {
            System.err.println("ERROR: Shader linking failed.");
            System.err.println(glGetProgramInfoLog(shaderProgram));
            throw new RuntimeException("Shader linking failed");
        }
        glDetachShader(shaderProgram, vs);
        glDetachShader(shaderProgram, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);

        glEnable(GL_DEPTH_TEST);
        glClearColor(0f, 0f, 0f, 1f);
        glPointSize(3.0f);

        // === ここで一度だけ MVP 行列を計算して uniform に設定する ===
        glUseProgram(shaderProgram);
        int uMVP      = glGetUniformLocation(shaderProgram, "uMVP");
        int uLightDir = glGetUniformLocation(shaderProgram, "uLightDir");
        int uColor    = glGetUniformLocation(shaderProgram, "uColor");

        // ライト方向 / 色
        glUniform3f(uLightDir, 0.0f, 0.0f, 1.0f);
        glUniform3f(uColor, 1.0f, 1.0f, 1.0f);

        // バウンディングボックスの中心と最大半径を計算
        float centerX = (minX + maxX) * 0.5f;
        float centerY = (minY + maxY) * 0.5f;
        float centerZ = (minZ + maxZ) * 0.5f;
        float halfX   = (maxX - minX) * 0.5f;
        float halfY   = (maxY - minY) * 0.5f;
        float halfZ   = (maxZ - minZ) * 0.5f;
        // "モデル空間" で最大幅
        float maxHalf = Math.max(halfX, Math.max(halfY, halfZ));
        
        // 1) モデル行列 (原点へ平行移動 → 全体を [-1,1] に収めるようスケール)
        float[] model = buildModelMatrix(centerX, centerY, centerZ, 1.0f / maxHalf);

        // 2) 投影行列 (正射影: -1～+1 に収まる想定)
        //   → 画面に収まりきるようにする。Zも -1～+1 でOKならこれで十分。
        //   必要に応じて near/far を大きめにしても構いません
        float[] proj = buildOrthoMatrix(-1f, 1f, -1f, 1f, -1f, 1f);
        //   あるいは透視投影にしたい場合は buildPerspectiveMatrix(...) などを使う

        // MVP = proj * model
        float[] mvp  = multiply4x4(proj, model);

        // uniformへ送信
        try (MemoryStack stack = stackPush()) {
            FloatBuffer fb = stack.mallocFloat(16);
            fb.put(mvp).flip();
            glUniformMatrix4fv(uMVP, false, fb);
        }
    }

    /**
     * メインループ
     */
    private void loop() {
        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // バインド
            glBindVertexArray(vaoId);
            glUseProgram(shaderProgram);

            // 点群描画
            glDrawArrays(GL_POINTS, 0, pointCount);

            // バッファ交換 & イベント
            glfwSwapBuffers(window);
            glfwPollEvents();
        }
    }

    /**
     * 終了時のリソース解放
     */
    private void cleanup() {
        glDeleteBuffers(vboId);
        glDeleteVertexArrays(vaoId);
        glDeleteProgram(shaderProgram);

        glfwFreeCallbacks(window);
        glfwDestroyWindow(window);
        glfwTerminate();
    }

    /**
     * ファイル(x, y, z, nx, ny, nz)を読み込み、FloatBufferに格納しつつBBを計算
     */
    private void loadPointCloud(String dataPath) {
        List<float[]> dataList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(dataPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                // 空白区切り
                String[] tokens = line.trim().split("\\s+");
                if (tokens.length < 6) {
                    continue;
                }
                float x  = Float.parseFloat(tokens[0]);
                float y  = Float.parseFloat(tokens[1]);
                float z  = Float.parseFloat(tokens[2]);
                float nx = Float.parseFloat(tokens[3]);
                float ny = Float.parseFloat(tokens[4]);
                float nz = Float.parseFloat(tokens[5]);

                // バウンディングボックスの更新
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (z < minZ) minZ = z;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                if (z > maxZ) maxZ = z;

                dataList.add(new float[]{x, y, z, nx, ny, nz});
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        pointCount = dataList.size();
        pointCloudBuffer = memAllocFloat(pointCount * 6);
        for (float[] arr : dataList) {
            pointCloudBuffer.put(arr);
        }
        pointCloudBuffer.flip();
    }

    /**
     * テキストファイル(シェーダー)を読み込み
     */
    private String loadTextFile(String path) {
        try {
            return new String(Files.readAllBytes(Paths.get(path)));
        } catch (IOException e) {
            e.printStackTrace();
            return "";
        }
    }

    /**
     * シェーダーをコンパイル
     */
    private int createShader(String source, int type) {
        int shaderId = glCreateShader(type);
        glShaderSource(shaderId, source);
        glCompileShader(shaderId);

        if (glGetShaderi(shaderId, GL_COMPILE_STATUS) == GL_FALSE) {
            System.err.println("ERROR: Shader compilation failed.");
            System.err.println(glGetShaderInfoLog(shaderId));
            throw new RuntimeException("Shader compile failed");
        }
        return shaderId;
    }

    // -------------------------------------------------------
    //  以下、行列ユーティリティ
    // -------------------------------------------------------

    /**
     * モデル行列を作る:
     *   1) 平行移動( -center )
     *   2) スケール( scale )  （全方向同じ倍率）
     */
    private float[] buildModelMatrix(float cx, float cy, float cz, float scale) {
        // 4x4 の単位行列
        float[] mat = identity4x4();

        // (1) 平行移動
        mat = translate4x4(mat, -cx, -cy, -cz);
        // (2) スケール
        mat = scale4x4(mat, scale, scale, scale);

        return mat;
    }

    /**
     * 正射影行列を作る: Ortho(left, right, bottom, top, near, far)
     */
    private float[] buildOrthoMatrix(float left, float right, float bottom,
                                     float top, float near, float far) {
        float[] m = new float[16];
        for(int i=0;i<16;i++) m[i]=0f;
        m[0] = 2f / (right - left);
        m[5] = 2f / (top - bottom);
        m[10] = -2f / (far - near);
        m[12] = -(right + left) / (right - left);
        m[13] = -(top + bottom) / (top - bottom);
        m[14] = -(far + near)   / (far - near);
        m[15] = 1f;
        return m;
    }

    // -------------------------------------------------------
    // 4x4 行列の操作 (簡易実装)
    // -------------------------------------------------------
    private float[] identity4x4() {
        float[] m = new float[16];
        for(int i=0; i<16; i++){
            m[i] = (i % 5 == 0)? 1f : 0f;
        }
        return m;
    }
    private float[] translate4x4(float[] m, float tx, float ty, float tz) {
        // m * translation
        float[] t = identity4x4();
        t[12] = tx;
        t[13] = ty;
        t[14] = tz;
        return multiply4x4(m, t);
    }
    private float[] scale4x4(float[] m, float sx, float sy, float sz) {
        // m * scale
        float[] s = identity4x4();
        s[0] = sx;
        s[5] = sy;
        s[10] = sz;
        return multiply4x4(m, s);
    }

    private float[] multiply4x4(float[] A, float[] B) {
        float[] out = new float[16];
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                out[i*4+j] =
                    A[i*4+0] * B[0*4+j] +
                    A[i*4+1] * B[1*4+j] +
                    A[i*4+2] * B[2*4+j] +
                    A[i*4+3] * B[3*4+j];
            }
        }
        return out;
    }
}
