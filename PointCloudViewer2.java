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

public class PointCloudViewer2 {
    // --- RBF 関連 ---
    private RBF rbf; 

    // ウィンドウハンドル
    private long window;

    // シェーダープログラムID
    private int shaderProgram;

    // VAO, VBO
    private int vaoId;
    private int vboId;

    // 読み込んだ点群データ (OpenGLで描画する用)
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

    // 点群ファイルのパス
    private static final String DATA_PATH = "../xyz/armadillo.xyz";

    public static void main(String[] args) {
        new PointCloudViewer2().run();
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

        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

        // ウィンドウ作成
        window = glfwCreateWindow(WIDTH, HEIGHT, "PointCloudViewer + RBF", NULL, NULL);
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
        GL.createCapabilities();

        // === RBF: まずは点群を読み込み、RBFに渡すサンプルを用意 ===
        //   (オンサーフェス + オフサーフェス)
        List<float[]> rawDataList = loadPointCloud(DATA_PATH);

        // RBF 用: SamplePoint のリスト
        List<RBF.SamplePoint> sampleList = new ArrayList<>();

        float EPS = 0.01f;   // オフサーフェス用オフセット
        float OFFSET_VAL = 0.01f; // fの値

        for (float[] arr : rawDataList) {
            float x  = arr[0];
            float y  = arr[1];
            float z  = arr[2];
            float nx = arr[3];
            float ny = arr[4];
            float nz = arr[5];

            // オンサーフェス -> f=0
            sampleList.add(new RBF.SamplePoint(x, y, z, 0.0));

            // 法線を正規化して外側/内側にオフセット
            float len = (float)Math.sqrt(nx*nx + ny*ny + nz*nz);
            if (len > 1e-9f) {
                nx /= len; 
                ny /= len; 
                nz /= len;

                // 外側 (f= +0.01など)
                sampleList.add(new RBF.SamplePoint(
                    x + EPS*nx, y + EPS*ny, z + EPS*nz,
                    +OFFSET_VAL
                ));
                // 内側 (f= -0.01など)
                sampleList.add(new RBF.SamplePoint(
                    x - EPS*nx, y - EPS*ny, z - EPS*nz,
                    -OFFSET_VAL
                ));
            }
        }

        // RBF のインスタンス生成 & build
        rbf = new RBF();
        rbf.buildRBF(sampleList);
        System.out.println("RBF build done! sample size = " + sampleList.size());

        // === OpenGL で表示するためのバッファ (点群の生データ) ===
        //   今回は「元の点群 (オンサーフェス)」をそのまま描画
        //   (オフサーフェス点は表示しない)
        createVBO(rawDataList);

        // シェーダープログラム等
        int vs = createShader(loadTextFile(VERTEX_SHADER_PATH), GL_VERTEX_SHADER);
        int fs = createShader(loadTextFile(FRAGMENT_SHADER_PATH), GL_FRAGMENT_SHADER);
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vs);
        glAttachShader(shaderProgram, fs);
        glLinkProgram(shaderProgram);

        glDetachShader(shaderProgram, vs);
        glDetachShader(shaderProgram, fs);
        glDeleteShader(vs);
        glDeleteShader(fs);

        glEnable(GL_DEPTH_TEST);
        glClearColor(1f, 1f, 1f, 1f);
        glPointSize(5.0f);

        // MVP行列を設定
        glUseProgram(shaderProgram);
        int uMVP      = glGetUniformLocation(shaderProgram, "uMVP");
        int uLightDir = glGetUniformLocation(shaderProgram, "uLightDir");
        int uColor    = glGetUniformLocation(shaderProgram, "uColor");

        glUniform3f(uLightDir, 0.0f, 0.0f, 1.0f);
        glUniform3f(uColor, 1.0f, 1.0f, 1.0f);

        // バウンディングボックスからモデル行列 & 投影行列を作る
        float centerX = (minX + maxX)*0.5f;
        float centerY = (minY + maxY)*0.5f;
        float centerZ = (minZ + maxZ)*0.5f;
        float halfX   = (maxX - minX)*0.5f;
        float halfY   = (maxY - minY)*0.5f;
        float halfZ   = (maxZ - minZ)*0.5f;
        float maxHalf = Math.max(halfX, Math.max(halfY, halfZ));

        float[] model = buildModelMatrix(centerX, centerY, centerZ, 1.0f / maxHalf);
        float[] proj  = buildOrthoMatrix(-1f, 1f, -1f, 1f, -1f, 1f);
        float[] mvp   = multiply4x4(proj, model);

        // uniformへ送信
        try (MemoryStack stack = stackPush()) {
            FloatBuffer fb = stack.mallocFloat(16);
            fb.put(mvp).flip();
            glUniformMatrix4fv(uMVP, false, fb);
        }
    }

    /**
     * 点群 → VAO/VBO を作成する
     */
    private void createVBO(List<float[]> rawDataList) {
        // 頂点バッファ
        pointCount = rawDataList.size();
        pointCloudBuffer = memAllocFloat(pointCount * 6);
        for (float[] arr : rawDataList) {
            pointCloudBuffer.put(arr);
        }
        pointCloudBuffer.flip();

        // VAO
        vaoId = glGenVertexArrays();
        glBindVertexArray(vaoId);

        // VBO
        vboId = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboId);
        glBufferData(GL_ARRAY_BUFFER, pointCloudBuffer, GL_STATIC_DRAW);

        // attribute 0 -> (x,y,z)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 6*Float.BYTES, 0L);

        // attribute 1 -> (nx, ny, nz)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 6*Float.BYTES, 3L*Float.BYTES);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);
    }

    /**
     * メインループ
     */
    private void loop() {
        while (!glfwWindowShouldClose(window)) {
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glBindVertexArray(vaoId);
            glUseProgram(shaderProgram);

            // 点群描画 (GL_POINTS)
            glDrawArrays(GL_POINTS, 0, pointCount);

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
     * (x, y, z, nx, ny, nz) を読み込み、List<float[]> として返す。
     * バウンディングボックスも更新。
     */
    private List<float[]> loadPointCloud(String dataPath) {
        List<float[]> dataList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(dataPath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                if (tokens.length < 6) continue;

                float x  = Float.parseFloat(tokens[0]);
                float y  = Float.parseFloat(tokens[1]);
                float z  = Float.parseFloat(tokens[2]);
                float nx = Float.parseFloat(tokens[3]);
                float ny = Float.parseFloat(tokens[4]);
                float nz = Float.parseFloat(tokens[5]);

                // バウンディングボックス更新
                if (x<minX) minX=x; if (x>maxX) maxX=x;
                if (y<minY) minY=y; if (y>maxY) maxY=y;
                if (z<minZ) minZ=z; if (z>maxZ) maxZ=z;

                dataList.add(new float[]{x,y,z,nx,ny,nz});
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        System.out.println("Point count: " + dataList.size());
        return dataList;
    }

    // シェーダーファイル読み込み
    private String loadTextFile(String path) {
        try {
            return new String(Files.readAllBytes(Paths.get(path)));
        } catch (IOException e) {
            e.printStackTrace();
            return "";
        }
    }

    // シェーダーコンパイル
    private int createShader(String source, int type) {
        int id = glCreateShader(type);
        glShaderSource(id, source);
        glCompileShader(id);
        if (glGetShaderi(id, GL_COMPILE_STATUS) == GL_FALSE) {
            System.err.println("Shader compile failed: " + glGetShaderInfoLog(id));
            throw new RuntimeException("Shader compile failed");
        }
        return id;
    }

    // --------------------------------------------------------------------------------
    //  以下、行列ユーティリティ (前回例と同様)
    // --------------------------------------------------------------------------------
    private float[] buildModelMatrix(float cx, float cy, float cz, float scale) {
        float[] mat = identity4x4();
        mat = translate4x4(mat, -cx, -cy, -cz);
        mat = scale4x4(mat, scale, scale, scale);
        return mat;
    }

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

    private float[] identity4x4() {
        float[] m = new float[16];
        for(int i=0; i<16; i++){
            m[i] = (i%5==0)?1f:0f;
        }
        return m;
    }
    private float[] translate4x4(float[] A, float tx, float ty, float tz) {
        float[] T = identity4x4();
        T[12]=tx; T[13]=ty; T[14]=tz;
        return multiply4x4(A, T);
    }
    private float[] scale4x4(float[] A, float sx, float sy, float sz) {
        float[] S = identity4x4();
        S[0]=sx; S[5]=sy; S[10]=sz;
        return multiply4x4(A, S);
    }
    private float[] multiply4x4(float[] A, float[] B) {
        float[] out = new float[16];
        for(int i=0; i<4; i++){
            for(int j=0; j<4; j++){
                out[i*4+j] = 
                    A[i*4+0]*B[j+ 0] +
                    A[i*4+1]*B[j+ 4] +
                    A[i*4+2]*B[j+ 8] +
                    A[i*4+3]*B[j+12];
            }
        }
        return out;
    }
}

/* 
 * REFERENCE
 * https://www.lwjgl.org/guide
 * https://www.glfw.org/docs/latest/window_guide.html
 * https://www.glfw.org/docs/latest/input_guide.html
 * https://zenryokuservice.com/wp/2020/05/05/java-3d-lwjgl-〜tutorial-1-windowを表示する〜/
 * 
 */
