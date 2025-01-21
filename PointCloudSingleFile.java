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
import static org.lwjgl.opengl.GL41.*;  // 3.x / 4.x のコアプロファイルに合わせる
import static org.lwjgl.system.MemoryStack.*;
import static org.lwjgl.system.MemoryUtil.*;

public class PointCloudSingleFile {
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

    // 画面サイズ
    private static final int WIDTH = 800;
    private static final int HEIGHT = 800;

    // 例: シェーダーファイルのパス (環境に合わせて修正)
    private static final String VERTEX_SHADER_PATH   = "vertex_shader.glsl";
    private static final String FRAGMENT_SHADER_PATH = "fragment_shader.glsl";
    // 例: ファイルのパス
    private static final String FILE_PATH = "../xyz/armadillo.xyz";

    public static void main(String[] args) {
        new PointCloudSingleFile().run();
    }

    public void run() {
        initWindow();        // GLFW ウィンドウと OpenGL コンテキストの初期化
        initOpenGLResources(); // VBO/VAO, シェーダーなどの初期化
        loop();              // メインループ
        cleanup();           // リソースの解放
    }

    /**
     * GLFW ウィンドウと OpenGL コンテキストの初期化
     */
    private void initWindow() {
        // GLFW の初期化
        if (!glfwInit()) {
            throw new IllegalStateException("Unable to initialize GLFW");
        }

        // OpenGL バージョン指定 (例: 3.3 Core)
        glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
        glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
        glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
        // macOS の場合は必須
        // glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

        // ウィンドウ作成
        window = glfwCreateWindow(WIDTH, HEIGHT, "LWJGL + GLFW PointCloud", NULL, NULL);
        if (window == NULL) {
            throw new RuntimeException("Failed to create the GLFW window");
        }

        // OpenGL コンテキストを現在のスレッドに紐づけ
        glfwMakeContextCurrent(window);

        // 垂直同期を有効にする (FPS制限)
        glfwSwapInterval(1);

        // ウィンドウ表示
        glfwShowWindow(window);
    }

    /**
     * VBO/VAO, シェーダー等 OpenGL 資源の初期化
     */
    private void initOpenGLResources() {
        // LWJGL で OpenGL 関数を利用できるようにする
        GL.createCapabilities();

        // ファイル読み込み
        loadPointCloud(FILE_PATH);

        // === VAO と VBO の設定 ===
        // VAO の作成
        vaoId = glGenVertexArrays();
        glBindVertexArray(vaoId);

        // VBO の作成
        vboId = glGenBuffers();
        glBindBuffer(GL_ARRAY_BUFFER, vboId);

        // 点群データ (位置x,y,z + 法線nx,ny,nz = float6つ) を転送
        glBufferData(GL_ARRAY_BUFFER, pointCloudBuffer, GL_STATIC_DRAW);

        // 頂点属性: location = 0 → 位置(vec3)
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 3, GL_FLOAT, false, 6 * Float.BYTES, 0L);

        // 頂点属性: location = 1 → 法線(vec3)
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 3, GL_FLOAT, false, 6 * Float.BYTES, 3L * Float.BYTES);

        // バインド解除
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        glBindVertexArray(0);

        // === シェーダープログラムのコンパイル ===
        int vertexShader   = createShader(loadTextFile(VERTEX_SHADER_PATH), GL_VERTEX_SHADER);
        int fragmentShader = createShader(loadTextFile(FRAGMENT_SHADER_PATH), GL_FRAGMENT_SHADER);

        // シェーダープログラムをリンク
        shaderProgram = glCreateProgram();
        glAttachShader(shaderProgram, vertexShader);
        glAttachShader(shaderProgram, fragmentShader);
        glLinkProgram(shaderProgram);
        // リンクエラーチェック
        if (glGetProgrami(shaderProgram, GL_LINK_STATUS) == GL_FALSE) {
            System.err.println("ERROR: Shader Program linking failed.");
            System.err.println(glGetProgramInfoLog(shaderProgram));
            throw new RuntimeException("Shader linking failed");
        }

        // シェーダーはリンク後に不要なので削除
        glDetachShader(shaderProgram, vertexShader);
        glDetachShader(shaderProgram, fragmentShader);
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);

        // OpenGL の基本設定
        glEnable(GL_DEPTH_TEST);
        // 背景色
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        // 点サイズ (見やすく)
        glPointSize(5.0f);
    }

    /**
     * メインループ
     */
    private void loop() {
        // シェーダーで使う uniform の場所を取得
        glUseProgram(shaderProgram);
        int uMVP      = glGetUniformLocation(shaderProgram, "uMVP");
        int uLightDir = glGetUniformLocation(shaderProgram, "uLightDir");
        int uColor    = glGetUniformLocation(shaderProgram, "uColor");

        // MVP行列を適当に単位行列として設定(拡張したい場合は行列計算ライブラリ等で計算可)
        try (MemoryStack stack = stackPush()) {
            FloatBuffer identity = stack.mallocFloat(16);
            identity.put(new float[]{
                    1, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1
            }).flip();
            glUniformMatrix4fv(uMVP, false, identity);
        }

        // ライト方向 (例: Zマイナス方向から来る → (0, 0, 1))
        glUniform3f(uLightDir, 0.0f, 0.0f, 1.0f);

        // オブジェクトの色 (例: 白)
        glUniform3f(uColor, 1.0f, 1.0f, 1.0f);

        // メインループ
        while (!glfwWindowShouldClose(window)) {
            // 背景/深度バッファをクリア
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // VAO バインド
            glBindVertexArray(vaoId);
            // シェーダープログラム使用
            glUseProgram(shaderProgram);

            // 点群を描画
            glDrawArrays(GL_POINTS, 0, pointCount);

            // バインド解除 (一応)
            glBindVertexArray(0);

            // ダブルバッファ交換
            glfwSwapBuffers(window);
            // イベントポーリング
            glfwPollEvents();
        }
    }

    /**
     * GLFW と OpenGL のリソース解放
     */
    private void cleanup() {
        // VBO/VAO 解放
        glDeleteBuffers(vboId);
        glDeleteVertexArrays(vaoId);

        // シェーダープログラム解放
        glDeleteProgram(shaderProgram);

        // ウィンドウコールバック解放
        glfwFreeCallbacks(window);
        // ウィンドウ破棄
        glfwDestroyWindow(window);
        // GLFW 終了
        glfwTerminate();
    }

    /**
     * ファイル (x, y, z, nx, ny, nz) を読み込み、FloatBuffer に格納する
     */
    private void loadPointCloud(String filePath) {
        List<float[]> dataList = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(filePath))) {
            String line;
            while ((line = br.readLine()) != null) {
                String[] tokens = line.trim().split("\\s+");
                if (tokens.length < 6) {
                    continue; // 不正行はスキップ
                }
                float x  = Float.parseFloat(tokens[0].trim());
                float y  = Float.parseFloat(tokens[1].trim());
                float z  = Float.parseFloat(tokens[2].trim());
                float nx = Float.parseFloat(tokens[3].trim());
                float ny = Float.parseFloat(tokens[4].trim());
                float nz = Float.parseFloat(tokens[5].trim());
                dataList.add(new float[]{x, y, z, nx, ny, nz});
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        pointCount = dataList.size();
        // FloatBuffer の確保
        pointCloudBuffer = memAllocFloat(pointCount * 6);

        for (float[] arr : dataList) {
            pointCloudBuffer.put(arr);
        }
        pointCloudBuffer.flip(); // バッファを読み込みモードに切り替え
    }

    /**
     * テキストファイルを読み込むユーティリティ
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
     * シェーダーをコンパイルするユーティリティ
     */
    private int createShader(String source, int shaderType) {
        int shaderId = glCreateShader(shaderType);
        glShaderSource(shaderId, source);
        glCompileShader(shaderId);

        // コンパイルチェック
        int status = glGetShaderi(shaderId, GL_COMPILE_STATUS);
        if (status == GL_FALSE) {
            System.err.println("ERROR: Shader compilation failed.");
            System.err.println(glGetShaderInfoLog(shaderId));
            throw new RuntimeException("Shader compile failed");
        }
        return shaderId;
    }
}
