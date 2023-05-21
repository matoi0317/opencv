import cv2
import random

#猫耳プログラムサイズ調整板
# Webカメラデバイスにアクセス
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラデバイスを指定します

# 透過PNG画像の読み込み
overlay_image = cv2.imread('nekomimi.png', cv2.IMREAD_UNCHANGED)

# 顔検出器の初期化
cascade_file = "haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_file)

# ループでフレームを繰り返し取得
while True:
    # フレームをキャプチャ
    ret, frame = cap.read()

    # グレースケールに変換
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 顔の検出
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 各顔に透過PNG画像を合成
    for (x, y, w, h) in faces:
        # 顔の横幅に合わせて透過PNG画像をリサイズ
        resized_overlay_image = cv2.resize(overlay_image, (w, int(w * overlay_image.shape[0] / overlay_image.shape[1])))

        # 頭の上に透過PNG画像を合成
        overlay_x = x
        overlay_y = y - resized_overlay_image.shape[0]
        if overlay_y > 0:
            # 画像が画面内に収まる場合のみ合成
            alpha = resized_overlay_image[:, :, 3] / 255.0
            for c in range(3):
                frame[overlay_y:overlay_y + resized_overlay_image.shape[0], overlay_x:overlay_x + w, c] = \
                    frame[overlay_y:overlay_y + resized_overlay_image.shape[0], overlay_x:overlay_x + w, c] * (1 - alpha) + \
                    resized_overlay_image[:, :, c] * alpha

    # フレームを表示
    cv2.imshow('Frame', frame)

    # 'q'キーを押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウの破棄
cap.release()
cv2.destroyAllWindows()
