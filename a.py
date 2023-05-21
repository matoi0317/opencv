# -*- coding: utf-8 -*-
import cv2

#サングラスプログラム
# Webカメラデバイスにアクセス
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラデバイスを指定します

# 透過PNG画像の読み込み
foreground_img = cv2.imread('sunglass_normal.png', cv2.IMREAD_UNCHANGED)

# フレームの幅と高さを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
        # 両目の領域を計算
        eye_region_width = int(w * 1.0)  # 両目の領域の幅を調整
        eye_region_height = int(h * 0.3)  # 両目の領域の高さを調整
        eye_region_x = int(x + (w - eye_region_width) / 2)  # 両目の領域の中央に位置するよう調整
        eye_region_y = int(y + h * 0.25)  # 両目の領域のY座標を調整
        # # 両目の領域を計算
        # eye_region_x = int(x + w * 0.15)
        # eye_region_y = int(y + h * 0.15)
        # eye_region_width = int(w * 1.2)
        # eye_region_height = int(h * 0.35)

        # 両目の領域に合わせて透過PNG画像をリサイズ
        resized_foreground_img = cv2.resize(foreground_img, (eye_region_width, eye_region_height))

        # 合成処理
        alpha = resized_foreground_img[:, :, 3] / 255.0
        for c in range(3):
            frame[eye_region_y:eye_region_y+eye_region_height, eye_region_x:eye_region_x+eye_region_width, c] = \
                frame[eye_region_y:eye_region_y+eye_region_height, eye_region_x:eye_region_x+eye_region_width, c] * (1 - alpha) + \
                resized_foreground_img[:, :, c] * alpha

    # フレームを表示
    cv2.imshow('Frame', frame)

    # 'q'キーを押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウの破棄
cap.release()
cv2.destroyAllWindows()
