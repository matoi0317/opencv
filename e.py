# -*- coding: utf-8 -*-
import cv2
import random

# Webカメラデバイスにアクセス
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラデバイスを指定します

# 透過PNG画像の読み込み
overlay_image = cv2.imread('nekomimi.png', cv2.IMREAD_UNCHANGED)

# 画像の幅と高さを取得
overlay_width = overlay_image.shape[1]
overlay_height = overlay_image.shape[0]

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
        # 頭の上に透過PNG画像を合成
        overlay_x = x + int((w - overlay_width) / 2)
        overlay_y = y - overlay_height
        if overlay_y > 0:
            # 画像が画面内に収まる場合のみ合成
            alpha = overlay_image[:, :, 3] / 255.0
            for c in range(3):
                frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width, c] = \
                    frame[overlay_y:overlay_y+overlay_height, overlay_x:overlay_x+overlay_width, c] * (1 - alpha) + \
                    overlay_image[:, :, c] * alpha

    # フレームを表示
    cv2.imshow('Frame', frame)

    # 'q'キーを押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウの破棄
cap.release()
cv2.destroyAllWindows()