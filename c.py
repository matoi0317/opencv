# -*- coding: utf-8 -*-
import cv2
import random
import time

# Webカメラデバイスにアクセス
cap = cv2.VideoCapture(0)  # 0はデフォルトのカメラデバイスを指定します

# 透過PNG画像の読み込み
foreground_images = ['s1_transparent.png', 'creeper.png', 'sunglass_normal.png']

# フレームの幅と高さを取得
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# 顔検出器の初期化
cascade_file = "haarcascades/haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_file)

# 合成のインターバル（秒）
interval = 3
last_composition_time = time.time()

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
        # 経過時間がインターバルを超えた場合にのみランダムなPNG画像を選択
        if time.time() - last_composition_time >= interval:
            foreground_img_file = random.choice(foreground_images)
            last_composition_time = time.time()

        foreground_img = cv2.imread(foreground_img_file, cv2.IMREAD_UNCHANGED)

        # 顔領域に合わせて透過PNG画像をリサイズ
        resized_foreground_img = cv2.resize(foreground_img, (w, h))

        # 合成処理
        alpha = resized_foreground_img[:, :, 3] / 255.0
        for c in range(3):
            frame[y:y+h, x:x+w, c] = frame[y:y+h, x:x+w, c] * (1 - alpha) + resized_foreground_img[:, :, c] * alpha

    # フレームを表示
    cv2.imshow('Frame', frame)

    # 'q'キーを押すとループを抜ける
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# キャプチャの解放とウィンドウの破棄
cap.release()
cv2.destroyAllWindows()
