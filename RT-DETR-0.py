
from ultralytics import RTDETR

# 1. 事前学習済みモデルの読み込み
# サイズのバリエーション: rtdetr-l.pt (Large), rtdetr-x.pt (Extra Large)
# 初回実行時に自動的にダウンロードされます。
model = RTDETR("rtdetr-l.pt")

# 2. ファインチューニングの実行
results = model.train(
    data="dataset.yaml",  # 先ほど作成した設定ファイル
    epochs=3,            # 学習の反復回数
    imgsz=640,            # 画像サイズ（640x640にリサイズして学習）
    batch=8,              # バッチサイズ（※Macのメモリが足りない場合は4や2に下げてください）
    device="mps",         # Apple Silicon GPU (Metal Performance Shaders) を使用
    project="rtdetr_project", # 実行結果が保存される親フォルダ名
    name="tray_detection",    # 今回の学習結果が保存されるフォルダ名
    exist_ok=True         # 同じ名前のフォルダがある場合上書きを許可する
)
