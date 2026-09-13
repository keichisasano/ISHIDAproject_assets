"""
infer_and_visualize.py

学習済みRT-DETRモデルで新しい画像(1枚 or フォルダ内の複数枚)に推論を行い、
検出結果(バウンディングボックス + confidence)を描画して保存する。

使い方:
  # 1枚の画像
  python3 infer_and_visualize.py --weights rtdetr_project/tray_detection/weights/best.pt \
      --source ./test_images/sample.jpg --out_dir ./infer_results

  # フォルダ内の全画像
  python3 infer_and_visualize.py --weights rtdetr_project/tray_detection/weights/best.pt \
      --source ./test_images --out_dir ./infer_results

  # 信頼度の閾値を変える(デフォルト0.25)
  python3 infer_and_visualize.py --weights ...best.pt --source ./test_images --conf 0.5
"""

import argparse
from pathlib import Path

from ultralytics import RTDETR


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="学習済み重みファイル (例: best.pt)")
    parser.add_argument("--source", required=True, help="推論対象の画像ファイル or フォルダ")
    parser.add_argument("--out_dir", default="./infer_results", help="結果の出力先フォルダ")
    parser.add_argument("--conf", type=float, default=0.25, help="検出の信頼度閾値")
    parser.add_argument("--imgsz", type=int, default=640, help="推論時の画像サイズ")
    parser.add_argument("--device", default="mps", help="使用デバイス (mps / cpu / 0 など)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"モデル読み込み中: {args.weights}")
    model = RTDETR(args.weights)

    print(f"推論実行中: {args.source} (conf={args.conf})")
    results = model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,          # 検出結果を描画した画像を保存
        project=str(out_dir.resolve().parent),
        name=out_dir.resolve().name,
        exist_ok=True,
    )

    # 各画像ごとの検出内容をコンソールにも表示
    total_detections = 0
    for r in results:
        n = len(r.boxes)
        total_detections += n
        img_name = Path(r.path).name
        if n == 0:
            print(f"  {img_name}: 検出なし")
        else:
            confs = [f"{c:.2f}" for c in r.boxes.conf.tolist()]
            print(f"  {img_name}: {n}件検出 (confidence: {', '.join(confs)})")

    actual_save_dir = results[0].save_dir if results else out_dir
    print(f"\n完了: 合計 {total_detections} 件検出")
    print(f"可視化済み画像の保存先: {actual_save_dir}")


if __name__ == "__main__":
    main()



"""

python3 infer_and_visualize.py \
    --weights ./runs/detect/rtdetr_project/tray_detection/weights/best.pt \
    --source ./sample_tray/ \
    --out_dir ./infer_results \
    --conf 0.25
    
"""

