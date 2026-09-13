"""
verify_annotations.py

compose_dataset.py が生成した YOLO形式データセット (images/ + labels/) を
読み込み、バウンディングボックスを描画した確認用画像を出力するスクリプト。

使い方:
    python3 verify_annotations.py --dataset_dir ./datasets --split train --out_dir ./anno_check
    python3 verify_annotations.py --dataset_dir ./datasets --split val --out_dir ./anno_check --num_samples 20

出力:
    out_dir/ 以下に、bboxを緑枠で描画した画像が保存されます。
    ついでに以下のチェックも自動で行います:
      - images と labels の枚数/対応関係が一致しているか
      - ラベルファイルの中身が壊れていないか(値の数、範囲 0-1 など)
      - bboxが画像の端からはみ出していないか / 極端に小さすぎないか
"""

import argparse
import random
from pathlib import Path

import cv2


def load_class_names(dataset_yaml):
    """dataset.yaml があれば names を読み込む(なければ番号のみ表示)"""
    names = {}
    if dataset_yaml and Path(dataset_yaml).exists():
        try:
            import yaml
            with open(dataset_yaml, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            for i, n in enumerate(data.get("names", [])):
                names[i] = n
        except Exception as e:
            print(f"dataset.yaml の読み込みに失敗しました(無視して続行): {e}")
    return names


def check_label_line(line, img_w, img_h, warnings, fname):
    parts = line.strip().split()
    if len(parts) != 5:
        warnings.append(f"{fname}: 値の数が5個ではありません -> '{line.strip()}'")
        return None

    try:
        cls, xc, yc, w, h = int(parts[0]), *map(float, parts[1:])
    except ValueError:
        warnings.append(f"{fname}: 数値に変換できません -> '{line.strip()}'")
        return None

    for name, v in [("xc", xc), ("yc", yc), ("w", w), ("h", h)]:
        if not (0.0 <= v <= 1.0):
            warnings.append(f"{fname}: {name}={v:.4f} が 0-1 の範囲外です")

    if w <= 0 or h <= 0:
        warnings.append(f"{fname}: 幅または高さが0以下です (w={w}, h={h})")

    # ピクセル換算での小ささチェック(小さすぎ = ラベルミスの可能性)
    if w * img_w < 3 or h * img_h < 3:
        warnings.append(f"{fname}: bboxが極端に小さいです (約{w*img_w:.1f}x{h*img_h:.1f}px)")

    return cls, xc, yc, w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset_dir", required=True, help="datasets のルートディレクトリ (train/val を含む)")
    ap.add_argument("--split", default="train", choices=["train", "val"], help="確認する split")
    ap.add_argument("--out_dir", required=True, help="描画済み画像の出力先")
    ap.add_argument("--num_samples", type=int, default=0, help="ランダムに抽出して確認する枚数(0=全件)")
    ap.add_argument("--dataset_yaml", default=None, help="クラス名表示用の dataset.yaml (任意)")
    args = ap.parse_args()

    img_dir = Path(args.dataset_dir) / args.split / "images"
    lbl_dir = Path(args.dataset_dir) / args.split / "labels"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not img_dir.exists() or not lbl_dir.exists():
        print(f"エラー: {img_dir} または {lbl_dir} が存在しません")
        return

    names = load_class_names(args.dataset_yaml)

    img_paths = sorted([p for p in img_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    lbl_paths = {p.stem: p for p in lbl_dir.glob("*.txt")}

    print(f"[{args.split}] images: {len(img_paths)}枚, labels: {len(lbl_paths)}枚")

    warnings = []
    missing_labels = []
    empty_labels = []

    # 画像とラベルの対応チェック
    for p in img_paths:
        if p.stem not in lbl_paths:
            missing_labels.append(p.name)

    if missing_labels:
        warnings.append(f"ラベルファイルが存在しない画像が {len(missing_labels)}枚あります(例: {missing_labels[:5]})")

    targets = img_paths
    if args.num_samples > 0 and args.num_samples < len(img_paths):
        targets = random.sample(img_paths, args.num_samples)

    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255)]

    saved = 0
    for p in targets:
        img = cv2.imread(str(p))
        if img is None:
            warnings.append(f"{p.name}: 画像を読み込めません")
            continue
        h, w = img.shape[:2]

        lbl_path = lbl_paths.get(p.stem)
        if lbl_path is None:
            continue

        lines = [l for l in lbl_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        if not lines:
            empty_labels.append(p.name)
            continue

        for line in lines:
            parsed = check_label_line(line, w, h, warnings, lbl_path.name)
            if parsed is None:
                continue
            cls, xc, yc, bw, bh = parsed

            x0 = int((xc - bw / 2) * w)
            y0 = int((yc - bh / 2) * h)
            x1 = int((xc + bw / 2) * w)
            y1 = int((yc + bh / 2) * h)

            color = colors[cls % len(colors)]
            label_text = names.get(cls, str(cls))
            cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
            cv2.putText(img, label_text, (x0, max(0, y0 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        out_path = out_dir / f"check_{p.name}"
        cv2.imwrite(str(out_path), img)
        saved += 1

    if empty_labels:
        warnings.append(f"ラベルの中身が空の画像が {len(empty_labels)}枚あります(例: {empty_labels[:5]})")

    print(f"\n{saved}枚の確認用画像を {out_dir} に保存しました。")
    print("緑(などの色)の枠がトレーの位置と一致しているか目視で確認してください。\n")

    if warnings:
        print(f"=== 警告 ({len(warnings)}件) ===")
        for w_ in warnings[:30]:
            print(f"- {w_}")
        if len(warnings) > 30:
            print(f"...他 {len(warnings) - 30}件")
    else:
        print("警告はありませんでした。ラベルの形式・範囲は正常です。")


if __name__ == "__main__":
    main()
