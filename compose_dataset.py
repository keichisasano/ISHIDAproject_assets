"""
compose_dataset.py

黒背景で撮影したトレー画像を「背景切り取りAI(rembg)」で切り抜き、
任意の背景画像にランダム合成して物体検出(YOLO/RT-DETR)用の
アノテーション付きデータセットを自動生成するスクリプト。

使い方:
  1. まずマスク抽出の確認 (rembgの切り抜き精度をチェック):
     python compose_dataset.py preview --tray_dir ./tray_images --out_dir ./mask_preview

  2. 問題なければ本番のデータセット生成 (背景に貼り付け + アノテーション自動生成):
     python compose_dataset.py generate \
         --tray_dir ./tray_images \
         --bg_dir ./backgrounds \
         --out_dir ./datasets \
         --num_images 500 \
         --val_ratio 0.2

出力構成 (dataset.yaml と互換):
  out_dir/
    train/images/*.jpg
    train/labels/*.txt
    val/images/*.jpg
    val/labels/*.txt
"""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

try:
    from rembg import remove, new_session
except ImportError:
    remove = None
    new_session = None


# ----------------------------------------------------------------------------
# マスク抽出 (背景切り取りAI: rembg)
# ----------------------------------------------------------------------------
_REMBG_SESSION = None


def get_rembg_session(model_name="u2net"):
    """rembgのセッションをキャッシュして使い回す(毎回モデルを読み込むと遅いため)"""
    global _REMBG_SESSION
    if new_session is None:
        raise RuntimeError(
            "rembgがインストールされていません。`pip3 install rembg` を実行してください。"
        )
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session(model_name)
    return _REMBG_SESSION


def extract_tray_mask(bgr_img, model_name="u2net", min_area_ratio=0.01):
    """
    背景切り取りAI(rembg)を使ってトレーのマスクを抽出する。
    黒背景・影・ムラがあっても、AIが被写体そのものを認識して切り抜くため
    閾値調整が不要で安定しやすい。

    - rembgでアルファチャンネル(前景=トレー)を取得
    - 軽くモルフォロジー処理でノイズを除去
    - 最大の輪郭のみを採用(トレー以外の小さなノイズを除去)

    Returns:
        mask: uint8, 0-255 (グレースケールのアルファマスク), shape (H, W)
    """
    session = get_rembg_session(model_name)

    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    result_rgba = remove(rgb_img, session=session)  # (H, W, 4) RGBA, uint8
    mask = result_rgba[:, :, 3]  # アルファチャンネル = 前景マスク

    # 軽いノイズ除去(AIの誤検出の小さい点などを除去)
    _, mask_bin = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)

    h, w = mask.shape
    min_area = h * w * min_area_ratio
    contours = [c for c in contours if cv2.contourArea(c) >= min_area]
    if not contours:
        return np.zeros_like(mask)

    largest = max(contours, key=cv2.contourArea)
    contour_mask = np.zeros_like(mask_bin)
    cv2.drawContours(contour_mask, [largest], -1, 255, thickness=cv2.FILLED)

    # 元のアルファ(グラデーション)を、最大輪郭の範囲内だけに限定して滑らかな縁を保つ
    clean_mask = np.where(contour_mask > 0, mask, 0).astype(np.uint8)

    return clean_mask


def feather_mask(mask, blur_size=9):
    """マスクの縁をぼかしてαブレンド用のfloatマスク(0.0〜1.0)にする"""
    mask_blur = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    return mask_blur.astype(np.float32) / 255.0


def crop_to_content(bgr_img, mask):
    """マスクのバウンディングボックスで画像とマスクを切り抜く(余白除去)"""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return bgr_img, mask
    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    return bgr_img[y0:y1 + 1, x0:x1 + 1], mask[y0:y1 + 1, x0:x1 + 1]


# ----------------------------------------------------------------------------
# 合成
# ----------------------------------------------------------------------------
def rotate_and_scale(img, mask, angle, scale):
    h, w = img.shape[:2]
    new_w, new_h = int(w * scale), int(h * scale)
    new_w, new_h = max(1, new_w), max(1, new_h)
    img_r = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask_r = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 回転時にキャンバスを広げてクロップされないようにする
    diag = int(np.sqrt(new_w ** 2 + new_h ** 2)) + 4
    canvas_img = np.zeros((diag, diag, 3), dtype=np.uint8)
    canvas_mask = np.zeros((diag, diag), dtype=np.uint8)
    y_off = (diag - new_h) // 2
    x_off = (diag - new_w) // 2
    canvas_img[y_off:y_off + new_h, x_off:x_off + new_w] = img_r
    canvas_mask[y_off:y_off + new_h, x_off:x_off + new_w] = mask_r

    M = cv2.getRotationMatrix2D((diag / 2, diag / 2), angle, 1.0)
    rot_img = cv2.warpAffine(canvas_img, M, (diag, diag))
    rot_mask = cv2.warpAffine(canvas_mask, M, (diag, diag))

    return crop_to_content(rot_img, rot_mask)


def adjust_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def composite(bg_img, tray_img, tray_mask, x, y):
    """
    bg_img に tray_img を alpha ブレンドで貼り付ける。
    (x, y) は貼り付け先の左上座標。
    範囲外にはみ出す場合はクリップする。
    Returns: 合成後画像, (x0, y0, x1, y1) 貼り付けられた範囲(bg座標系)
    """
    bh, bw = bg_img.shape[:2]
    th, tw = tray_img.shape[:2]

    x0, y0 = x, y
    x1, y1 = x + tw, y + th

    # bg範囲内にクリップ
    src_x0, src_y0 = 0, 0
    src_x1, src_y1 = tw, th

    if x0 < 0:
        src_x0 = -x0
        x0 = 0
    if y0 < 0:
        src_y0 = -y0
        y0 = 0
    if x1 > bw:
        src_x1 -= (x1 - bw)
        x1 = bw
    if y1 > bh:
        src_y1 -= (y1 - bh)
        y1 = bh

    if x0 >= x1 or y0 >= y1:
        return bg_img, None

    tray_crop = tray_img[src_y0:src_y1, src_x0:src_x1]
    mask_crop = tray_mask[src_y0:src_y1, src_x0:src_x1]
    alpha = mask_crop[:, :, None]

    roi = bg_img[y0:y1, x0:x1].astype(np.float32)
    blended = roi * (1 - alpha) + tray_crop.astype(np.float32) * alpha
    bg_img = bg_img.copy()
    bg_img[y0:y1, x0:x1] = blended.astype(np.uint8)

    return bg_img, (x0, y0, x1, y1)


def _iou(box_a, box_b):
    """2つのbbox (x0, y0, x1, y1) のIoUを計算する(重なり判定用)"""
    ax0, ay0, ax1, ay1 = box_a
    bx0, by0, bx1, by1 = box_b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / min(area_a, area_b)


# ----------------------------------------------------------------------------
# メイン処理: preview
# ----------------------------------------------------------------------------
def cmd_preview(args):
    tray_dir = Path(args.tray_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tray_paths = sorted([p for p in tray_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    if not tray_paths:
        print(f"エラー: {tray_dir} に画像が見つかりません")
        return

    for p in tray_paths:
        img = cv2.imread(str(p))
        if img is None:
            print(f"読み込み失敗: {p}")
            continue
        print(f"処理中: {p.name} (rembgで切り抜き中...)")
        mask = extract_tray_mask(img, model_name=args.model_name)

        # 可視化: 元画像 / マスク / マスクを緑でオーバーレイ したものを横に並べる
        overlay = img.copy()
        overlay[mask > 0] = (0.5 * overlay[mask > 0] + 0.5 * np.array([0, 255, 0])).astype(np.uint8)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        h = img.shape[0]
        combined = np.hstack([img, mask_bgr, overlay])
        out_path = out_dir / f"preview_{p.stem}.jpg"
        cv2.imwrite(str(out_path), combined)
        print(f"保存: {out_path}")

    print("\nプレビュー画像を確認してください(左:元画像 中:マスク 右:オーバーレイ)。")
    print("トレーの形とマスクがずれている場合は --model_name を変えて試してください")
    print("(例: u2net, u2netp, isnet-general-use, silueta)")


# ----------------------------------------------------------------------------
# メイン処理: generate
# ----------------------------------------------------------------------------
def cmd_generate(args):
    tray_dir = Path(args.tray_dir)
    bg_dir = Path(args.bg_dir)
    out_dir = Path(args.out_dir)

    tray_paths = sorted([p for p in tray_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])
    bg_paths = sorted([p for p in bg_dir.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")])

    if not tray_paths:
        print(f"エラー: {tray_dir} にトレー画像が見つかりません")
        return
    if not bg_paths:
        print(f"エラー: {bg_dir} に背景画像が見つかりません")
        return

    print(f"トレー画像: {len(tray_paths)}枚, 背景画像: {len(bg_paths)}枚")

    # 事前にマスクを全て抽出してキャッシュ(処理高速化)
    tray_cache = []
    for p in tray_paths:
        img = cv2.imread(str(p))
        if img is None:
            continue
        print(f"マスク抽出中(rembg): {p.name}")
        mask = extract_tray_mask(img, model_name=args.model_name)
        cropped_img, cropped_mask = crop_to_content(img, mask)
        if cropped_mask.sum() == 0:
            print(f"警告: {p} からトレーを抽出できませんでした。スキップします。")
            continue
        alpha = feather_mask(cropped_mask, blur_size=args.feather)
        tray_cache.append((cropped_img, alpha))

    if not tray_cache:
        print("有効なトレー画像がありません。--black_thresh を調整して preview で確認してください。")
        return

    # train/val 分割
    n_total = args.num_images
    n_val = int(n_total * args.val_ratio)
    n_train = n_total - n_val

    for split, n in [("train", n_train), ("val", n_val)]:
        img_out = out_dir / split / "images"
        lbl_out = out_dir / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for i in range(n):
            bg_path = random.choice(bg_paths)
            bg_img = cv2.imread(str(bg_path))
            if bg_img is None:
                continue

            # 背景の短辺を基準に、必要ならリサイズ(あまり巨大な画像を避ける)
            max_side = args.bg_max_side
            bh, bw = bg_img.shape[:2]
            if max(bh, bw) > max_side:
                scale_bg = max_side / max(bh, bw)
                bg_img = cv2.resize(bg_img, (int(bw * scale_bg), int(bh * scale_bg)))
                bh, bw = bg_img.shape[:2]

            composed = bg_img
            bboxes = []  # このimageに貼り付けた全トレーのbbox(bg座標系)を蓄積

            # 1枚の画像に貼り付けるトレー枚数をランダムに決定 (例: 1〜3枚)
            n_trays = random.randint(args.min_trays, args.max_trays)

            for _ in range(n_trays):
                tray_img, tray_alpha_full = random.choice(tray_cache)

                # ランダム回転・スケール
                angle = random.uniform(-args.max_rotation, args.max_rotation)
                # トレーが背景に対して大きすぎ/小さすぎないよう、背景サイズ基準でスケールを決める
                target_ratio = random.uniform(args.min_scale, args.max_scale)
                th0, tw0 = tray_img.shape[:2]
                base_scale = (min(bh, bw) * target_ratio) / max(th0, tw0)

                mask_uint8 = (tray_alpha_full * 255).astype(np.uint8)
                rot_img, rot_mask = rotate_and_scale(tray_img, mask_uint8, angle, base_scale)
                if rot_mask.sum() == 0:
                    continue
                rot_alpha = feather_mask(rot_mask, blur_size=args.feather)

                # 明るさジッター(背景に馴染ませる)
                brightness_factor = random.uniform(0.85, 1.15)
                rot_img = adjust_brightness(rot_img, brightness_factor)

                # 左右反転
                if random.random() < 0.5:
                    rot_img = cv2.flip(rot_img, 1)
                    rot_alpha = cv2.flip(rot_alpha, 1)

                th, tw = rot_img.shape[:2]
                if th >= bh or tw >= bw:
                    # トレーが背景より大きい場合はスキップ(スケール範囲を見直すこと)
                    continue

                # 複数トレーが極端に重ならないよう、何回か配置場所を試す
                placed = False
                for _try in range(args.max_place_tries):
                    x = random.randint(0, bw - tw)
                    y = random.randint(0, bh - th)
                    cand_box = (x, y, x + tw, y + th)

                    # 既存のbboxとの重なり(IoU)が閾値以下ならOK
                    ok = True
                    for ex in bboxes:
                        if _iou(cand_box, ex) > args.max_overlap:
                            ok = False
                            break
                    if ok:
                        placed = True
                        break

                if not placed:
                    # 重ならない場所が見つからなくても、最後に試した位置でそのまま貼る
                    pass

                composed, bbox = composite(composed, rot_img, rot_alpha, x, y)
                if bbox is None:
                    continue
                bboxes.append(bbox)

            if not bboxes:
                continue

            fname = f"{split}_{i:05d}"
            cv2.imwrite(str(img_out / f"{fname}.jpg"), composed)
            with open(lbl_out / f"{fname}.txt", "w") as f:
                for bx0, by0, bx1, by1 in bboxes:
                    # YOLO形式: class x_center y_center width height (全て0-1に正規化)
                    xc = (bx0 + bx1) / 2 / bw
                    yc = (by0 + by1) / 2 / bh
                    bw_norm = (bx1 - bx0) / bw
                    bh_norm = (by1 - by0) / bh
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw_norm:.6f} {bh_norm:.6f}\n")

        print(f"{split}: {n}枚 生成完了 -> {img_out}")

    print("\n完了しました。dataset.yaml の train/val パスをこの出力先に合わせて更新してください。")


# ----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p_preview = sub.add_parser("preview", help="マスク抽出結果を確認する")
    p_preview.add_argument("--tray_dir", required=True, help="トレー黒背景画像のフォルダ")
    p_preview.add_argument("--out_dir", required=True, help="プレビュー画像の出力先")
    p_preview.add_argument("--model_name", default="u2net", help="rembgのモデル名 (u2net, u2netp, isnet-general-use, silueta など)")
    p_preview.set_defaults(func=cmd_preview)

    p_gen = sub.add_parser("generate", help="合成データセットを生成する")
    p_gen.add_argument("--tray_dir", required=True, help="トレー黒背景画像のフォルダ")
    p_gen.add_argument("--bg_dir", required=True, help="背景画像のフォルダ")
    p_gen.add_argument("--out_dir", required=True, help="データセットの出力先")
    p_gen.add_argument("--num_images", type=int, default=500, help="生成する合成画像の総数")
    p_gen.add_argument("--val_ratio", type=float, default=0.2, help="検証用データの割合")
    p_gen.add_argument("--model_name", default="u2net", help="rembgのモデル名 (u2net, u2netp, isnet-general-use, silueta など)")
    p_gen.add_argument("--feather", type=int, default=9, help="マスク境界のぼかし幅(奇数)")
    p_gen.add_argument("--max_rotation", type=float, default=180.0, help="ランダム回転の最大角度")
    p_gen.add_argument("--min_scale", type=float, default=0.2, help="背景に対するトレーの最小サイズ比")
    p_gen.add_argument("--max_scale", type=float, default=0.75, help="背景に対するトレーの最大サイズ比(大きめのトレーも生成するため0.5から0.75に拡大)")
    p_gen.add_argument("--bg_max_side", type=int, default=1280, help="背景画像の最大辺(これより大きい場合縮小)")
    p_gen.add_argument("--min_trays", type=int, default=1, help="1枚の画像に合成するトレーの最小枚数")
    p_gen.add_argument("--max_trays", type=int, default=3, help="1枚の画像に合成するトレーの最大枚数")
    p_gen.add_argument("--max_overlap", type=float, default=0.2, help="複数トレー配置時に許容する最大重なり(IoU)")
    p_gen.add_argument("--max_place_tries", type=int, default=15, help="重ならない配置を探す試行回数")
    p_gen.set_defaults(func=cmd_generate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()



"""
画像切り抜きのみ
python3 compose_dataset.py preview --tray_dir ./tray_images --out_dir ./mask_preview
"""

"""
python3 compose_dataset.py generate \
    --tray_dir ./tray_images \
    --bg_dir ./backgrounds \
    --out_dir ./datasets \
    --num_images 500 \
    --val_ratio 0.2

python3 compose_dataset.py generate \
    --tray_dir ./tray_images \
    --bg_dir ./backgrounds \
    --out_dir ./datasets \
    --num_images 500 \
    --val_ratio 0.2 \
    --min_trays 1 --max_trays 3 \
    --min_scale 0.2 --max_scale 0.75

"""