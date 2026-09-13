"""
orbbec_tray_cuboid.py

Orbbec Astra2 (深度カメラ + カラーカメラ) を使って、
フレーム内に写っている「トレー」1つ1つに対して、
それをちょうど囲む3D直方体 (Oriented Bounding Box, OBB) を生成するスクリプト。

------------------------------------------------------------------------------
全体の仕組み (詳しい解説は末尾のコメント、またはチャット本文を参照)
------------------------------------------------------------------------------
1. Orbbec Astra2 から カラー画像 + 深度画像 を取得し、depth-to-color で位置合わせする
2. カラー画像上でトレー領域を検出する
   - RT-DETR (compose_dataset.py で生成したデータセットを RT-DETR-0.py で学習したモデル) で
     トレーの矩形(bounding box)を検出・分離する
   - 各bbox内だけで rembg による前景抽出を行い、矩形に混ざる背景ピクセルを除いた
     精緻なマスクを得る (深度点群の質を保つため)
   ※ 以前は rembgの前景抽出+連結成分でトレーごとに分離していたが、
     トレー同士が接触していると分離に失敗する問題があった。
     RT-DETRのbboxで先に個体を分離することでこれを解消する。
3. 各トレー領域について、対応する深度値を使って画素を3D点群に逆投影する
4. 深度センサのノイズ(外れ値)を統計的に除去する
5. 点群に対してPCA(主成分分析)を行い、トレーの向きに沿った座標系を作る
6. その座標系で点群のmin/maxを取ることで「ちょうど囲む直方体(OBB)」を求める
7. 直方体の8頂点をカラー画像に再投影して可視化する

依存ライブラリ:
  pip3 install pyorbbecsdk opencv-python numpy rembg onnxruntime ultralytics --break-system-packages

  ※ pyorbbecsdk は Orbbec 公式SDK (OrbbecSDK / pyorbbecsdk) を別途ビルド/インストールする
     必要があります (Astra2 は Orbbec SDK v2 / Femto系と同系列のAPIで動作します)。
     https://github.com/orbbec/pyorbbecsdk

  ※ RT-DETRの学習済み重み (RT-DETR-0.py の実行結果) が必要です。
     デフォルトの参照先: rtdetr_project/tray_detection/weights/best.pt

使い方:
  python3 orbbec_tray_cuboid.py \
      --rtdetr_weights rtdetr_project/tray_detection/weights/best.pt \
      --conf 0.5 \
      --model_name u2net \
      --out_dir ./cuboid_preview
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np

try:
    from rembg import remove, new_session
except ImportError:
    remove = None
    new_session = None

try:
    from pyorbbecsdk import Pipeline, Config, OBSensorType, OBFormat, OBAlignMode
except ImportError:
    Pipeline = None  # SDK未インストールでもファイル自体は読める(--dry_runで確認可能)

try:
    from ultralytics import RTDETR
except ImportError:
    RTDETR = None  # ultralytics未インストールでもファイル自体は読める(--dry_runで確認可能)


# ----------------------------------------------------------------------------
# 1. Orbbec Astra2 からの取得
# ----------------------------------------------------------------------------
class AstraCamera:
    """Orbbec Astra2 のカラー/深度ストリームを開き、位置合わせ済みフレームと内部パラメータを提供する"""

    def __init__(self, width=800, height=600, fps=30):
        if Pipeline is None:
            raise RuntimeError(
                "pyorbbecsdk がインストールされていません。"
                "https://github.com/orbbec/pyorbbecsdk の手順に従ってセットアップしてください。"
            )
        self.pipeline = Pipeline()
        config = Config()

        color_profiles = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = self._pick_profile(color_profiles, width, height, OBFormat.RGB, fps, "カラー")
        config.enable_stream(color_profile)

        depth_profiles = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        depth_profile = self._pick_profile(depth_profiles, width, height, OBFormat.Y16, fps, "深度")
        config.enable_stream(depth_profile)

        # 深度をカラー座標系に合わせる (画素ごとに深度とRGBが1対1で対応するようにする)
        config.set_align_mode(OBAlignMode.SW_MODE)
        self.pipeline.enable_frame_sync()
        self.pipeline.start(config)

        # 内部パラメータ(カラーカメラの焦点距離・光学中心)。位置合わせ後はカラー側の内部パラメータを使う。
        cam_param = self.pipeline.get_camera_param()
        self.fx = cam_param.rgb_intrinsic.fx
        self.fy = cam_param.rgb_intrinsic.fy
        self.cx = cam_param.rgb_intrinsic.cx
        self.cy = cam_param.rgb_intrinsic.cy
        self.depth_scale = 1.0  # Astra2は基本mm単位。SDKにより異なる場合はここを調整。

    @staticmethod
    def _pick_profile(profile_list, width, height, fmt, fps, label):
        """
        指定の解像度/フォーマット/fpsに一致するプロファイルを探す。
        見つからない場合は、実際にサポートされている全プロファイルを表示した上で、
        デフォルトプロファイル(センサーが用意している0番目)にフォールバックする。
        """
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            print(f"[{label}] {width}x{height} fmt={fmt} fps={fps} は非対応でした。"
                  f"利用可能なプロファイル一覧:")
            count = profile_list.get_count()
            for i in range(count):
                p = profile_list.get_stream_profile_by_index(i)
                try:
                    print(f"  - {p.get_width()}x{p.get_height()} "
                          f"fmt={p.get_format()} fps={p.get_fps()}")
                except Exception:
                    print(f"  - (詳細取得不可) profile[{i}]")
            print(f"[{label}] デフォルトプロファイルにフォールバックします。")
            return profile_list.get_default_video_stream_profile()

    def get_frames(self, timeout_ms=3000):
        frames = self.pipeline.wait_for_frames(timeout_ms)
        if frames is None:
            return None, None
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        if color_frame is None or depth_frame is None:
            return None, None

        color = np.frombuffer(color_frame.get_data(), dtype=np.uint8).reshape(
            color_frame.get_height(), color_frame.get_width(), 3
        )
        color_bgr = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        depth = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
            depth_frame.get_height(), depth_frame.get_width()
        )
        depth_mm = depth.astype(np.float32) * self.depth_scale
        return color_bgr, depth_mm

    def stop(self):
        self.pipeline.stop()


# ----------------------------------------------------------------------------
# 2. カラー画像上でのトレー領域検出 (RT-DETRでbbox検出 → 各bbox内をrembgでマスク精緻化)
# ----------------------------------------------------------------------------
_REMBG_SESSION = None
_RTDETR_MODEL = None


def get_rembg_session(model_name="u2net"):
    global _REMBG_SESSION
    if new_session is None:
        raise RuntimeError("rembgがインストールされていません。`pip3 install rembg` を実行してください。")
    if _REMBG_SESSION is None:
        _REMBG_SESSION = new_session(model_name)
    return _REMBG_SESSION


def load_rtdetr_model(weights_path="rtdetr_project/tray_detection/weights/best.pt"):
    """
    compose_dataset.py で生成したデータセットを RT-DETR-0.py で学習した
    重み(best.pt)を読み込む。
    """
    global _RTDETR_MODEL
    if RTDETR is None:
        raise RuntimeError(
            "ultralyticsがインストールされていません。`pip3 install ultralytics` を実行してください。"
        )
    if not Path(weights_path).exists():
        raise FileNotFoundError(
            f"RT-DETRの重みファイルが見つかりません: {weights_path}\n"
            "先に RT-DETR-0.py で学習を行うか、--rtdetr_weights で正しいパスを指定してください。"
        )
    if _RTDETR_MODEL is None:
        _RTDETR_MODEL = RTDETR(weights_path)
    return _RTDETR_MODEL


def detect_tray_boxes(color_bgr, model, conf=0.5, device="mps"):
    """
    RT-DETRでカラー画像上のトレーを検出し、bbox(x0, y0, x1, y1)のリストを返す。
    (画像座標系, int, 画像範囲内にクリップ済み)
    """
    h, w = color_bgr.shape[:2]
    try:
        results = model.predict(source=color_bgr, conf=conf, device=device, verbose=False)
    except Exception:
        # MPS非対応環境などでは device指定なし(CPU)にフォールバック
        results = model.predict(source=color_bgr, conf=conf, verbose=False)

    boxes = []
    if not results:
        return boxes
    for box in results[0].boxes:
        x0, y0, x1, y1 = box.xyxy[0].tolist()
        x0 = max(0, int(round(x0)))
        y0 = max(0, int(round(y0)))
        x1 = min(w, int(round(x1)))
        y1 = min(h, int(round(y1)))
        if x1 > x0 and y1 > y0:
            boxes.append((x0, y0, x1, y1))
    return boxes


def refine_mask_in_box(color_bgr, box, model_name="u2net", margin_ratio=0.05):
    """
    RT-DETRのbbox 1つ分について、bboxを少し広げた範囲だけを切り出し、
    その範囲内でrembgによる前景抽出を行う(bboxの矩形そのままだと背景の
    深度ピクセルが混ざるため、点群の質を保つのに使う)。

    Returns: 画像全体サイズのバイナリマスク(0/255)。box以外は全て0。
    """
    h, w = color_bgr.shape[:2]
    x0, y0, x1, y1 = box
    bw, bh = x1 - x0, y1 - y0
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    cx0 = max(0, x0 - mx)
    cy0 = max(0, y0 - my)
    cx1 = min(w, x1 + mx)
    cy1 = min(h, y1 + my)

    crop = color_bgr[cy0:cy1, cx0:cx1]
    session = get_rembg_session(model_name)
    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    result_rgba = remove(rgb_crop, session=session)
    alpha = result_rgba[:, :, 3]

    _, mask_bin = cv2.threshold(alpha, 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask_bin = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, kernel, iterations=1)

    # crop内で最大の連結成分のみ採用(rembgがcrop内の余白などを誤検出した場合の保険)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if n_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask_bin = (labels == largest_label).astype(np.uint8) * 255

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[cy0:cy1, cx0:cx1] = mask_bin
    return full_mask


def detect_and_segment_trays(color_bgr, detector, model_name="u2net", conf=0.5,
                              margin_ratio=0.05, device="mps"):
    """
    RT-DETRでトレーごとのbboxを検出し、各bbox内をrembgで精緻化した
    トレー1つ1つのバイナリマスクのリストを返す(旧segment_traysの置き換え)。
    """
    boxes = detect_tray_boxes(color_bgr, detector, conf=conf, device=device)
    tray_masks = [
        refine_mask_in_box(color_bgr, box, model_name=model_name, margin_ratio=margin_ratio)
        for box in boxes
    ]
    return tray_masks


# ----------------------------------------------------------------------------
# 3-4. 深度からの3D点群化 + 外れ値除去
# ----------------------------------------------------------------------------
def backproject_to_points(mask, depth_mm, fx, fy, cx, cy, depth_min=100, depth_max=3000):
    """
    マスク内の各画素を、ピンホールカメラモデルで3D点(カメラ座標系, mm単位)に逆投影する。
      X = (u - cx) * Z / fx
      Y = (v - cy) * Z / fy
      Z = depth
    """
    ys, xs = np.where(mask > 0)
    z = depth_mm[ys, xs]

    valid = (z > depth_min) & (z < depth_max)
    xs, ys, z = xs[valid], ys[valid], z[valid]

    x = (xs - cx) * z / fx
    y = (ys - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    return points


def remove_outliers(points, std_ratio=2.0):
    """各軸で平均±std_ratio*標準偏差の範囲外を外れ値として除去する(簡易版統計的外れ値除去)"""
    if len(points) < 10:
        return points
    mean = points.mean(axis=0)
    std = points.std(axis=0)
    keep = np.all(np.abs(points - mean) < std_ratio * std, axis=1)
    return points[keep]


# ----------------------------------------------------------------------------
# 5-6. PCAによる向き推定 + 直方体(OBB)計算
# ----------------------------------------------------------------------------
def compute_oriented_bbox(points):
    """
    点群からPCAで主軸を求め、その座標系でmin/maxを取ることで
    「ちょうど囲む直方体」(Oriented Bounding Box) を計算する。

    Returns:
        center: 直方体の中心 (カメラ座標系, mm)
        R: 3x3回転行列 (直方体のローカル軸 -> カメラ座標系)
        extent: 直方体の各辺の長さ (ローカルx, y, z方向)
        corners: (8, 3) 8頂点の座標 (カメラ座標系)
    """
    mean = points.mean(axis=0)
    centered = points - mean

    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # 固有値の昇順で返る
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]  # 列ベクトルが主軸(分散が大きい順)

    # 回転行列の行列式が-1(鏡映)にならないよう補正
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1

    local = centered @ R  # 点群を主軸座標系に変換
    min_local = local.min(axis=0)
    max_local = local.max(axis=0)
    extent = max_local - min_local
    center_local = (min_local + max_local) / 2
    center = mean + R @ center_local

    # 8頂点をローカル座標系で生成してからカメラ座標系に戻す
    half = extent / 2
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    corners_local = center_local + signs * half
    corners = corners_local @ R.T + mean

    return center, R, extent, corners


# ----------------------------------------------------------------------------
# 7. 直方体をカラー画像へ再投影して可視化
# ----------------------------------------------------------------------------
_EDGES = [
    (0, 1), (0, 2), (0, 4), (1, 3), (1, 5),
    (2, 3), (2, 6), (3, 7), (4, 5), (4, 6),
    (5, 7), (6, 7),
]


def project_points(points_3d, fx, fy, cx, cy):
    x, y, z = points_3d[:, 0], points_3d[:, 1], points_3d[:, 2]
    z_safe = np.where(z == 0, 1e-6, z)
    u = (x * fx / z_safe) + cx
    v = (y * fy / z_safe) + cy
    return np.stack([u, v], axis=1)


def draw_cuboid(image, corners_3d, fx, fy, cx, cy, color=(0, 255, 0)):
    pts_2d = project_points(corners_3d, fx, fy, cx, cy).astype(int)
    for i, j in _EDGES:
        cv2.line(image, tuple(pts_2d[i]), tuple(pts_2d[j]), color, 2)
    return image


# ----------------------------------------------------------------------------
# メイン処理
# ----------------------------------------------------------------------------
def process_frame(color_bgr, depth_mm, fx, fy, cx, cy, detector,
                   model_name="u2net", conf=0.5, margin_ratio=0.05, device="mps"):
    tray_masks = detect_and_segment_trays(
        color_bgr, detector, model_name=model_name, conf=conf,
        margin_ratio=margin_ratio, device=device,
    )
    vis = color_bgr.copy()
    results = []

    for idx, mask in enumerate(tray_masks):
        points = backproject_to_points(mask, depth_mm, fx, fy, cx, cy)
        points = remove_outliers(points)
        if len(points) < 20:
            print(f"トレー{idx}: 有効な深度点が少なすぎるためスキップ")
            continue

        center, R, extent, corners = compute_oriented_bbox(points)
        results.append({
            "id": idx,
            "center_mm": center,
            "extent_mm": extent,
            "rotation": R,
            "corners_mm": corners,  # (8, 3) 直方体の8頂点座標(カメラ座標系, mm)
        })

        draw_cuboid(vis, corners, fx, fy, cx, cy)
        u, v = project_points(center[None, :], fx, fy, cx, cy)[0].astype(int)
        label = f"tray{idx} {extent[0]:.0f}x{extent[1]:.0f}x{extent[2]:.0f}mm"
        cv2.putText(vis, label, (u - 40, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        print(f"トレー{idx}: 中心(mm)={center.round(1)}, サイズ(mm, L/W/H)={extent.round(1)}")
        print(f"  8頂点座標(mm, カメラ座標系):")
        for corner_idx, corner in enumerate(corners):
            print(f"    点{corner_idx}: {corner.round(1)}")

    return vis, results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="./cuboid_preview", help="可視化結果の保存先")
    parser.add_argument("--model_name", default="u2net", help="rembgのモデル名(bbox内マスク精緻化用)")
    parser.add_argument(
        "--rtdetr_weights", default="rtdetr_project/tray_detection/weights/best.pt",
        help="RT-DETR-0.py で学習した重みファイルのパス",
    )
    parser.add_argument("--conf", type=float, default=0.5, help="RT-DETRの検出確信度しきい値")
    parser.add_argument(
        "--margin_ratio", type=float, default=0.05,
        help="bboxを広げてrembgに渡す際の余白比率(トレーの縁が切れないようにする)",
    )
    parser.add_argument("--device", default="mps", help="RT-DETR推論デバイス (mps/cpu/cuda)")
    parser.add_argument("--num_frames", type=int, default=1, help="取得して処理するフレーム数")
    parser.add_argument("--width", type=int, default=800, help="カラー/深度の要求解像度(幅)")
    parser.add_argument("--height", type=int, default=600, help="カラー/深度の要求解像度(高さ)")
    parser.add_argument("--fps", type=int, default=30, help="要求フレームレート")
    parser.add_argument(
        "--dry_run", action="store_true",
        help="カメラなしで動作確認だけしたい場合(実カメラ接続時は付けない)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print("dry_run: カメラ接続なしのため処理をスキップします。ロジック確認のみ完了。")
        return

    detector = load_rtdetr_model(args.rtdetr_weights)

    cam = AstraCamera(width=args.width, height=args.height, fps=args.fps)
    try:
        for i in range(args.num_frames):
            color_bgr, depth_mm = cam.get_frames()
            if color_bgr is None:
                print("フレーム取得失敗。リトライします。")
                time.sleep(0.1)
                retry_color, retry_depth = None, None
                for _ in range(20):
                    retry_color, retry_depth = cam.get_frames()
                    if retry_color is not None:
                        break
                    time.sleep(0.1)
                color_bgr, depth_mm = retry_color, retry_depth
                if color_bgr is None:
                    print("複数回リトライしても取得できませんでした。スキップします。")
                    continue

            vis, results = process_frame(
                color_bgr, depth_mm, cam.fx, cam.fy, cam.cx, cam.cy, detector,
                model_name=args.model_name, conf=args.conf,
                margin_ratio=args.margin_ratio, device=args.device,
            )
            out_path = out_dir / f"cuboid_{i:03d}.jpg"
            cv2.imwrite(str(out_path), vis)
            print(f"保存: {out_path} (トレー検出数: {len(results)})")
    finally:
        cam.stop()


if __name__ == "__main__":
    main()




'''
コマンド
↓


sudo python3 orbbec_tray_cuboid.py --rtdetr_weights runs/detect/rtdetr_project/tray_detection/weights/best.pt --conf 0.5 --model_name u2net --device mps --out_dir ./cuboid_preview                                                         


'''