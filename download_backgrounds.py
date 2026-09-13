"""
download_backgrounds.py

CSV(1列目のURL: *_z.jpg = 中サイズ画像)を読み込み、
backgrounds フォルダの中身をすべて置き換える形でダウンロードするスクリプト。

使い方:
    python3 download_backgrounds.py --csv all_image_urls.csv --out_dir ./backgrounds

必要ライブラリ:
    pip3 install requests
"""



"""

https://github.com/pedropro/TACO/blob/master/data/all_image_urls.csv

"""

import argparse
import csv
import shutil
from pathlib import Path
from urllib.parse import urlparse

import requests


def get_filename_from_url(url: str) -> str:
    """URLからファイル名部分を取り出す"""
    return Path(urlparse(url).path).name


def download_backgrounds(csv_path: Path, out_dir: Path, column: int, clear_existing: bool, timeout: int):
    if clear_existing and out_dir.exists():
        print(f"既存の {out_dir} を削除します...")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSVからURLを収集(ヘッダー無し前提)
    urls = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            if len(row) <= column:
                continue
            url = row[column].strip()
            if url:
                urls.append(url)

    print(f"{len(urls)} 件のURLを見つけました。ダウンロードを開始します...")

    success = 0
    failed = []

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})

    for i, url in enumerate(urls, start=1):
        fname = get_filename_from_url(url)
        if not fname:
            fname = f"bg_{i:05d}.jpg"
        out_path = out_dir / fname

        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                f.write(resp.content)
            success += 1
            if i % 50 == 0 or i == len(urls):
                print(f"[{i}/{len(urls)}] 完了 ({success} 成功, {len(failed)} 失敗)")
        except Exception as e:
            failed.append((url, str(e)))
            print(f"失敗: {url} -> {e}")

    print(f"\n完了: {success}/{len(urls)} 件ダウンロード成功 -> {out_dir}")
    if failed:
        fail_log = out_dir.parent / "download_failed.txt"
        with open(fail_log, "w", encoding="utf-8") as f:
            for url, err in failed:
                f.write(f"{url}\t{err}\n")
        print(f"失敗したURLは {fail_log} に記録しました ({len(failed)} 件)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", required=True, help="URLが入ったCSVファイルのパス")
    parser.add_argument("--out_dir", default="./backgrounds", help="背景画像の保存先フォルダ")
    parser.add_argument("--column", type=int, default=0, help="使用するCSVの列番号(0始まり)。デフォルト0 = 1列目(_z.jpg)")
    parser.add_argument("--no_clear", action="store_true", help="指定するとフォルダを削除せず追記モードでダウンロードする")
    parser.add_argument("--timeout", type=int, default=15, help="1件あたりのダウンロードタイムアウト秒数")
    args = parser.parse_args()

    download_backgrounds(
        csv_path=Path(args.csv),
        out_dir=Path(args.out_dir),
        column=args.column,
        clear_existing=not args.no_clear,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
