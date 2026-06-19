"""下载 Globe-LFMC 2.0(figshare, Springer Nature;CC-BY-4.0)。

数据:`Globe-LFMC-2.0 final.xlsx`(~72.5 MB,单文件 3 sheet)
来源:Yebra et al. 2024, Sci Data 11:332;figshare item 25413790
DOI:10.1038/s41597-024-03159-6

用法:
    python src/download_data.py                 # 下到 data/raw/
    python src/download_data.py --out data/raw

若网络中断,可用命令行断点续传:
    curl -L -C - "https://ndownloader.figshare.com/files/45049786" \\
        -o "data/raw/Globe-LFMC-2.0 final.xlsx"
"""
from __future__ import annotations
import argparse, os, sys, urllib.request

sys.path.insert(0, os.path.dirname(__file__))
from common import FIGSHARE_DOWNLOAD_URL, XLSX_NAME, RAW_DIR

EXPECTED_BYTES = 72_494_093   # figshare API 报的大小,用于校验完整性


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    total = total_size if total_size > 0 else EXPECTED_BYTES
    pct = min(100.0, done * 100.0 / total)
    sys.stdout.write(f"\r  下载中 {done/1e6:6.1f} / {total/1e6:6.1f} MB  ({pct:5.1f}%)")
    sys.stdout.flush()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=RAW_DIR, help="原始数据落地目录")
    ap.add_argument("--force", action="store_true", help="已存在也重新下")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    xlsx_path = os.path.join(args.out, XLSX_NAME)

    if os.path.exists(xlsx_path) and not args.force:
        size = os.path.getsize(xlsx_path)
        if abs(size - EXPECTED_BYTES) < 1_000_000:
            print(f"已存在且大小正常({size/1e6:.1f} MB):{xlsx_path}")
            print("如需重下加 --force。")
            return
        print(f"已存在但大小异常({size/1e6:.1f} MB,期望 ~72.5 MB),重新下载…")

    print(f"下载 {FIGSHARE_DOWNLOAD_URL}\n  → {xlsx_path}")
    urllib.request.urlretrieve(FIGSHARE_DOWNLOAD_URL, xlsx_path, _progress)
    print()

    size = os.path.getsize(xlsx_path)
    if abs(size - EXPECTED_BYTES) > 1_000_000:
        print(f"⚠️  下载大小 {size/1e6:.1f} MB 与期望 ~72.5 MB 不符,可能不完整,"
              f"建议用 curl -C - 断点续传重下。", file=sys.stderr)
        sys.exit(1)
    print(f"完成({size/1e6:.1f} MB)。下一步:python src/prepare_lfmc.py")


if __name__ == "__main__":
    main()
