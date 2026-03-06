#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量测试：按文件夹区分单据，每个文件夹为一个单据，其下所有文件为该单据的附件。
先对附件做分类，再对支持的类型做抽取；输出按分类分表，每张表为该类附件的编号及提取字段。
"""

import os
import csv
import json  # 仅用于 extract_fields_from_response 解析 result
import argparse
import requests
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# 支持的扩展名
SUPPORTED_EXT = ('.pdf', '.png', '.jpg', '.jpeg')

# 抽取接口支持的类型（与 server type_mapping 一致）
EXTRACT_SUPPORTED = {
    "发票", "订单合同", "商城到货单", "质保金",
    "银付凭证", "转账凭证", "支付申请单或XX证明"
}

DEFAULT_BUSINESS_TYPE = "工程报销支付申请_电网管理平台(深圳局)"


def collect_by_doc(root_dir: str) -> list:
    """
    按单据（文件夹）收集附件。
    返回 [(单据编号, 附件路径), ...]，单据编号 = 文件夹名。
    """
    root = Path(root_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"目录不存在: {root_dir}")
    items = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        doc_id = sub.name
        for ext in SUPPORTED_EXT:
            for f in sub.glob(f"*{ext}"):
                items.append((doc_id, str(f.resolve())))
    return sorted(items, key=lambda x: (x[0], x[1]))


def classify_batch(base_url: str, business_type: str, file_paths: list) -> list:
    """调用分类接口，返回与 file_paths 顺序对应的分类结果列表。"""
    url = f"{base_url.rstrip('/')}/nfdw/model/classify"
    payload = {
        "businessType": business_type,
        "list": [
            {"fileName": os.path.basename(p), "fileUrl": p, "fileId": str(i)}
            for i, p in enumerate(file_paths)
        ]
    }
    try:
        r = requests.post(url, json=payload, timeout=120)
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as e:
        return [{"error": str(e), "fileUrl": p} for p in file_paths]

    if data.get("code") != 200:
        return [{"error": data.get("message", "未知错误"), "fileUrl": p} for p in file_paths]

    results = data.get("data") or []
    out = []
    for i, p in enumerate(file_paths):
        found = None
        for item in results:
            if item.get("fileUrl") == p or item.get("fileName") == os.path.basename(p):
                found = item
                break
        if found is None and i < len(results):
            found = results[i]
        elif found is None and results:
            found = results[-1]
        out.append(found or {"error": "无分类结果", "fileUrl": p})
    return out


def extract_one(base_url: str, classify: str, file_path: str) -> dict:
    """对单个文件调用抽取接口。"""
    url = f"{base_url.rstrip('/')}/nfdw/api/v1/model/extract"
    payload = {"classify": classify, "fileUrl": file_path}
    try:
        r = requests.post(url, json=payload, timeout=180)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        return {"code": -1, "message": str(e), "data": None}


def extract_fields_from_response(ext_response: dict) -> dict:
    """
    从抽取接口返回的 data 中解析出扁平字段 dict。
    data 通常为 [ { "page", "pageUrl", "result": "json_str" } ]，result 可能为合并后的单条或多条。
    """
    data = ext_response.get("data")
    if not data:
        return {}
    merged = {}
    for item in data:
        raw = item.get("result")
        if not raw:
            continue
        if isinstance(raw, str):
            try:
                raw = json.loads(raw)
            except json.JSONDecodeError:
                continue
        if isinstance(raw, dict):
            for k, v in raw.items():
                if v is None or v == "" or (isinstance(v, (dict, list)) and not v):
                    continue
                merged[k] = v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
        elif isinstance(raw, list):
            for elem in raw:
                if isinstance(elem, dict):
                    for k, v in elem.items():
                        if k not in merged and v not in (None, ""):
                            merged[k] = v if not isinstance(v, (dict, list)) else json.dumps(v, ensure_ascii=False)
    return merged


def run(
    root_dir: str,
    base_url: str = "http://0.0.0.0:8001",
    business_type: str = DEFAULT_BUSINESS_TYPE,
    batch_size: int = 20,
    output_dir: str = None,
    skip_extract: bool = False,
):
    """
    按文件夹区分单据，对每个附件先分类再抽取；按分类输出表格，每类一张表。
    """
    doc_files = collect_by_doc(root_dir)
    if not doc_files:
        print(f"未在 {root_dir} 下找到子文件夹及其中的支持文件（{SUPPORTED_EXT}）")
        return

    file_paths = [p for _, p in doc_files]
    print(f"共 {len(doc_files)} 个附件（{len(set(d for d, _ in doc_files))} 个单据），业务类型: {business_type}")

    out_dir = Path(output_dir or root_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) 批量分类
    classify_results = []
    for i in range(0, len(file_paths), batch_size):
        batch = file_paths[i : i + batch_size]
        print(f"分类进度: {i + 1}-{min(i + batch_size, len(file_paths))} / {len(file_paths)}")
        classify_results.extend(classify_batch(base_url, business_type, batch))

    # 2) 按 (单据编号, 附件路径) 整理，并做抽取；按分类归类
    by_classify = defaultdict(list)  # classify -> [ { doc_id, file_name, path, confidence, fields } ]
    full_results = []

    for (doc_id, path), cr in zip(doc_files, classify_results):
        file_name = os.path.basename(path)
        classify = cr.get("classify") if isinstance(cr, dict) else None
        confidence = cr.get("sorce") if isinstance(cr, dict) else None
        err = cr.get("error") if isinstance(cr, dict) else None

        row = {
            "doc_id": doc_id,
            "file_name": file_name,
            "file_path": path,
            "classify": classify or err or "未知",
            "confidence": confidence,
            "extract_fields": {},
            "extract_error": None,
        }

        if classify and classify in EXTRACT_SUPPORTED and not skip_extract:
            ex = extract_one(base_url, classify, path)
            row["extract_code"] = ex.get("code")
            row["extract_message"] = ex.get("message")
            if ex.get("code") == 200:
                row["extract_fields"] = extract_fields_from_response(ex)
            else:
                row["extract_error"] = ex.get("message") or "抽取失败"
        else:
            row["extract_error"] = "未抽取" if not skip_extract else "已跳过抽取"

        full_results.append(row)
        by_classify[row["classify"]].append(row)

    # 按分类写表格：每类一张 CSV，列为 附件编号、分类、置信度、以及该类抽取到的所有字段
    for classify_name, rows in sorted(by_classify.items()):
        if not rows:
            continue
        all_keys = ["附件编号", "分类", "置信度"]
        field_set = set()
        for r in rows:
            for k in r.get("extract_fields") or {}:
                field_set.add(k)
        all_keys.extend(sorted(field_set))

        safe_name = "".join(c if c not in r'\/:*?"<>|' else "_" for c in classify_name)
        csv_path = out_dir / f"表格_{safe_name}_{timestamp}.csv"
        with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore", restval="")
            w.writeheader()
            for r in rows:
                out_row = {
                    "附件编号": r["file_name"],
                    "分类": r["classify"],
                    "置信度": r.get("confidence") if r.get("confidence") is not None else "",
                }
                out_row.update(r.get("extract_fields") or {})
                w.writerow(out_row)
        print(f"已写入: {csv_path}")


def main():
    parser = argparse.ArgumentParser(
        description="按文件夹区分单据，批量分类+抽取，按分类输出表格（每类一张表，含编号及提取字段）"
    )
    parser.add_argument("root_dir", help="根目录：其下每个子文件夹为一个单据，子文件夹内文件为该单据附件")
    parser.add_argument("--base-url", default="http://0.0.0.0:8001", help="服务 base URL")
    parser.add_argument("--business-type", default=DEFAULT_BUSINESS_TYPE, help="分类接口的 businessType")
    parser.add_argument("--batch-size", type=int, default=20, help="分类每批请求文件数")
    parser.add_argument("--output-dir", default=None, help="结果输出目录（默认与 root_dir 相同）")
    parser.add_argument("--skip-extract", action="store_true", help="只做分类，不调用抽取")
    args = parser.parse_args()

    run(
        root_dir=args.root_dir,
        base_url=args.base_url,
        business_type=args.business_type,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        skip_extract=args.skip_extract,
    )


if __name__ == "__main__":
    main()
