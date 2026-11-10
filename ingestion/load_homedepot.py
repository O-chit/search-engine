"""Nạp, ghép bảng và làm sạch dữ liệu sản phẩm Home Depot."""

import io
import os
import unicodedata
from typing import Any

import pandas as pd

from config import (
    ATTR_MAX,
    PROCESSED_DATA_DIR,
    PRODUCTS_PARQUET_PATH,
    RAW_DATA_DIR,
)
from ingestion.normalize import clean_text


def read_csv_robust(file_path: str, **kwargs: Any) -> pd.DataFrame:
    """Đọc file CSV bền vững bằng cách giải mã từng dòng theo thứ tự utf-8, cp1252, ISO-8859-1."""
    counts = {"utf-8": 0, "cp1252": 0, "ISO-8859-1": 0}
    decoded_lines = []
    with open(file_path, "rb") as f:
        for line_bytes in f:
            try:
                line_str = line_bytes.decode("utf-8")
                counts["utf-8"] += 1
            except UnicodeDecodeError:
                try:
                    line_str = line_bytes.decode("cp1252")
                    counts["cp1252"] += 1
                except UnicodeDecodeError:
                    line_str = line_bytes.decode("ISO-8859-1")
                    counts["ISO-8859-1"] += 1
            decoded_lines.append(line_str)

    fname = os.path.basename(file_path)
    print(
        f"File: {fname} -> số dòng giải mã: "
        f"utf-8={counts['utf-8']}, "
        f"cp1252={counts['cp1252']}, "
        f"ISO-8859-1={counts['ISO-8859-1']}"
    )
    return pd.read_csv(io.StringIO("".join(decoded_lines)), **kwargs)


def inspect_raw_files(raw_dir: str) -> None:
    """In tên cột, số dòng theo từng encoding và 3 dòng đầu của từng file dữ liệu thô."""
    files = [
        "train.csv",
        "test.csv",
        "product_descriptions.csv",
        "attributes.csv",
    ]
    print("=== KIỂM TRA LƯỢC ĐỒ VÀ MÃ HÓA CÁC FILE RAW ===")
    for fname in files:
        fpath = os.path.join(raw_dir, fname)
        df_sample = read_csv_robust(fpath, nrows=3, dtype=str)
        print(f"Tên cột: {list(df_sample.columns)}")
        print("3 dòng đầu:")
        print(df_sample.to_string(index=False))
        print("-" * 50)


def load_and_process_homedepot() -> pd.DataFrame:
    """Nạp, làm sạch và lưu dữ liệu Home Depot thành products.parquet."""
    inspect_raw_files(RAW_DATA_DIR)

    train_path = os.path.join(RAW_DATA_DIR, "train.csv")
    test_path = os.path.join(RAW_DATA_DIR, "test.csv")
    desc_path = os.path.join(RAW_DATA_DIR, "product_descriptions.csv")
    attr_path = os.path.join(RAW_DATA_DIR, "attributes.csv")

    print("\n[1/5] Đọc train.csv và test.csv...")
    train_df = read_csv_robust(
        train_path, dtype={"product_uid": str, "product_title": str}
    )
    test_df = read_csv_robust(
        test_path, dtype={"product_uid": str, "product_title": str}
    )

    combined_titles = pd.concat(
        [
            train_df[["product_uid", "product_title"]],
            test_df[["product_uid", "product_title"]],
        ],
        ignore_index=True,
    )
    combined_titles = combined_titles.dropna(subset=["product_uid"])
    products = combined_titles.drop_duplicates(
        subset=["product_uid"], keep="first"
    ).copy()
    products["product_uid"] = products["product_uid"].astype(str)
    products["title"] = products["product_title"].fillna("").apply(
        lambda s: unicodedata.normalize("NFC", str(s)).strip()
    )
    products = products[["product_uid", "title"]]
    print(f" -> Tổng số product_uid hợp nhất: {len(products)}")

    print("[2/5] Đọc và xử lý attributes.csv...")
    attr_df = read_csv_robust(
        attr_path, dtype={"product_uid": str, "name": str, "value": str}
    )
    attr_df = attr_df.dropna(subset=["product_uid", "name"]).copy()
    attr_df["product_uid"] = attr_df["product_uid"].astype(str)
    attr_df["name"] = attr_df["name"].astype(str)
    attr_df["value"] = attr_df["value"].fillna("").astype(str)

    # Brand: name == 'MFG Brand Name', title và brand chỉ NFC và strip
    brand_mask = attr_df["name"] == "MFG Brand Name"
    brand_df = attr_df[brand_mask].drop_duplicates(
        subset=["product_uid"], keep="first"
    ).copy()
    brand_df["brand"] = brand_df["value"].apply(
        lambda s: unicodedata.normalize("NFC", str(s)).strip()
    )
    brand_map = brand_df.set_index("product_uid")["brand"]

    # Bullets: name bắt đầu bằng Bullet, sắp theo tên, nối khoảng trắng, áp dụng clean_text
    bullet_mask = attr_df["name"].str.startswith("Bullet")
    bullet_df = attr_df[bullet_mask].sort_values(by=["product_uid", "name"])
    bullet_grouped = bullet_df.groupby("product_uid")["value"].agg(" ".join)
    bullet_cleaned = bullet_grouped.apply(clean_text)

    # attributes_text: loại Bullet* và MFG Brand Name, tối đa ATTR_MAX thuộc tính mỗi sp, nối '; ', clean_text
    other_mask = ~brand_mask & ~bullet_mask
    other_attr_df = (
        attr_df[other_mask]
        .groupby("product_uid", sort=False)
        .head(ATTR_MAX)
        .copy()
    )
    other_attr_df["pair"] = other_attr_df["name"] + ": " + other_attr_df["value"]
    attr_grouped = other_attr_df.groupby("product_uid")["pair"].agg("; ".join)
    attr_cleaned = attr_grouped.apply(clean_text)
    print(" -> Đã trích xuất xong brand, bullets và attributes_text")

    print("[3/5] Đọc và xử lý product_descriptions.csv...")
    desc_df = read_csv_robust(
        desc_path, dtype={"product_uid": str, "product_description": str}
    )
    desc_df = desc_df.dropna(subset=["product_uid"]).copy()
    desc_df["product_uid"] = desc_df["product_uid"].astype(str)
    desc_df["description"] = (
        desc_df["product_description"].fillna("").apply(clean_text)
    )
    desc_map = (
        desc_df.drop_duplicates(subset=["product_uid"], keep="first")
        .set_index("product_uid")["description"]
    )
    print(" -> Đã làm sạch xong product_description")

    print("[4/5] Ghép bảng và sắp xếp đúng cột...")
    products["brand"] = products["product_uid"].map(brand_map).fillna("")
    products["bullets"] = products["product_uid"].map(bullet_cleaned).fillna("")
    products["description"] = products["product_uid"].map(desc_map).fillna("")
    products["attributes_text"] = products["product_uid"].map(attr_cleaned).fillna("")

    final_columns = [
        "product_uid",
        "title",
        "brand",
        "bullets",
        "description",
        "attributes_text",
    ]
    products = products[final_columns].copy()
    for col in final_columns:
        products[col] = products[col].astype(str)

    print(f"[5/5] Ghi dữ liệu vào {PRODUCTS_PARQUET_PATH}...")
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    products.to_parquet(PRODUCTS_PARQUET_PATH, index=False, engine="pyarrow")
    print(" -> Ghi file parquet thành công")

    total_products = len(products)
    has_brand = int((products["brand"] != "").sum())
    has_bullets = int((products["bullets"] != "").sum())
    has_description = int((products["description"] != "").sum())
    has_attributes = int((products["attributes_text"] != "").sum())

    print("\n" + "=" * 50)
    print(
        f"Thống kê cuối: số sản phẩm={total_products}, "
        f"có brand={has_brand}, "
        f"có bullets={has_bullets}, "
        f"có description={has_description}, "
        f"có attributes_text={has_attributes}"
    )
    print("=" * 50)

    return products


def main() -> None:
    """Hàm chính thực thi nạp dữ liệu Home Depot từ dòng lệnh."""
    load_and_process_homedepot()


if __name__ == "__main__":
    main()
