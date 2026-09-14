"""Product, category, brand, and SKU normalization with uncertainty preservation.
"""
from typing import Any, Dict, List, Optional, Tuple
import re
import difflib
import pandas as pd


# Canonical dictionaries for common retail categories and products
CATEGORY_SYNONYMS = {
    "t-shirt": "Apparel - Tops",
    "tshirt": "Apparel - Tops",
    "tee": "Apparel - Tops",
    "t-shirts": "Apparel - Tops",
    "shirt": "Apparel - Tops",
    "hoodie": "Apparel - Outerwear",
    "sweatshirt": "Apparel - Outerwear",
    "jacket": "Apparel - Outerwear",
    "jeans": "Apparel - Bottoms",
    "denim": "Apparel - Bottoms",
    "pants": "Apparel - Bottoms",
    "trousers": "Apparel - Bottoms",
    "sneakers": "Footwear",
    "running shoes": "Footwear",
    "shoes": "Footwear",
    "boots": "Footwear",
    "laptop": "Electronics - Computers",
    "notebook": "Electronics - Computers",
    "macbook": "Electronics - Computers",
    "smartphone": "Electronics - Mobile",
    "phone": "Electronics - Mobile",
    "iphone": "Electronics - Mobile",
    "headphones": "Electronics - Audio",
    "earbuds": "Electronics - Audio",
}

BRAND_SYNONYMS = {
    "nike": "Nike",
    "nike inc": "Nike",
    "nike sportswear": "Nike",
    "adidas": "Adidas",
    "adidas ag": "Adidas",
    "adidas originals": "Adidas",
    "apple": "Apple",
    "apple inc": "Apple",
    "sony": "Sony",
    "sony corp": "Sony",
    "samsung": "Samsung",
    "samsung electronics": "Samsung",
    "puma": "Puma",
    "puma se": "Puma",
}


def normalize_sku(raw_sku: Any) -> Tuple[Optional[str], float, Optional[str]]:
    """Standardize SKU: uppercase, strip punctuation, validate length.
    Returns: (normalized_sku, confidence, uncertainty_note)
    """
    if pd.isna(raw_sku) or not str(raw_sku).strip():
        return None, 0.0, "Missing SKU"

    sku_str = str(raw_sku).strip().upper()
    cleaned = re.sub(r"[\s\-_]", "", sku_str)

    if not cleaned.isalnum():
        return sku_str, 0.5, "Non-alphanumeric characters present"

    if len(cleaned) < 4 or len(cleaned) > 20:
        return cleaned, 0.7, f"Unusual SKU length ({len(cleaned)})"

    return cleaned, 1.0, None


def normalize_brand(raw_brand: Any) -> Tuple[str, float, Optional[str]]:
    """Normalize brand name with uncertainty preservation."""
    if pd.isna(raw_brand) or not str(raw_brand).strip():
        return "Unknown", 0.0, "Missing brand"

    b_str = str(raw_brand).strip()
    b_lower = b_str.lower()

    # Exact match in synonyms
    if b_lower in BRAND_SYNONYMS:
        return BRAND_SYNONYMS[b_lower], 1.0, None

    # Fuzzy match
    for syn, canonical in BRAND_SYNONYMS.items():
        ratio = difflib.SequenceMatcher(None, b_lower, syn).ratio()
        if ratio >= 0.85:
            return canonical, round(ratio, 2), f"Fuzzy matched to '{canonical}' ({round(ratio, 2)})"

    # Preserved original with title case if unrecognized
    return b_str.title(), 0.75, "Brand not in canonical taxonomy"


def normalize_category_or_product(
    raw_val: Any,
    taxonomy: Optional[Dict[str, str]] = None
) -> Tuple[str, float, Optional[str]]:
    """Normalize category / product type while preserving uncertainty."""
    if pd.isna(raw_val) or not str(raw_val).strip():
        return "Uncategorized", 0.0, "Missing category"

    val_str = str(raw_val).strip()
    val_lower = val_str.lower()
    mapping = taxonomy or CATEGORY_SYNONYMS

    # 1. Direct match
    if val_lower in mapping:
        return mapping[val_lower], 1.0, None

    # 2. Substring match
    for term, canonical in mapping.items():
        if term in val_lower:
            return canonical, 0.9, f"Substring matched '{term}'"

    # 3. Fuzzy similarity
    best_match = None
    best_ratio = 0.0
    for term, canonical in mapping.items():
        ratio = difflib.SequenceMatcher(None, val_lower, term).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_match = canonical

    if best_ratio >= 0.80 and best_match:
        return best_match, round(best_ratio, 2), f"Fuzzy matched '{best_match}' ({round(best_ratio, 2)})"

    # 4. Retain original if uncertain
    return val_str.title(), 0.6, "Low confidence match; original preserved"


def enrich_products_dataset(df_products: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Apply retail normalization to products dataframe, recording uncertainty metadata."""
    df_out = df_products.copy()
    stats = {"total_rows": len(df_out), "normalized_categories": 0, "uncertain_items": 0}

    # Normalize category if present
    cat_col = next((c for c in df_out.columns if "category" in c.lower()), None)
    if cat_col:
        norm_cats = []
        confidences = []
        notes = []
        for val in df_out[cat_col]:
            canonical, conf, note = normalize_category_or_product(val)
            norm_cats.append(canonical)
            confidences.append(conf)
            notes.append(note)
            if conf < 0.85:
                stats["uncertain_items"] += 1
            else:
                stats["normalized_categories"] += 1

        df_out[f"{cat_col}_normalized"] = norm_cats
        df_out[f"{cat_col}_confidence"] = confidences
        df_out[f"{cat_col}_uncertainty_note"] = notes

    # Normalize brand if present
    brand_col = next((c for c in df_out.columns if "brand" in c.lower()), None)
    if brand_col:
        norm_brands = []
        brand_conf = []
        for val in df_out[brand_col]:
            canonical, conf, _ = normalize_brand(val)
            norm_brands.append(canonical)
            brand_conf.append(conf)
        df_out[f"{brand_col}_normalized"] = norm_brands
        df_out[f"{brand_col}_confidence"] = brand_conf

    # Normalize SKU if present
    sku_col = next((c for c in df_out.columns if "sku" in c.lower()), None)
    if sku_col:
        norm_skus = []
        for val in df_out[sku_col]:
            c_sku, _, _ = normalize_sku(val)
            norm_skus.append(c_sku)
        df_out[f"{sku_col}_clean"] = norm_skus

    return df_out, stats
