from __future__ import annotations

import argparse
import os
from typing import Optional, List

from .taxonomy import TAXONOMY_KEYWORDS


def _keyword_based_classification(text: str) -> str:
    """Classify using simple keyword matching."""
    scores = {}
    for category, keywords in TAXONOMY_KEYWORDS.items():
        scores[category] = sum(1 for kw in keywords if kw in text)

    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return "Unknown"
    return best_category


def identify_google_taxonomy(title: str, description: str, size: str = "", image_path: Optional[str] = None) -> str:
    """Identify the Google Taxonomy for a product.

    Args:
        title: Product title.
        description: Product description.
        size: Optional size information.
        image_path: Optional path to an image file. This module does not
            analyse the image but checks whether the path exists.

    Returns:
        A string representing the best matching Google Taxonomy category.
    """
    text = f"{title} {description} {size}".lower()

    # Basic validation of the image path if provided
    if image_path and not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    categories: List[str] = list(TAXONOMY_KEYWORDS.keys())
    try:
        from transformers import pipeline  # type: ignore

        classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
        )
        result = classifier(text, categories)
        if result and "labels" in result and result["labels"]:
            return result["labels"][0]
    except Exception:
        # Any failure (e.g. transformers not installed or model download issue)
        # falls back to rule-based classification.
        pass

    return _keyword_based_classification(text)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Classify a retail product using Google Taxonomy keywords")
    parser.add_argument("--title", required=True, help="Product title")
    parser.add_argument("--description", required=True, help="Product description")
    parser.add_argument("--size", default="", help="Size description")
    parser.add_argument("--image", help="Path to product image")
    args = parser.parse_args(argv)

    category = identify_google_taxonomy(args.title, args.description, args.size, args.image)
    print(category)


if __name__ == "__main__":  # pragma: no cover
    main()
