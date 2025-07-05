import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from product_classifier.classifier import identify_google_taxonomy


def test_apparel_classification():
    category = identify_google_taxonomy(
        title="Men's Running Shoes",
        description="Comfortable athletic shoes for running",
        size="US 10"
    )
    assert category == "Apparel & Accessories"


def test_unknown_classification():
    category = identify_google_taxonomy(
        title="Mystery Item",
        description="This product does not match any known category",
        size=""
    )
    assert category == "Unknown"
