# product_classification

This project provides a simple rule-based classifier that attempts to map retail products to the Google product taxonomy. The classifier matches keywords in the product title, description and size description to a small set of categories.

## Installation

This project has no external dependencies. Clone the repository and run the classifier directly with Python 3.

```bash
python -m product_classifier.classifier --title "Men's Running Shoes" --description "Comfortable athletic shoes" --size "US 10"
```

## Testing

Run the unit tests using `pytest`:

```bash
pytest
```
