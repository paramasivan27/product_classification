# product_classification

This project provides a product classifier that attempts to map retail products to the Google product taxonomy. By default the classifier uses an open source language model to perform zero-shot classification and falls back to a simple keyword based approach if the model or its dependencies are unavailable.

## Installation

Install the optional dependency `transformers` to enable the language model based classifier:

```bash
pip install transformers
```

After installing, run the classifier directly with Python 3.

```bash
python -m product_classifier.classifier --title "Men's Running Shoes" --description "Comfortable athletic shoes" --size "US 10"
```

## Testing

Run the unit tests using `pytest`:

```bash
pytest
```
