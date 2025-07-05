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

## Streamlit UI

Install the optional dependency `streamlit` and launch the interactive interface:

```bash
pip install streamlit
streamlit run product_classifier/streamlit_ui.py
```

## Docker

Use docker-compose to build the image and run the Streamlit interface:

```bash
docker-compose up --build
```

Once running, open http://localhost:8501 in your browser. You can also run the
CLI inside the container:

```bash
docker compose run --rm app python -m product_classifier.classifier --title "Men's Running Shoes" --description "Comfortable athletic shoes" --size "US 10"
```
