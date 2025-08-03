# LLM Product Classification System

A sophisticated product classification system that uses Large Language Models (LLMs) and advanced NLP models to categorize retail products into Google Product Taxonomy categories. The system combines LLM-based classification with semantic similarity and keyword-based fallback for robust product categorization.

## Features

- **🤖 LLM Integration**: OpenAI GPT and Hugging Face models for intelligent classification
- **📋 Google Product Taxonomy**: Uses official Google Product Taxonomy categories
- **🧠 Advanced NLP Models**: Sentence transformers for semantic understanding
- **🔄 Fallback System**: Automatic fallback to keyword matching if LLMs fail
- **🎨 Modern UI**: Beautiful Streamlit interface with LLM configuration
- **🐳 Docker Support**: Easy deployment with Docker and Docker Compose
- **💻 CLI Interface**: Command-line tool with LLM options
- **🔧 Flexible Configuration**: Support for multiple LLM providers

## LLM Providers

### OpenAI
- **Models**: GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Features**: High accuracy, structured JSON responses
- **Setup**: Requires OpenAI API key

### Hugging Face
- **Models**: DialoGPT, GPT-2, DistilGPT-2
- **Features**: Local deployment, no API costs
- **Setup**: Automatic model download

## Google Product Taxonomy Categories

The system uses the official Google Product Taxonomy with 24 major categories:

1. **Animals & Pet Supplies**
2. **Apparel & Accessories**
3. **Arts & Entertainment**
4. **Baby & Toddler**
5. **Beauty & Personal Care**
6. **Books & Magazines**
7. **Business & Industrial**
8. **Cameras & Optics**
9. **Clothing, Shoes & Jewelry**
10. **Computers & Electronics**
11. **Food, Beverages & Tobacco**
12. **Furniture**
13. **Hardware**
14. **Health & Beauty**
15. **Home & Garden**
16. **Luggage & Bags**
17. **Mature**
18. **Media**
19. **Office Products**
20. **Religious & Ceremonial**
21. **Software**
22. **Sporting Goods**
23. **Toys & Games**
24. **Vehicles & Parts**

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd product_classifier
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up OpenAI API key:
```bash
export OPENAI_API_KEY='your-api-key-here'
```

4. Run the application:
```bash
streamlit run streamlit_ui.py
```

### Option 2: Docker Installation

1. Build and run with Docker Compose:
```bash
docker-compose up --build
```

2. Or build and run manually:
```bash
docker build -t product-classifier .
docker run -p 8501:8501 -e OPENAI_API_KEY=your-key product-classifier
```

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:8501`
2. Configure LLM settings in the sidebar
3. Enter product title and description
4. Choose whether to use LLM classification
5. Click "Classify Product" to get results

### Command Line Interface

```bash
# Basic classification
python classifier.py --title "Men's Running Shoes" --description "Comfortable athletic shoes" --size "US 10"

# With OpenAI LLM
python classifier.py --title "iPhone 15 Pro" --description "Latest smartphone" --llm-type openai

# With Hugging Face LLM
python classifier.py --title "Gaming Laptop" --description "High-performance laptop" --llm-type huggingface

# Disable LLM (use fallback methods only)
python classifier.py --title "Product" --description "Description" --no-llm
```

Options:
- `--title`: Product title (required)
- `--description`: Product description (required)
- `--size`: Size or additional info (optional)
- `--image`: Path to product image (optional)
- `--llm-type`: LLM provider ("openai", "huggingface", "none")
- `--no-llm`: Disable LLM classification
- `--verbose`: Enable verbose output

### Programmatic Usage

```python
from product_classifier.classifier import get_classifier

# With OpenAI
classifier = get_classifier(
    llm_type="openai",
    llm_config={"api_key": "your-key", "model": "gpt-3.5-turbo"}
)
result = classifier.classify_product(
    title="iPhone 15 Pro",
    description="Latest smartphone with advanced features",
    size="256GB",
    use_llm=True
)

print(f"Category: {result['category']}")
print(f"Method: {result['method']}")
print(f"Confidence: {result['confidence']}")
```

## Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key for LLM classification

### LLM Configuration

#### OpenAI
```python
llm_config = {
    "api_key": "your-openai-api-key",
    "model": "gpt-3.5-turbo"  # or "gpt-4", "gpt-4-turbo"
}
```

#### Hugging Face
```python
llm_config = {
    "model_name": "microsoft/DialoGPT-medium"  # or "gpt2", "distilgpt2"
}
```

## Technical Details

### Classification Methods

1. **LLM Classification**: Uses OpenAI GPT or Hugging Face models for intelligent classification
2. **Semantic Classification**: Uses sentence transformers to compute embeddings and find similar categories
3. **Keyword Classification**: Fallback method using keyword matching when ML models are unavailable

### Models Used

- **Primary LLM**: OpenAI GPT-3.5-turbo or Hugging Face DialoGPT
- **Semantic**: `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- **Fallback**: Keyword-based matching with Google Product Taxonomy

### Performance

- **Accuracy**: High accuracy with LLM classification
- **Speed**: Fast classification with pre-computed embeddings
- **Reliability**: Robust fallback system ensures classification always works
- **Cost**: OpenAI API calls incur costs, Hugging Face models are free

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
product_classifier/
├── classifier.py          # Main classification logic
├── llm_taxonomy.py       # LLM-based taxonomy classification
├── taxonomy.py           # Legacy taxonomy (for fallback)
├── streamlit_ui.py       # Web interface
├── demo.py              # Demo script
├── requirements.txt     # Python dependencies
├── Dockerfile          # Docker configuration
├── docker-compose.yml  # Docker Compose configuration
└── README.md          # This file
```

### Adding New LLM Providers

1. Create a new class inheriting from `LLMTaxonomyClassifier`
2. Implement the `classify_product` method
3. Add the provider to the `get_llm_classifier` factory function
4. Update the UI to support the new provider

## Troubleshooting

### Common Issues

1. **OpenAI API Key Issues**: Ensure `OPENAI_API_KEY` is set correctly
2. **Model Download Issues**: Hugging Face models will download automatically
3. **Memory Issues**: LLM models require significant RAM
4. **Network Issues**: OpenAI requires internet connection

### Error Messages

- **"OpenAI API key not found"**: Set the `OPENAI_API_KEY` environment variable
- **"LLM classification failed"**: Check API key and network connection
- **"Model not loaded"**: Ensure sufficient memory and disk space

### Logs

Enable verbose logging:
```bash
python classifier.py --verbose --title "test" --description "test"
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 