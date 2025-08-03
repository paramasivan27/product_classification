# Product Classification System

A sophisticated product classification system that uses advanced NLP models to categorize retail products into predefined categories. The system combines semantic similarity with sentence transformers and keyword-based classification for robust product categorization.

## Features

- **Advanced NLP Models**: Uses sentence transformers for semantic understanding
- **Comprehensive Categories**: 10 major product categories with detailed keywords
- **Fallback System**: Automatic fallback to keyword matching if ML models fail
- **Modern UI**: Beautiful Streamlit interface with examples and detailed results
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **CLI Interface**: Command-line tool for batch processing

## Product Categories

The system supports the following product categories:

1. **Apparel & Accessories** - Clothing, shoes, jewelry, bags
2. **Electronics & Technology** - Phones, computers, gadgets
3. **Home & Garden** - Furniture, decor, garden items
4. **Sports & Outdoors** - Athletic gear, camping equipment
5. **Health & Beauty** - Cosmetics, personal care, wellness
6. **Toys & Games** - Children's toys, board games, video games
7. **Books & Media** - Books, movies, music, magazines
8. **Automotive** - Car parts, accessories, tools
9. **Food & Beverages** - Food, drinks, snacks, alcohol
10. **Baby & Kids** - Baby products, children's items

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

3. Run the application:
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
docker run -p 8501:8501 product-classifier
```

## Usage

### Web Interface

1. Open your browser and go to `http://localhost:8501`
2. Enter the product title and description
3. Optionally add size or additional information
4. Click "Classify Product" to get results

### Command Line Interface

```bash
python classifier.py --title "Men's Running Shoes" --description "Comfortable athletic shoes for running" --size "US 10"
```

Options:
- `--title`: Product title (required)
- `--description`: Product description (required)
- `--size`: Size or additional info (optional)
- `--image`: Path to product image (optional, not used in current version)
- `--verbose`: Enable verbose output

### Programmatic Usage

```python
from product_classifier.classifier import get_classifier

classifier = get_classifier()
result = classifier.classify_product(
    title="iPhone 15 Pro",
    description="Latest smartphone with advanced features",
    size="256GB"
)

print(f"Category: {result['category']}")
print(f"Confidence: {result['confidence']}")
print(f"Method: {result['method']}")
```

## Technical Details

### Classification Methods

1. **Semantic Classification**: Uses sentence transformers to compute embeddings and find the most similar category
2. **Keyword Classification**: Fallback method using keyword matching when ML models are unavailable

### Models Used

- **Primary**: `sentence-transformers/all-MiniLM-L6-v2` for semantic similarity
- **Fallback**: Keyword-based matching with comprehensive product taxonomies

### Performance

- **Accuracy**: High accuracy for well-described products
- **Speed**: Fast classification with pre-computed category embeddings
- **Reliability**: Robust fallback system ensures classification always works

## Development

### Running Tests

```bash
pytest tests/
```

### Project Structure

```
product_classifier/
├── classifier.py          # Main classification logic
├── taxonomy.py           # Product categories and keywords
├── streamlit_ui.py       # Web interface
├── requirements.txt      # Python dependencies
├── Dockerfile           # Docker configuration
├── docker-compose.yml   # Docker Compose configuration
├── README.md           # This file
└── tests/              # Test files
    └── test_classifier.py
```

### Adding New Categories

1. Edit `taxonomy.py` to add new categories and keywords
2. Update tests to include new categories
3. The system will automatically use new categories

## Troubleshooting

### Common Issues

1. **Model Download Issues**: The system will automatically fall back to keyword matching
2. **Memory Issues**: The sentence transformer model requires ~500MB RAM
3. **Docker Issues**: Ensure Docker has sufficient memory allocated

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