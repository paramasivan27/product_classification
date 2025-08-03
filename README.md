# AI Retail Product Classification

This repository contains a sophisticated product classification system that uses advanced NLP models to categorize retail products into predefined categories.

## Quick Start

The main application is located in the `product_classifier/` directory. Please refer to the [product_classifier/README.md](product_classifier/README.md) for detailed documentation.

### Quick Commands

```bash
# Navigate to the application directory
cd product_classifier

# Install dependencies
pip install -r requirements.txt

# Run the web interface
streamlit run streamlit_ui.py

# Or run with Docker
docker-compose up --build
```

## Project Structure

```
product_classification/
├── product_classifier/          # Main application directory
│   ├── classifier.py           # Core classification logic
│   ├── taxonomy.py            # Product categories and keywords
│   ├── streamlit_ui.py        # Web interface
│   ├── requirements.txt       # Python dependencies
│   ├── Dockerfile            # Docker configuration
│   ├── docker-compose.yml    # Docker Compose configuration
│   └── README.md            # Detailed documentation
└── tests/                    # Test files
    └── test_classifier.py
```

## Features

- **Advanced NLP Models**: Uses sentence transformers for semantic understanding
- **Comprehensive Categories**: 10 major product categories with detailed keywords
- **Fallback System**: Automatic fallback to keyword matching if ML models fail
- **Modern UI**: Beautiful Streamlit interface with examples and detailed results
- **Docker Support**: Easy deployment with Docker and Docker Compose
- **CLI Interface**: Command-line tool for batch processing

## Documentation

For complete documentation, installation instructions, and usage examples, please see:

**[product_classifier/README.md](product_classifier/README.md)**

## License

This project is licensed under the MIT License. 