# AI Retail Product Classification with LLMs

This repository contains a sophisticated product classification system that uses Large Language Models (LLMs) and advanced NLP models to categorize retail products into Google Product Taxonomy categories.

## 🚀 Quick Start

The main application is located in the `product_classifier/` directory. Please refer to the [product_classifier/README.md](product_classifier/README.md) for detailed documentation.

### Quick Commands

```bash
# Navigate to the application directory
cd product_classifier

# Install dependencies
pip install -r requirements.txt

# Set up OpenAI API key (optional, for LLM classification)
export OPENAI_API_KEY='your-api-key-here'

# Run the web interface
streamlit run streamlit_ui.py

# Or run with Docker
docker-compose up --build
```

## 🤖 LLM Features

- **OpenAI Integration**: GPT-3.5-turbo, GPT-4, GPT-4-turbo for intelligent classification
- **Hugging Face Support**: Local LLM models (DialoGPT, GPT-2, etc.)
- **Google Product Taxonomy**: Uses official Google Product Taxonomy with 24 categories
- **Fallback System**: Automatic fallback to semantic and keyword-based classification
- **Flexible Configuration**: Easy switching between LLM providers

## 📁 Project Structure

```
product_classification/
├── product_classifier/          # Main application directory
│   ├── classifier.py           # Core classification logic with LLM integration
│   ├── llm_taxonomy.py        # LLM-based taxonomy classification
│   ├── taxonomy.py            # Legacy taxonomy (for fallback)
│   ├── streamlit_ui.py        # Web interface with LLM configuration
│   ├── demo.py               # Demo script showcasing LLM features
│   ├── requirements.txt      # Python dependencies including LLM libraries
│   ├── Dockerfile           # Docker configuration
│   ├── docker-compose.yml   # Docker Compose configuration
│   └── README.md           # Detailed documentation
└── tests/                    # Test files
    └── test_classifier.py
```

## 🎯 Key Features

- **🤖 LLM Integration**: OpenAI GPT and Hugging Face models for intelligent classification
- **📋 Google Product Taxonomy**: Uses official Google Product Taxonomy categories
- **🧠 Advanced NLP Models**: Sentence transformers for semantic understanding
- **🔄 Fallback System**: Automatic fallback to keyword matching if LLMs fail
- **🎨 Modern UI**: Beautiful Streamlit interface with LLM configuration
- **🐳 Docker Support**: Easy deployment with Docker and Docker Compose
- **💻 CLI Interface**: Command-line tool with LLM options
- **🔧 Flexible Configuration**: Support for multiple LLM providers

## 📋 Google Product Taxonomy Categories

The system uses the official Google Product Taxonomy with 24 major categories including:
- Animals & Pet Supplies
- Apparel & Accessories
- Arts & Entertainment
- Baby & Toddler
- Beauty & Personal Care
- Books & Magazines
- Business & Industrial
- Cameras & Optics
- Clothing, Shoes & Jewelry
- Computers & Electronics
- Food, Beverages & Tobacco
- Furniture
- Hardware
- Health & Beauty
- Home & Garden
- Luggage & Bags
- Mature
- Media
- Office Products
- Religious & Ceremonial
- Software
- Sporting Goods
- Toys & Games
- Vehicles & Parts

## 🔧 Usage Examples

### Web Interface
```bash
cd product_classifier
streamlit run streamlit_ui.py
```

### Command Line
```bash
# Basic classification
python classifier.py --title "Men's Running Shoes" --description "Comfortable athletic shoes"

# With OpenAI LLM
python classifier.py --title "iPhone 15 Pro" --description "Latest smartphone" --llm-type openai

# With Hugging Face LLM
python classifier.py --title "Gaming Laptop" --description "High-performance laptop" --llm-type huggingface

# Disable LLM (use fallback methods only)
python classifier.py --title "Product" --description "Description" --no-llm
```

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
    use_llm=True
)
```

## 📚 Documentation

For complete documentation, installation instructions, and usage examples, please see:

**[product_classifier/README.md](product_classifier/README.md)**

## 🔑 Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key for LLM classification

### LLM Configuration
- **OpenAI**: Requires API key, supports GPT-3.5-turbo, GPT-4, GPT-4-turbo
- **Hugging Face**: Local models, no API costs, supports DialoGPT, GPT-2, etc.

## 🧪 Testing

```bash
cd product_classifier
python demo.py  # Run demo with LLM features
python -m pytest ../tests/  # Run test suite
```

## 📄 License

This project is licensed under the MIT License. 