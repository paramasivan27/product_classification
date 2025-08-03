#!/usr/bin/env python3
"""
Demo script for the LLM-based Product Classification System
"""

import os
from classifier import get_classifier
from llm_taxonomy import GOOGLE_TAXONOMY_CATEGORIES

def main():
    print("🤖 LLM Product Classification System Demo")
    print("=" * 60)
    
    # Check for OpenAI API key
    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        print("✅ OpenAI API key found - LLM classification enabled")
        llm_type = "openai"
        llm_config = {"api_key": openai_key, "model": "gpt-3.5-turbo"}
    else:
        print("⚠️  OpenAI API key not found - using fallback methods")
        print("   Set OPENAI_API_KEY environment variable to enable LLM classification")
        llm_type = "none"
        llm_config = {}
    
    # Initialize the classifier
    classifier = get_classifier(llm_type=llm_type, llm_config=llm_config)
    
    # Demo products
    demo_products = [
        {
            "title": "Nike Air Max Running Shoes",
            "description": "Comfortable athletic shoes for running and training with advanced cushioning technology",
            "size": "US 10"
        },
        {
            "title": "iPhone 15 Pro Max",
            "description": "Latest smartphone with advanced camera system and A17 Pro chip for professional photography",
            "size": "256GB"
        },
        {
            "title": "Organic Cotton T-Shirt",
            "description": "Comfortable casual wear made from 100% organic cotton, perfect for everyday use",
            "size": "Large"
        },
        {
            "title": "Leather Sofa",
            "description": "Premium leather sofa for living room with comfortable seating and elegant design",
            "size": "3-seater"
        },
        {
            "title": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with active noise cancellation and long battery life",
            "size": "One size"
        },
        {
            "title": "Baby Diapers",
            "description": "Soft and absorbent diapers for babies, hypoallergenic and comfortable",
            "size": "Size 4"
        },
        {
            "title": "Gaming Laptop",
            "description": "High-performance gaming laptop with RTX graphics and fast processor",
            "size": "17-inch"
        }
    ]
    
    print(f"\n📋 Google Product Taxonomy Categories ({len(GOOGLE_TAXONOMY_CATEGORIES)}):")
    for i, category in enumerate(GOOGLE_TAXONOMY_CATEGORIES, 1):
        print(f"  {i:2d}. {category}")
    
    print("\n" + "=" * 60)
    print("🤖 LLM Classification Results:")
    print("=" * 60)
    
    for i, product in enumerate(demo_products, 1):
        print(f"\n{i}. {product['title']}")
        print(f"   Description: {product['description']}")
        print(f"   Size: {product['size']}")
        
        try:
            # Try with LLM first
            result_llm = classifier.classify_product(
                title=product['title'],
                description=product['description'],
                size=product['size'],
                use_llm=True
            )
            
            print(f"   🤖 LLM Result:")
            print(f"      → Category: {result_llm['category']}")
            print(f"      → Method: {result_llm['method']}")
            print(f"      → Confidence: {result_llm['confidence']}")
            
            if result_llm.get("llm_response"):
                print(f"      → LLM Response: {result_llm['llm_response'][:100]}...")
            
            # Try without LLM for comparison
            result_fallback = classifier.classify_product(
                title=product['title'],
                description=product['description'],
                size=product['size'],
                use_llm=False
            )
            
            if result_fallback['method'] != result_llm['method']:
                print(f"   🔄 Fallback Result:")
                print(f"      → Category: {result_fallback['category']}")
                print(f"      → Method: {result_fallback['method']}")
                print(f"      → Confidence: {result_fallback['confidence']}")
            
        except Exception as e:
            print(f"   → Error: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demo completed!")
    print("\n📖 Usage Instructions:")
    print("  Web Interface:")
    print("    streamlit run streamlit_ui.py")
    print("\n  Command Line:")
    print("    python classifier.py --title 'Product Title' --description 'Description'")
    print("    python classifier.py --title 'Product Title' --description 'Description' --llm-type openai")
    print("    python classifier.py --title 'Product Title' --description 'Description' --no-llm")
    print("\n🔧 Configuration:")
    print("  Set OPENAI_API_KEY environment variable for LLM classification")
    print("  Example: export OPENAI_API_KEY='your-api-key-here'")

if __name__ == "__main__":
    main() 