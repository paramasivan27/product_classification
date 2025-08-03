#!/usr/bin/env python3
"""
Demo script for the Product Classification System
"""

from classifier import get_classifier
from taxonomy import PRODUCT_CATEGORIES

def main():
    print("🛍️ Product Classification System Demo")
    print("=" * 50)
    
    # Initialize the classifier
    classifier = get_classifier()
    
    # Demo products
    demo_products = [
        {
            "title": "Nike Air Max Running Shoes",
            "description": "Comfortable athletic shoes for running and training with advanced cushioning",
            "size": "US 10"
        },
        {
            "title": "iPhone 15 Pro Max",
            "description": "Latest smartphone with advanced camera system and A17 Pro chip",
            "size": "256GB"
        },
        {
            "title": "Organic Cotton T-Shirt",
            "description": "Comfortable casual wear made from 100% organic cotton",
            "size": "Large"
        },
        {
            "title": "Leather Sofa",
            "description": "Premium leather sofa for living room with comfortable seating",
            "size": "3-seater"
        },
        {
            "title": "Wireless Bluetooth Headphones",
            "description": "High-quality wireless headphones with noise cancellation",
            "size": "One size"
        }
    ]
    
    print(f"Available Categories: {len(PRODUCT_CATEGORIES)}")
    for i, category in enumerate(PRODUCT_CATEGORIES.keys(), 1):
        print(f"  {i}. {category}")
    
    print("\n" + "=" * 50)
    print("Classification Results:")
    print("=" * 50)
    
    for i, product in enumerate(demo_products, 1):
        print(f"\n{i}. {product['title']}")
        print(f"   Description: {product['description']}")
        print(f"   Size: {product['size']}")
        
        try:
            result = classifier.classify_product(
                title=product['title'],
                description=product['description'],
                size=product['size']
            )
            
            print(f"   → Category: {result['category']}")
            print(f"   → Method: {result['method']}")
            print(f"   → Confidence: {result['confidence']}")
            
        except Exception as e:
            print(f"   → Error: {e}")
    
    print("\n" + "=" * 50)
    print("Demo completed! 🎉")
    print("\nTo run the web interface:")
    print("  streamlit run streamlit_ui.py")
    print("\nTo use the command line interface:")
    print("  python classifier.py --title 'Product Title' --description 'Product Description'")

if __name__ == "__main__":
    main() 