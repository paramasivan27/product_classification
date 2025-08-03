from __future__ import annotations

import argparse
import os
from typing import Optional, List, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from .taxonomy import PRODUCT_CATEGORIES
except ImportError:
    from taxonomy import PRODUCT_CATEGORIES


class ProductClassifier:
    """A product classifier that uses advanced NLP models for product categorization."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the product classifier.
        
        Args:
            model_name: The sentence transformer model to use for embeddings
        """
        self.model_name = model_name
        self.model = None
        self.category_embeddings = None
        self._load_model()
    
    def _load_model(self) -> None:
        """Load the sentence transformer model and compute category embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            logger.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Compute embeddings for all category descriptions
            category_descriptions = [
                f"Products in the category of {category}" 
                for category in PRODUCT_CATEGORIES.keys()
            ]
            
            self.category_embeddings = self.model.encode(category_descriptions)
            logger.info("Model loaded successfully")
            
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to keyword matching")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model = None
    
    def _semantic_classification(self, text: str) -> str:
        """Classify using semantic similarity with sentence transformers."""
        if self.model is None or self.category_embeddings is None:
            return self._keyword_based_classification(text)
        
        try:
            from numpy import dot
            from numpy.linalg import norm
            
            # Encode the input text
            text_embedding = self.model.encode([text])[0]
            
            # Calculate cosine similarity with all categories
            similarities = []
            for category_embedding in self.category_embeddings:
                cos_sim = dot(text_embedding, category_embedding) / (
                    norm(text_embedding) * norm(category_embedding)
                )
                similarities.append(cos_sim)
            
            # Find the category with highest similarity
            best_idx = max(range(len(similarities)), key=lambda i: similarities[i])
            best_category = list(PRODUCT_CATEGORIES.keys())[best_idx]
            
            # Only return if similarity is above threshold
            if similarities[best_idx] > 0.3:
                return best_category
            else:
                return "Unknown"
                
        except Exception as e:
            logger.error(f"Error in semantic classification: {e}")
            return self._keyword_based_classification(text)
    
    def _keyword_based_classification(self, text: str) -> str:
        """Classify using keyword matching as fallback."""
        text_lower = text.lower()
        scores = {}
        
        for category, keywords in PRODUCT_CATEGORIES.items():
            score = sum(1 for keyword in keywords if keyword.lower() in text_lower)
            scores[category] = score
        
        best_category = max(scores, key=scores.get)
        if scores[best_category] == 0:
            return "Unknown"
        return best_category
    
    def classify_product(self, title: str, description: str, size: str = "", 
                        image_path: Optional[str] = None) -> Dict[str, Any]:
        """Classify a product using title and description.
        
        Args:
            title: Product title
            description: Product description
            size: Optional size information
            image_path: Optional path to product image (not used in current implementation)
            
        Returns:
            Dictionary containing classification results
        """
        # Validate inputs
        if not title or not description:
            raise ValueError("Both title and description are required")
        
        # Validate image path if provided
        if image_path and not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        # Combine text for classification
        combined_text = f"{title} {description} {size}".strip()
        
        # Perform classification
        category = self._semantic_classification(combined_text)
        
        return {
            "category": category,
            "confidence": "high" if self.model is not None else "medium",
            "input_text": combined_text,
            "method": "semantic" if self.model is not None else "keyword"
        }


# Global classifier instance
_classifier_instance = None

def get_classifier() -> ProductClassifier:
    """Get or create a global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ProductClassifier()
    return _classifier_instance


def identify_google_taxonomy(title: str, description: str, size: str = "", 
                           image_path: Optional[str] = None) -> str:
    """Legacy function for backward compatibility."""
    classifier = get_classifier()
    result = classifier.classify_product(title, description, size, image_path)
    return result["category"]


def main(argv: Optional[List[str]] = None) -> None:
    """Command line interface for the product classifier."""
    parser = argparse.ArgumentParser(
        description="Classify retail products using advanced NLP models"
    )
    parser.add_argument("--title", required=True, help="Product title")
    parser.add_argument("--description", required=True, help="Product description")
    parser.add_argument("--size", default="", help="Size description")
    parser.add_argument("--image", help="Path to product image")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        classifier = get_classifier()
        result = classifier.classify_product(
            args.title, args.description, args.size, args.image
        )
        
        print(f"Category: {result['category']}")
        print(f"Method: {result['method']}")
        print(f"Confidence: {result['confidence']}")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
