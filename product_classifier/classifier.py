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

try:
    from .llm_taxonomy import get_llm_classifier, GOOGLE_TAXONOMY_CATEGORIES
except ImportError:
    from llm_taxonomy import get_llm_classifier, GOOGLE_TAXONOMY_CATEGORIES


class ProductClassifier:
    """A product classifier that uses LLMs and advanced NLP models for product categorization."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 llm_type: str = "openai", llm_config: Optional[Dict] = None):
        """Initialize the product classifier.
        
        Args:
            model_name: The sentence transformer model to use for embeddings
            llm_type: Type of LLM to use ("openai", "huggingface", or "none")
            llm_config: Configuration for LLM (API keys, model names, etc.)
        """
        self.model_name = model_name
        self.llm_type = llm_type.lower()
        self.llm_config = llm_config or {}
        
        # Initialize components
        self.model = None
        self.category_embeddings = None
        self.llm_classifier = None
        
        # Load models
        self._load_sentence_transformer()
        self._load_llm_classifier()
    
    def _load_sentence_transformer(self) -> None:
        """Load the sentence transformer model and compute category embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
            
            logger.info(f"Loading sentence transformer model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            
            # Use Google Taxonomy categories for embeddings
            category_descriptions = [
                f"Products in the category of {category}" 
                for category in GOOGLE_TAXONOMY_CATEGORIES
            ]
            
            self.category_embeddings = self.model.encode(category_descriptions)
            logger.info("Sentence transformer model loaded successfully")
            
        except ImportError:
            logger.warning("sentence-transformers not available, falling back to keyword matching")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading sentence transformer model: {e}")
            self.model = None
    
    def _load_llm_classifier(self) -> None:
        """Load the LLM classifier."""
        if self.llm_type == "none":
            logger.info("LLM classifier disabled")
            return
            
        try:
            if self.llm_type == "openai":
                # Check for OpenAI API key
                api_key = self.llm_config.get("api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    logger.warning("OpenAI API key not found. LLM classification will be disabled.")
                    return
                
                model = self.llm_config.get("model", "gpt-3.5-turbo")
                self.llm_classifier = get_llm_classifier("openai", api_key=api_key, model=model)
                logger.info(f"OpenAI LLM classifier initialized with model: {model}")
                
            elif self.llm_type in ["huggingface", "hf"]:
                model_name = self.llm_config.get("model_name", "microsoft/DialoGPT-medium")
                self.llm_classifier = get_llm_classifier("huggingface", model_name=model_name)
                logger.info(f"Hugging Face LLM classifier initialized with model: {model_name}")
                
        except Exception as e:
            logger.error(f"Error loading LLM classifier: {e}")
            self.llm_classifier = None
    
    def _llm_classification(self, text: str) -> Optional[Dict[str, Any]]:
        """Classify using LLM if available."""
        if self.llm_classifier is None:
            return None
        
        try:
            # For LLM classification, we need to split the text back into title and description
            # This is a simplified approach - in practice, you might want to store these separately
            words = text.split()
            if len(words) <= 3:
                title = text
                description = ""
            else:
                # Assume first few words are title, rest is description
                title = " ".join(words[:3])
                description = " ".join(words[3:])
            
            result = self.llm_classifier.classify_product(title, description)
            return result
            
        except Exception as e:
            logger.error(f"LLM classification error: {e}")
            return None
    
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
            best_category = GOOGLE_TAXONOMY_CATEGORIES[best_idx]
            
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
        
        # Use Google Taxonomy categories for keyword matching
        for category in GOOGLE_TAXONOMY_CATEGORIES:
            # Simple keyword matching based on category name
            category_words = category.lower().split()
            score = sum(1 for word in category_words if word in text_lower)
            scores[category] = score
        
        best_category = max(scores, key=scores.get)
        if scores[best_category] == 0:
            return "Unknown"
        return best_category
    
    def classify_product(self, title: str, description: str, size: str = "", 
                        image_path: Optional[str] = None, use_llm: bool = True) -> Dict[str, Any]:
        """Classify a product using title and description.
        
        Args:
            title: Product title
            description: Product description
            size: Optional size information
            image_path: Optional path to product image (not used in current implementation)
            use_llm: Whether to use LLM classification (if available)
            
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
        
        # Try LLM classification first if enabled and available
        if use_llm and self.llm_classifier is not None:
            try:
                llm_result = self.llm_classifier.classify_product(title, description, size)
                if llm_result and llm_result.get("category") != "Unknown":
                    return llm_result
            except Exception as e:
                logger.warning(f"LLM classification failed, falling back to other methods: {e}")
        
        # Fallback to semantic classification
        category = self._semantic_classification(combined_text)
        
        # Determine method and confidence
        if self.model is not None:
            method = "semantic"
            confidence = "high"
        else:
            method = "keyword"
            confidence = "medium"
        
        return {
            "category": category,
            "confidence": confidence,
            "method": method,
            "input_text": combined_text,
            "llm_available": self.llm_classifier is not None
        }


# Global classifier instance
_classifier_instance = None

def get_classifier(llm_type: str = "openai", llm_config: Optional[Dict] = None) -> ProductClassifier:
    """Get or create a global classifier instance."""
    global _classifier_instance
    if _classifier_instance is None:
        _classifier_instance = ProductClassifier(llm_type=llm_type, llm_config=llm_config)
    return _classifier_instance


def identify_google_taxonomy(title: str, description: str, size: str = "", 
                           image_path: Optional[str] = None, use_llm: bool = True) -> str:
    """Legacy function for backward compatibility."""
    classifier = get_classifier()
    result = classifier.classify_product(title, description, size, image_path, use_llm)
    return result["category"]


def main(argv: Optional[List[str]] = None) -> None:
    """Command line interface for the product classifier."""
    parser = argparse.ArgumentParser(
        description="Classify retail products using LLMs and advanced NLP models"
    )
    parser.add_argument("--title", required=True, help="Product title")
    parser.add_argument("--description", required=True, help="Product description")
    parser.add_argument("--size", default="", help="Size description")
    parser.add_argument("--image", help="Path to product image")
    parser.add_argument("--llm-type", choices=["openai", "huggingface", "none"], 
                       default="openai", help="Type of LLM to use")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM classification")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args(argv)
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Configure LLM
        llm_config = {}
        if args.llm_type == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key:
                llm_config["api_key"] = api_key
                llm_config["model"] = "gpt-3.5-turbo"
        
        classifier = get_classifier(llm_type=args.llm_type, llm_config=llm_config)
        result = classifier.classify_product(
            args.title, args.description, args.size, args.image, 
            use_llm=not args.no_llm
        )
        
        print(f"Category: {result['category']}")
        print(f"Method: {result['method']}")
        print(f"Confidence: {result['confidence']}")
        if result.get("llm_available"):
            print(f"LLM Available: Yes")
        if result.get("llm_response"):
            print(f"LLM Response: {result['llm_response']}")
        
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
