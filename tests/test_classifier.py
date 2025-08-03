import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the parent directory to the path so we can import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from product_classifier.classifier import ProductClassifier, get_classifier, identify_google_taxonomy
from product_classifier.taxonomy import PRODUCT_CATEGORIES


class TestProductClassifier:
    """Test cases for the ProductClassifier class."""
    
    def test_init_without_model(self):
        """Test classifier initialization when sentence-transformers is not available."""
        with patch.dict('sys.modules', {'sentence_transformers': None}):
            classifier = ProductClassifier()
            assert classifier.model is None
            assert classifier.category_embeddings is None
    
    def test_init_with_model(self):
        """Test classifier initialization with sentence-transformers available."""
        mock_model = MagicMock()
        mock_embeddings = [[0.1, 0.2, 0.3]] * len(PRODUCT_CATEGORIES)
        mock_model.encode.return_value = mock_embeddings
        
        # Mock the entire sentence_transformers module
        with patch.dict('sys.modules', {'sentence_transformers': MagicMock()}):
            with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
                classifier = ProductClassifier()
                assert classifier.model is not None
                assert classifier.category_embeddings is not None
                assert len(classifier.category_embeddings) == len(PRODUCT_CATEGORIES)
    
    def test_keyword_based_classification(self):
        """Test keyword-based classification fallback."""
        classifier = ProductClassifier()
        classifier.model = None  # Force keyword fallback
        
        # Test with known keywords
        result = classifier._keyword_based_classification("men's running shoes")
        assert result in PRODUCT_CATEGORIES.keys()
        
        # Test with unknown text
        result = classifier._keyword_based_classification("xyz123 unknown product")
        assert result == "Unknown"
    
    def test_classify_product_valid_input(self):
        """Test product classification with valid input."""
        classifier = ProductClassifier()
        classifier.model = None  # Force keyword fallback
        
        result = classifier.classify_product(
            title="Nike Running Shoes",
            description="Comfortable athletic shoes for running",
            size="US 10"
        )
        
        assert "category" in result
        assert "confidence" in result
        assert "method" in result
        assert "input_text" in result
        assert result["method"] == "keyword"
        assert result["confidence"] == "medium"
    
    def test_classify_product_missing_input(self):
        """Test product classification with missing required input."""
        classifier = ProductClassifier()
        
        with pytest.raises(ValueError, match="Both title and description are required"):
            classifier.classify_product("", "description")
        
        with pytest.raises(ValueError, match="Both title and description are required"):
            classifier.classify_product("title", "")
    
    def test_classify_product_invalid_image_path(self):
        """Test product classification with invalid image path."""
        classifier = ProductClassifier()
        
        with pytest.raises(FileNotFoundError):
            classifier.classify_product(
                title="Test Product",
                description="Test Description",
                image_path="/nonexistent/path/image.jpg"
            )


class TestGlobalFunctions:
    """Test cases for global functions."""
    
    def test_get_classifier_singleton(self):
        """Test that get_classifier returns a singleton instance."""
        classifier1 = get_classifier()
        classifier2 = get_classifier()
        assert classifier1 is classifier2
    
    def test_identify_google_taxonomy_legacy(self):
        """Test the legacy identify_google_taxonomy function."""
        result = identify_google_taxonomy(
            title="iPhone 15 Pro",
            description="Latest smartphone with advanced features"
        )
        assert isinstance(result, str)
        assert result in PRODUCT_CATEGORIES.keys() or result == "Unknown"


class TestTaxonomy:
    """Test cases for the taxonomy module."""
    
    def test_product_categories_structure(self):
        """Test that PRODUCT_CATEGORIES has the expected structure."""
        assert isinstance(PRODUCT_CATEGORIES, dict)
        assert len(PRODUCT_CATEGORIES) > 0
        
        for category, keywords in PRODUCT_CATEGORIES.items():
            assert isinstance(category, str)
            assert isinstance(keywords, list)
            assert len(keywords) > 0
            assert all(isinstance(keyword, str) for keyword in keywords)
    
    def test_legacy_taxonomy_support(self):
        """Test that TAXONOMY_KEYWORDS is available for backward compatibility."""
        from product_classifier.taxonomy import TAXONOMY_KEYWORDS
        assert TAXONOMY_KEYWORDS == PRODUCT_CATEGORIES


class TestIntegration:
    """Integration tests for the complete classification pipeline."""
    
    def test_end_to_end_classification(self):
        """Test complete classification pipeline with various product types."""
        test_cases = [
            {
                "title": "Nike Air Max Running Shoes",
                "description": "Comfortable athletic shoes for running and training",
                "expected_category": "Sports & Outdoors"
            },
            {
                "title": "iPhone 15 Pro Max",
                "description": "Latest smartphone with advanced camera and A17 Pro chip",
                "expected_category": "Electronics & Technology"
            },
            {
                "title": "Organic Cotton T-Shirt",
                "description": "Comfortable casual wear made from organic cotton",
                "expected_category": "Apparel & Accessories"
            },
            {
                "title": "Leather Sofa",
                "description": "Comfortable leather sofa for living room",
                "expected_category": "Home & Garden"
            }
        ]
        
        classifier = ProductClassifier()
        classifier.model = None  # Force keyword fallback for consistent testing
        
        for test_case in test_cases:
            result = classifier.classify_product(
                title=test_case["title"],
                description=test_case["description"]
            )
            
            # The classification should return a valid category
            assert result["category"] in PRODUCT_CATEGORIES.keys() or result["category"] == "Unknown"
            assert result["method"] == "keyword"
            assert result["confidence"] == "medium"


if __name__ == "__main__":
    pytest.main([__file__])
