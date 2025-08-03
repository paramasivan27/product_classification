"""
LLM-based Google Product Taxonomy Classification
"""

import os
import json
import logging
from typing import Dict, List, Optional, Any
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# Google Product Taxonomy categories (simplified version for LLM classification)
GOOGLE_TAXONOMY_CATEGORIES = [
    "Animals & Pet Supplies",
    "Apparel & Accessories",
    "Arts & Entertainment",
    "Baby & Toddler",
    "Beauty & Personal Care",
    "Books & Magazines",
    "Business & Industrial",
    "Cameras & Optics",
    "Clothing, Shoes & Jewelry",
    "Computers & Electronics",
    "Food, Beverages & Tobacco",
    "Furniture",
    "Hardware",
    "Health & Beauty",
    "Home & Garden",
    "Luggage & Bags",
    "Mature",
    "Media",
    "Office Products",
    "Religious & Ceremonial",
    "Software",
    "Sporting Goods",
    "Toys & Games",
    "Vehicles & Parts"
]


class LLMTaxonomyClassifier(ABC):
    """Abstract base class for LLM-based taxonomy classification."""
    
    @abstractmethod
    def classify_product(self, title: str, description: str, size: str = "") -> Dict[str, Any]:
        """Classify a product using LLM."""
        pass


class OpenAITaxonomyClassifier(LLMTaxonomyClassifier):
    """OpenAI-based Google Product Taxonomy classifier."""
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-3.5-turbo"):
        """Initialize OpenAI classifier.
        
        Args:
            api_key: OpenAI API key (will try to get from environment if not provided)
            model: OpenAI model to use
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable or pass api_key parameter.")
        
        try:
            import openai
            self.client = openai.OpenAI(api_key=self.api_key)
            logger.info(f"OpenAI client initialized with model: {model}")
        except ImportError:
            raise ImportError("openai library is required. Install with: pip install openai")
    
    def classify_product(self, title: str, description: str, size: str = "") -> Dict[str, Any]:
        """Classify a product using OpenAI."""
        try:
            # Prepare the prompt
            prompt = self._create_classification_prompt(title, description, size)
            
            # Make API call
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a product classification expert. Your task is to classify products into the most appropriate Google Product Taxonomy category based on the product title and description."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent classification
                max_tokens=100
            )
            
            # Parse response
            result = self._parse_openai_response(response.choices[0].message.content)
            
            return {
                "category": result["category"],
                "confidence": result["confidence"],
                "method": "openai_llm",
                "input_text": f"{title} {description} {size}".strip(),
                "llm_response": response.choices[0].message.content,
                "model_used": self.model
            }
            
        except Exception as e:
            logger.error(f"OpenAI classification error: {e}")
            raise
    
    def _create_classification_prompt(self, title: str, description: str, size: str) -> str:
        """Create the classification prompt for OpenAI."""
        combined_text = f"{title} {description} {size}".strip()
        
        prompt = f"""
Please classify the following product into the most appropriate Google Product Taxonomy category:

Product: {combined_text}

Available categories:
{chr(10).join(f"- {category}" for category in GOOGLE_TAXONOMY_CATEGORIES)}

Please respond in the following JSON format:
{{
    "category": "exact_category_name_from_list",
    "confidence": "high/medium/low",
    "reasoning": "brief explanation of why this category was chosen"
}}

Only use categories from the provided list. If the product doesn't clearly fit any category, choose the closest match and set confidence to "low".
"""
        return prompt
    
    def _parse_openai_response(self, response: str) -> Dict[str, str]:
        """Parse OpenAI response to extract category and confidence."""
        try:
            # Try to extract JSON from response
            if "{" in response and "}" in response:
                start = response.find("{")
                end = response.rfind("}") + 1
                json_str = response[start:end]
                result = json.loads(json_str)
                
                category = result.get("category", "Unknown")
                confidence = result.get("confidence", "medium")
                
                # Validate category is in our list
                if category not in GOOGLE_TAXONOMY_CATEGORIES:
                    category = "Unknown"
                    confidence = "low"
                
                return {
                    "category": category,
                    "confidence": confidence
                }
            else:
                # Fallback parsing
                lines = response.strip().split('\n')
                category = "Unknown"
                confidence = "low"
                
                for line in lines:
                    line = line.strip().lower()
                    if "category:" in line:
                        category = line.split("category:")[-1].strip()
                    elif "confidence:" in line:
                        confidence = line.split("confidence:")[-1].strip()
                
                return {
                    "category": category.title() if category != "unknown" else "Unknown",
                    "confidence": confidence
                }
                
        except Exception as e:
            logger.error(f"Error parsing OpenAI response: {e}")
            return {
                "category": "Unknown",
                "confidence": "low"
            }


class HuggingFaceTaxonomyClassifier(LLMTaxonomyClassifier):
    """Hugging Face-based Google Product Taxonomy classifier."""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        """Initialize Hugging Face classifier.
        
        Args:
            model_name: Hugging Face model to use
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the Hugging Face model."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            
            logger.info(f"Loading Hugging Face model: {self.model_name}")
            
            # Use a more appropriate model for classification
            if "gpt" in self.model_name.lower() or "dialo" in self.model_name.lower():
                # For better classification, use a model that can handle structured output
                self.model_name = "microsoft/DialoGPT-medium"
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Add padding token if not present
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            logger.info("Hugging Face model loaded successfully")
            
        except ImportError:
            logger.warning("transformers library not available")
            self.model = None
        except Exception as e:
            logger.error(f"Error loading Hugging Face model: {e}")
            self.model = None
    
    def classify_product(self, title: str, description: str, size: str = "") -> Dict[str, Any]:
        """Classify a product using Hugging Face model."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Hugging Face model not loaded")
        
        try:
            # Create prompt
            prompt = self._create_classification_prompt(title, description, size)
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt", max_length=512, truncation=True)
            
            # Generate response
            import torch
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse response
            result = self._parse_hf_response(response, prompt)
            
            return {
                "category": result["category"],
                "confidence": result["confidence"],
                "method": "huggingface_llm",
                "input_text": f"{title} {description} {size}".strip(),
                "llm_response": response,
                "model_used": self.model_name
            }
            
        except Exception as e:
            logger.error(f"Hugging Face classification error: {e}")
            raise
    
    def _create_classification_prompt(self, title: str, description: str, size: str) -> str:
        """Create the classification prompt for Hugging Face."""
        combined_text = f"{title} {description} {size}".strip()
        
        prompt = f"""
Product Classification Task:
Product: {combined_text}

Available Google Product Taxonomy categories:
{chr(10).join(f"- {category}" for category in GOOGLE_TAXONOMY_CATEGORIES)}

Please classify this product into the most appropriate category from the list above.
Response format: Category: [category_name], Confidence: [high/medium/low]

Category:"""
        return prompt
    
    def _parse_hf_response(self, response: str, prompt: str) -> Dict[str, str]:
        """Parse Hugging Face response to extract category and confidence."""
        try:
            # Remove the prompt from response
            if prompt in response:
                response = response.replace(prompt, "").strip()
            
            # Parse the response
            category = "Unknown"
            confidence = "low"
            
            lines = response.strip().split('\n')
            for line in lines:
                line = line.strip().lower()
                if "category:" in line:
                    category = line.split("category:")[-1].strip()
                elif "confidence:" in line:
                    confidence = line.split("confidence:")[-1].strip()
            
            # Validate category
            if category not in [cat.lower() for cat in GOOGLE_TAXONOMY_CATEGORIES]:
                category = "Unknown"
                confidence = "low"
            
            return {
                "category": category.title() if category != "unknown" else "Unknown",
                "confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Error parsing Hugging Face response: {e}")
            return {
                "category": "Unknown",
                "confidence": "low"
            }


def get_llm_classifier(llm_type: str = "openai", **kwargs) -> LLMTaxonomyClassifier:
    """Factory function to get the appropriate LLM classifier.
    
    Args:
        llm_type: Type of LLM to use ("openai" or "huggingface")
        **kwargs: Additional arguments for the specific classifier
    
    Returns:
        LLMTaxonomyClassifier instance
    """
    if llm_type.lower() == "openai":
        return OpenAITaxonomyClassifier(**kwargs)
    elif llm_type.lower() in ["huggingface", "hf"]:
        return HuggingFaceTaxonomyClassifier(**kwargs)
    else:
        raise ValueError(f"Unsupported LLM type: {llm_type}. Use 'openai' or 'huggingface'") 