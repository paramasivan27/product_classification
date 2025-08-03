import streamlit as st
import pandas as pd
import os

try:
    from .classifier import get_classifier
    from .llm_taxonomy import GOOGLE_TAXONOMY_CATEGORIES
except ImportError:  # pragma: no cover - fallback when run as a script
    try:
        from product_classifier.classifier import get_classifier
        from product_classifier.llm_taxonomy import GOOGLE_TAXONOMY_CATEGORIES
    except ImportError:
        from classifier import get_classifier
        from llm_taxonomy import GOOGLE_TAXONOMY_CATEGORIES


def main() -> None:
    st.set_page_config(
        page_title="LLM Product Classification System",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 LLM Product Classification System")
    st.markdown("---")
    
    # Sidebar for LLM configuration
    with st.sidebar:
        st.header("🤖 LLM Configuration")
        
        llm_type = st.selectbox(
            "LLM Type",
            ["openai", "huggingface", "none"],
            help="Choose the LLM provider for classification"
        )
        
        if llm_type == "openai":
            st.subheader("OpenAI Configuration")
            api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key (or set OPENAI_API_KEY environment variable)"
            )
            
            if not api_key and not os.getenv("OPENAI_API_KEY"):
                st.warning("⚠️ OpenAI API key required for LLM classification")
            
            model = st.selectbox(
                "OpenAI Model",
                ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"],
                help="Choose the OpenAI model to use"
            )
            
        elif llm_type == "huggingface":
            st.subheader("Hugging Face Configuration")
            model_name = st.selectbox(
                "Hugging Face Model",
                [
                    "microsoft/DialoGPT-medium",
                    "gpt2",
                    "distilgpt2"
                ],
                help="Choose the Hugging Face model to use"
            )
        
        st.markdown("---")
        st.header("📋 Available Categories")
        show_categories = st.checkbox("Show Google Product Taxonomy", value=False)
        
        if show_categories:
            st.subheader("Google Product Taxonomy")
            for i, category in enumerate(GOOGLE_TAXONOMY_CATEGORIES, 1):
                st.write(f"{i}. {category}")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Product Information")
        
        # Product input form
        with st.form("product_form"):
            title = st.text_input(
                "Product Title *",
                placeholder="e.g., Men's Running Shoes, iPhone 15 Pro, etc."
            )
            
            description = st.text_area(
                "Product Description *",
                placeholder="Describe the product features, materials, use cases, etc.",
                height=120
            )
            
            size = st.text_input(
                "Size/Additional Info",
                placeholder="e.g., Large, 10.5, Red, etc."
            )
            
            use_llm = st.checkbox(
                "Use LLM Classification",
                value=True,
                help="Enable LLM-based classification (if available)"
            )
            
            submitted = st.form_submit_button("Classify Product", type="primary")
    
    with col2:
        st.subheader("Quick Examples")
        
        examples = [
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
            }
        ]
        
        for i, example in enumerate(examples):
            with st.expander(f"Example {i+1}"):
                st.write(f"**Title:** {example['title']}")
                st.write(f"**Description:** {example['description']}")
                st.write(f"**Size:** {example['size']}")
                
                if st.button(f"Use Example {i+1}", key=f"example_{i}"):
                    st.session_state.title = example['title']
                    st.session_state.description = example['description']
                    st.session_state.size = example['size']
                    st.rerun()
    
    # Handle form submission
    if submitted:
        if not title or not description:
            st.error("⚠️ Please provide both a title and description.")
        else:
            try:
                with st.spinner("🤖 Analyzing product with LLM..."):
                    # Configure LLM settings
                    llm_config = {}
                    if llm_type == "openai":
                        api_key_to_use = api_key or os.getenv("OPENAI_API_KEY")
                        if api_key_to_use:
                            llm_config["api_key"] = api_key_to_use
                            llm_config["model"] = model
                        else:
                            st.warning("⚠️ OpenAI API key not found. Falling back to other methods.")
                            llm_type = "none"
                    elif llm_type == "huggingface":
                        llm_config["model_name"] = model_name
                    
                    classifier = get_classifier(llm_type=llm_type, llm_config=llm_config)
                    result = classifier.classify_product(
                        title, description, size, use_llm=use_llm
                    )
                
                # Display results
                st.markdown("---")
                st.subheader("🎯 Classification Results")
                
                # Create a nice results display
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        label="Category",
                        value=result["category"],
                        delta=None
                    )
                
                with col2:
                    confidence_color = "🟢" if result["confidence"] == "high" else "🟡" if result["confidence"] == "medium" else "🔴"
                    st.metric(
                        label="Confidence",
                        value=f"{confidence_color} {result['confidence'].title()}"
                    )
                
                with col3:
                    method_icon = "🤖" if "llm" in result["method"] else "🧠" if result["method"] == "semantic" else "🔍"
                    st.metric(
                        label="Method",
                        value=f"{method_icon} {result['method'].title()}"
                    )
                
                with col4:
                    llm_status = "✅" if result.get("llm_available") else "❌"
                    st.metric(
                        label="LLM Available",
                        value=f"{llm_status} {'Yes' if result.get('llm_available') else 'No'}"
                    )
                
                # Show detailed information
                with st.expander("🔍 Analysis Details"):
                    st.write(f"**Input Text:** {result['input_text']}")
                    st.write(f"**Classification Method:** {result['method']}")
                    st.write(f"**Confidence Level:** {result['confidence']}")
                    
                    if result.get("llm_response"):
                        st.write(f"**LLM Response:** {result['llm_response']}")
                    
                    if result.get("model_used"):
                        st.write(f"**Model Used:** {result['model_used']}")
                
                # Show category information
                if result["category"] in GOOGLE_TAXONOMY_CATEGORIES:
                    st.info(f"**📂 Google Product Taxonomy Category:** {result['category']}")
                
            except Exception as exc:
                st.error(f"❌ Error during classification: {exc}")
                st.info("💡 Try disabling LLM classification or check your API keys.")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Powered by LLMs & Advanced NLP Models | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
