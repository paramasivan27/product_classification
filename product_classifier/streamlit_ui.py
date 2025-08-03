import streamlit as st
import pandas as pd

try:
    from .classifier import get_classifier
    from .taxonomy import PRODUCT_CATEGORIES
except ImportError:  # pragma: no cover - fallback when run as a script
    try:
        from product_classifier.classifier import get_classifier
        from product_classifier.taxonomy import PRODUCT_CATEGORIES
    except ImportError:
        from classifier import get_classifier
        from taxonomy import PRODUCT_CATEGORIES


def main() -> None:
    st.set_page_config(
        page_title="Product Classification System",
        page_icon="🛍️",
        layout="wide"
    )
    
    st.title("🛍️ Product Classification System")
    st.markdown("---")
    
    # Sidebar for additional options
    with st.sidebar:
        st.header("Settings")
        show_categories = st.checkbox("Show Available Categories", value=False)
        
        if show_categories:
            st.subheader("Available Categories")
            for category in PRODUCT_CATEGORIES.keys():
                st.write(f"• {category}")
    
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
            
            submitted = st.form_submit_button("Classify Product", type="primary")
    
    with col2:
        st.subheader("Quick Examples")
        
        examples = [
            {
                "title": "Nike Air Max Running Shoes",
                "description": "Comfortable athletic shoes for running and training",
                "size": "US 10"
            },
            {
                "title": "iPhone 15 Pro Max",
                "description": "Latest smartphone with advanced camera and A17 Pro chip",
                "size": "256GB"
            },
            {
                "title": "Organic Cotton T-Shirt",
                "description": "Comfortable casual wear made from organic cotton",
                "size": "Large"
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
                with st.spinner("Analyzing product..."):
                    classifier = get_classifier()
                    result = classifier.classify_product(title, description, size)
                
                # Display results
                st.markdown("---")
                st.subheader("Classification Results")
                
                # Create a nice results display
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Category",
                        value=result["category"],
                        delta=None
                    )
                
                with col2:
                    confidence_color = "🟢" if result["confidence"] == "high" else "🟡"
                    st.metric(
                        label="Confidence",
                        value=f"{confidence_color} {result['confidence'].title()}"
                    )
                
                with col3:
                    method_icon = "🧠" if result["method"] == "semantic" else "🔍"
                    st.metric(
                        label="Method",
                        value=f"{method_icon} {result['method'].title()}"
                    )
                
                # Show input text used for classification
                with st.expander("Analysis Details"):
                    st.write(f"**Input Text:** {result['input_text']}")
                    st.write(f"**Classification Method:** {result['method']}")
                    st.write(f"**Confidence Level:** {result['confidence']}")
                
                # Show category description
                if result["category"] in PRODUCT_CATEGORIES:
                    st.info(f"**Category Description:** {result['category']} includes products like: {', '.join(PRODUCT_CATEGORIES[result['category']][:10])}...")
                
            except Exception as exc:
                st.error(f"❌ Error during classification: {exc}")
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
        <p>Powered by Advanced NLP Models | Built with Streamlit</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
