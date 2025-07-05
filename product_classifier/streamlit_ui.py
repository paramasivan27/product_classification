import streamlit as st

from .classifier import identify_google_taxonomy


def main() -> None:
    st.title("Product Classification")
    st.write("Enter product details to classify them using the Google product taxonomy.")

    title = st.text_input("Product Title")
    description = st.text_area("Description")
    size = st.text_input("Size", "")

    if st.button("Classify"):
        if not title or not description:
            st.warning("Please provide both a title and description.")
        else:
            try:
                category = identify_google_taxonomy(title, description, size)
                st.success(f"Category: {category}")
            except Exception as exc:
                st.error(f"Error: {exc}")


if __name__ == "__main__":  # pragma: no cover
    main()
