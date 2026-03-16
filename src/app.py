import streamlit as st
import requests

st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.write(
    "Enter a movie review and the ML model will predict if it's positive or negative."
)

review = st.text_area(
    "Your review:", height=150, placeholder="Type or paste a movie review here..."
)

if st.button("Analyze Sentiment", type="primary"):
    if review.strip():
        with st.spinner("Analyzing..."):
            try:
                response = requests.post(
                    "http://localhost:8000/predict", json={"text": review}
                )
                result = response.json()

                col1, col2 = st.columns(2)
                with col1:
                    if result["sentiment"] == "positive":
                        st.success(f"**Sentiment:** {result['sentiment'].upper()}")
                    else:
                        st.error(f"**Sentiment:** {result['sentiment'].upper()}")
                with col2:
                    st.metric("Confidence", f"{result['confidence'] * 100:.1f}%")

            except requests.exceptions.ConnectionError:
                st.error(
                    "Cannot connect to API. Make sure the FastAPI server is running on port 8000."
                )
    else:
        st.warning("Please enter a review first.")
