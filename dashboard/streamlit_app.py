import streamlit as st
import pandas as pd
import requests

API_URL = "https://recommender-api-i3ta.onrender.com"

st.set_page_config(
    page_title="Recommendation Engine Dashboard",
    page_icon="🛒",
    layout="wide"
)

st.title("🛒 Recommendation Engine Dashboard")

st.markdown("""
Interactive dashboard for the deployed Recommendation Engine API.
""")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Recommendation Demo", "Model Info", "API Stats"]
)

if page == "Overview":
    st.header("Project Overview")

    st.markdown("""
    This project is a deployed recommendation system that uses:

    - SVD Collaborative Filtering
    - Popularity-Based Fallback
    - Cold-Start Handling
    - Diversity-Aware Ranking
    - FastAPI Backend
    """)

    try:
        response = requests.get(f"{API_URL}/health")

        if response.status_code == 200:
            st.success("API is live and healthy")
        else:
            st.error("API is not responding properly")

    except:
        st.error("Could not connect to API")


elif page == "Recommendation Demo":
    st.header("Generate Recommendations")

    customer_id = st.number_input(
        "Customer ID",
        min_value=1,
        value=12347
    )

    n = st.slider(
        "Number of Recommendations",
        min_value=1,
        max_value=50,
        value=10
    )

    diversity_weight = st.slider(
        "Diversity Weight",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )

    if st.button("Generate Recommendations"):
        payload = {
            "customer_id": int(customer_id),
            "n": int(n),
            "diversity_weight": float(diversity_weight)
        }

        try:
            response = requests.post(
                f"{API_URL}/recommend",
                json=payload
            )

            if response.status_code == 200:
                data = response.json()

                st.success(f"Strategy Used: {data['strategy']}")

                recs = pd.DataFrame(data["recommendations"])

                st.dataframe(recs, use_container_width=True)

            else:
                st.error("Failed to generate recommendations")
                st.write(response.text)

        except:
            st.error("Could not connect to API")


elif page == "Model Info":
    st.header("Model Information")

    try:
        response = requests.get(f"{API_URL}/model-info")

        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error("Could not fetch model info")

    except:
        st.error("Could not connect to API")


elif page == "API Stats":
    st.header("API Statistics")

    try:
        response = requests.get(f"{API_URL}/stats")

        if response.status_code == 200:
            stats = response.json()

            col1, col2, col3, col4 = st.columns(4)

            col1.metric("Total Users", stats["total_users"])
            col2.metric("Total Items", stats["total_items"])
            col3.metric("Cold Users", stats["cold_users"])
            col4.metric("Warm Users", stats["warm_users"])

            st.subheader("Raw Stats")
            st.json(stats)

        else:
            st.error("Could not fetch stats")

    except:
        st.error("Could not connect to API")