import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import html
import json
import pickle
import logging
import warnings
import os

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

# Data and model paths
DASHBOARD_DATA_PATH = "data/steam_cleaned_data.csv"
PRICE_MODEL_PATH = "models/price_model.pkl"
PRICE_SCALER_PATH = "models/price_scaler.pkl"
PRICE_FEATURES_PATH = "models/price_features.pkl"
SENTIMENT_MODEL_PATH = "models/sentiment_model.pkl"
SENTIMENT_VECTORIZER_PATH = "models/sentiment_vectorizer.pkl"
METRICS_PATH = "models/metrics.json"

# Page configuration
st.set_page_config(
    page_title="Steam ML Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e88e5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background-color: #e3f2fd;
        padding: 2rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border: 2px solid #1e88e5;
        text-align: center;
    }
    .prediction-box * { color: #0d47a1; }
    .positive-sentiment * { color: #1b5e20; }
    .negative-sentiment * { color: #b71c1c; }
    .positive-sentiment {
        background-color: #c8e6c9;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4caf50;
    }
    .negative-sentiment {
        background-color: #ffcdd2;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_metrics():
    """Load the metrics written by train_all_models.py."""
    try:
        with open(METRICS_PATH, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return None

@st.cache_data
def load_dashboard_data():
    """Load dashboard data."""
    try:
        if not os.path.exists(DASHBOARD_DATA_PATH):
            return None
        df = pd.read_csv(DASHBOARD_DATA_PATH)
        return df
    except Exception as e:
        st.error(f"Error loading dashboard data: {str(e)}")
        return None

@st.cache_resource
def load_price_model():
    """Load price prediction model and components."""
    try:
        if not all([os.path.exists(PRICE_MODEL_PATH), 
                   os.path.exists(PRICE_SCALER_PATH),
                   os.path.exists(PRICE_FEATURES_PATH)]):
            return None, None, None
        
        with open(PRICE_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(PRICE_SCALER_PATH, 'rb') as f:
            scaler = pickle.load(f)
        with open(PRICE_FEATURES_PATH, 'rb') as f:
            features = pickle.load(f)
        return model, scaler, features
    except Exception as e:
        st.error(f"Error loading price model: {str(e)}")
        return None, None, None

@st.cache_resource
def load_sentiment_model():
    """Load sentiment classification model and vectorizer."""
    try:
        if not all([os.path.exists(SENTIMENT_MODEL_PATH),
                   os.path.exists(SENTIMENT_VECTORIZER_PATH)]):
            return None, None
        
        with open(SENTIMENT_MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(SENTIMENT_VECTORIZER_PATH, 'rb') as f:
            vectorizer = pickle.load(f)
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        return None, None

def main():
    """Main application function."""
    
    # Header
    st.markdown('<h1 class="main-header">🎮 Steam ML Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page:",
        ["📊 Dashboard", "💰 Price Predictor", "😊 Sentiment Classifier"]
    )
    
    st.sidebar.markdown("---")
    
    st.sidebar.markdown(
        """
        <div style="text-align: center; padding: 20px 0;">
            <div style="font-size: 24px; font-weight: bold; color: #1e88e5; margin-bottom: 10px;">
                STEAM
            </div>
            <div style="font-size: 14px; color: #666; margin-bottom: 15px;">
                Machine Learning Dashboard
            </div>
            <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                        width: 80px; height: 80px; border-radius: 50%; 
                        margin: 0 auto; display: flex; align-items: center; justify-content: center;
                        box-shadow: 0 4px 15px rgba(0,0,0,0.2);">
                <span style="font-size: 36px; color: white;">🎮</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Route to appropriate page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "💰 Price Predictor":
        show_price_predictor()
    elif page == "😊 Sentiment Classifier":
        show_sentiment_classifier()

def show_dashboard():
    """Display analytics dashboard."""
    
    st.markdown("## 📊 Steam Games Analytics Dashboard")
    
    # Load data
    df = load_dashboard_data()
    
    if df is None:
        st.error("⚠️ Dashboard data not found. Please ensure 'data/steam_cleaned_data.csv' exists.")
        st.info("""
        **Setup Instructions:**
        1. Run data analysis scripts to generate cleaned data
        2. Save the cleaned data as 'data/steam_cleaned_data.csv'
        3. Refresh this page
        """)
        return
    
    st.success(f"✅ Loaded {len(df):,} games")
    
    # Key Metrics
    st.markdown("### 📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_price = df['price_final'].mean()
        st.metric("Average Price", f"${avg_price:.2f}")
    
    with col2:
        total_reviews = df['total_reviews'].sum()
        st.metric("Total Reviews", f"{total_reviews:,.0f}")
    
    with col3:
        avg_rating = df['positive_ratio'].mean() * 100
        st.metric("Avg Rating", f"{avg_rating:.1f}%")
    
    with col4:
        total_games = len(df)
        st.metric("Total Games", f"{total_games:,}")
    
    # Visualizations
    st.markdown("### 📊 Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Price Distribution", "Genres Analysis", "Ratings"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            df[df['price_final'] > 0]['price_final'].hist(bins=50, edgecolor='black', ax=ax)
            ax.set_xlabel('Price ($)')
            ax.set_ylabel('Frequency')
            ax.set_title('Game Price Distribution')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Price by platform
            fig, ax = plt.subplots(figsize=(10, 6))
            if 'platform_count' in df.columns:
                df.boxplot(column='price_final', by='platform_count', ax=ax)
                ax.set_xlabel('Number of Platforms')
                ax.set_ylabel('Price ($)')
                ax.set_title('Price by Platform Availability')
                plt.suptitle('')
                st.pyplot(fig)
                plt.close()
    
    with tab2:
        if 'genre_count' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                # Genre count distribution
                fig, ax = plt.subplots(figsize=(10, 6))
                df['genre_count'].value_counts().sort_index().plot(kind='bar', ax=ax, color='steelblue')
                ax.set_xlabel('Number of Genres')
                ax.set_ylabel('Number of Games')
                ax.set_title('Games by Genre Count')
                st.pyplot(fig)
                plt.close()
            
            with col2:
                # Price by genre count
                fig, ax = plt.subplots(figsize=(10, 6))
                df.boxplot(column='price_final', by='genre_count', ax=ax)
                ax.set_xlabel('Number of Genres')
                ax.set_ylabel('Price ($)')
                ax.set_title('Price by Genre Count')
                plt.suptitle('')
                st.pyplot(fig)
                plt.close()
    
    with tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            # Rating distribution
            fig, ax = plt.subplots(figsize=(10, 6))
            df['positive_ratio'].hist(bins=50, edgecolor='black', color='green', alpha=0.7, ax=ax)
            ax.set_xlabel('Positive Review Ratio')
            ax.set_ylabel('Frequency')
            ax.set_title('Rating Distribution')
            st.pyplot(fig)
            plt.close()
        
        with col2:
            # Reviews vs Price scatter
            fig, ax = plt.subplots(figsize=(10, 6))
            sample = df.sample(min(1000, len(df)))
            ax.scatter(np.log1p(sample['total_reviews']), sample['price_final'], alpha=0.5)
            ax.set_xlabel('Log(Total Reviews + 1)')
            ax.set_ylabel('Price ($)')
            ax.set_title('Reviews vs Price')
            st.pyplot(fig)
            plt.close()
    
    # Top Games
    st.markdown("### 🏆 Top Rated Games")
    
    if 'name' in df.columns and 'positive_ratio' in df.columns:
        top_games = df.nlargest(10, 'positive_ratio')[['name', 'price_final', 'positive_ratio', 'total_reviews']]
        top_games['positive_ratio'] = (top_games['positive_ratio'] * 100).round(1)
        top_games.columns = ['Game Name', 'Price ($)', 'Rating (%)', 'Total Reviews']
        st.dataframe(top_games, width='stretch')

def show_price_predictor():
    """Display price prediction interface."""
    
    st.markdown("## 💰 Game Price Predictor")
    metrics = load_metrics()
    price_info = metrics["price"] if metrics else None
    price_model_name = price_info["best_model"] if price_info else "regression"
    st.markdown(f"Predict the price of a Steam game based on its features using the trained {price_model_name} model.")
    
    # Load model
    model, scaler, features = load_price_model()
    
    if model is None:
        st.error("⚠️ Price prediction model not found.")
        st.info("""
        **Setup Instructions:**
        1. Train the price prediction model using the provided scripts
        2. Ensure models are saved in 'models/' directory:
           - price_model.pkl
           - price_scaler.pkl
           - price_features.pkl
        3. Refresh this page
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    st.info(f"**Model:** {price_model_name} | **Features:** {len(features)}")
    if price_info:
        best = price_info["models"][price_model_name]
        st.caption(
            f"Trained on {price_info['n_games']} games. Cross-validated R² {best['cv_r2_mean']:.2f}, "
            f"average error about ${best['cv_mae_mean']:.2f}, so treat predictions as rough estimates."
        )
    
    st.markdown("---")
    st.markdown("### 🎮 Enter Game Features")
    st.caption("Defaults are typical values for a paid game in the dataset. The dataset holds popular games, so very low review counts tend to predict a free game.")
    
    col1, col2, col3 = st.columns(3)
    
    user_input = {}
    
    # Create input fields dynamically based on ALL available features
    with col1:
        if 'discount_percent' in features:
            user_input['discount_percent'] = st.slider("💸 Discount %", 0, 90, 0, 5)
        if 'release_year' in features:
            user_input['release_year'] = st.number_input("📅 Release Year", 1998, 2024, 2020)
        if 'game_age_years' in features:
            user_input['game_age_years'] = 2024 - user_input.get('release_year', 2020)
        if 'positive_ratio' in features:
            user_input['positive_ratio'] = st.slider("⭐ Positive Ratio", 0.0, 1.0, 0.89, 0.01)
        if 'review_score' in features:
            user_input['review_score'] = st.slider("📊 Review Score", 0, 9, 8)
        if 'genre_count' in features:
            user_input['genre_count'] = st.slider("🎭 Genres", 0, 8, 2)
    
    with col2:
        if 'platform_count' in features:
            user_input['platform_count'] = st.slider("🖥️ Platforms", 1, 3, 1)
        if 'achievements_total' in features:
            user_input['achievements_total'] = st.number_input("🏆 Achievements", 0, 1000, 50)
        if 'metacritic_score' in features:
            user_input['metacritic_score'] = st.slider("🎯 Metacritic", 60, 100, 84)
        if 'developer_count' in features:
            user_input['developer_count'] = st.slider("👨‍💻 Developers", 1, 8, 1)
        if 'release_month' in features:
            user_input['release_month'] = st.slider("📆 Release Month", 1, 12, 8)
        if 'total_negative' in features:
            user_input['total_negative'] = st.number_input("👎 Negative Reviews", 0, 1000000, 5000, 500)
    
    with col3:
        if 'category_count' in features:
            user_input['category_count'] = st.slider("📂 Categories", 1, 28, 11)
        if 'total_reviews' in features:
            user_input['total_reviews'] = st.number_input("💬 Total Reviews", 0, 100000, 1000, 100)
        if 'reviews_per_year' in features:
            user_input['reviews_per_year'] = st.number_input("📈 Reviews/Year", 0.0, 1000000.0, 17000.0, 1000.0)
        if 'has_dlc' in features:
            user_input['has_dlc'] = st.selectbox("🎁 Has DLC", [0, 1], index=1)
        if 'is_multiplatform' in features:
            user_input['is_multiplatform'] = st.selectbox("🌐 Multiplatform", [0, 1], index=0)
        if 'review_quality_score' in features:
            user_input['review_quality_score'] = st.slider("⭐ Review Quality", 0.0, 15.0, 9.9, 0.1)
        if 'has_achievements' in features:
            user_input['has_achievements'] = st.selectbox("🏅 Has Achievements", [0, 1], index=1)
    
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("💵 Predict Price", type="primary", width='stretch'):
            # Prepare input
            input_data = []
            for feature in features:
                input_data.append(user_input.get(feature, 0))
            
            input_array = np.array(input_data).reshape(1, -1)
            
            # Scale and predict
            input_scaled = scaler.transform(input_array)
            prediction = max(0.0, float(model.predict(input_scaled)[0]))
            
            # Display prediction
            # Price category
            if prediction < 10:
                category = "Budget Game"
                emoji = "💰"
            elif prediction < 30:
                category = "Mid-Range Game"
                emoji = "💵"
            else:
                category = "Premium Game"
                emoji = "💎"
            
            st.markdown(
                f"""<div class="prediction-box">
                <h3>Predicted Price</h3>
                <h1>&#36;{prediction:.2f}</h1>
                <h3>{emoji} {category}</h3>
                </div>""",
                unsafe_allow_html=True,
            )
            
            # Show feature values used
            with st.expander("📋 Input Features Used"):
                feature_df = pd.DataFrame([user_input]).T
                feature_df.columns = ['Value']
                st.dataframe(feature_df, width='stretch')

def show_sentiment_classifier():
    """Display sentiment classification interface."""
    
    st.markdown("## 😊 Review Sentiment Classifier")
    metrics = load_metrics()
    sentiment_info = metrics["sentiment"] if metrics else None
    sentiment_model_name = sentiment_info["best_model"] if sentiment_info else "text classification"
    st.markdown(f"Classify the sentiment of Steam game reviews using the trained {sentiment_model_name} model.")
    
    # Load model
    model, vectorizer = load_sentiment_model()
    
    if model is None:
        st.error("⚠️ Sentiment model not found.")
        st.info("""
        **Setup Instructions:**
        1. Train the sentiment model using the provided scripts
        2. Ensure models are saved in 'models/' directory:
           - sentiment_model.pkl
           - sentiment_vectorizer.pkl
        3. Refresh this page
        """)
        return
    
    st.success("✅ Model loaded successfully!")
    st.info(f"**Model:** {sentiment_model_name} with TF-IDF Vectorization")
    
    st.markdown("---")
    
    # Tabs for single and batch classification
    tab1, tab2 = st.tabs(["📝 Single Review", "📋 Batch Reviews"])
    
    with tab1:
        st.markdown("### Classify a Single Review")
        
        review_text = st.text_area(
            "Enter a game review:",
            "This game is absolutely amazing! Best game ever! The graphics are stunning and gameplay is so smooth.",
            height=150
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Classify Review", type="primary", width='stretch'):
                if review_text.strip():
                    # Vectorize
                    review_vec = vectorizer.transform([review_text])
                    
                    # Predict
                    prediction = model.predict(review_vec)[0]
                    
                    # Get probability if available
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(review_vec)[0]
                    elif hasattr(model, 'decision_function'):
                        # For LinearSVC, use decision function and normalize
                        decision = model.decision_function(review_vec)[0]
                        # Convert to probability-like score (0-1 range)
                        proba_pos = 1 / (1 + np.exp(-decision))
                        proba = np.array([1 - proba_pos, proba_pos])
                    else:
                        proba = None
                    
                    # Display result
                    positive = prediction == 1
                    confidence = proba[1 if positive else 0] if proba is not None else None
                    css_class = "positive-sentiment" if positive else "negative-sentiment"
                    label = "👍 POSITIVE Sentiment" if positive else "👎 NEGATIVE Sentiment"
                    confidence_html = f"<h4>Confidence: {confidence*100:.1f}%</h4>" if confidence is not None else ""
                    st.markdown(
                        f'<div class="{css_class}"><h3>{label}</h3>{confidence_html}</div>',
                        unsafe_allow_html=True,
                    )
                    if confidence is not None:
                        st.progress(float(confidence))
                    
                    st.markdown("---")
                    st.markdown("**Review Text:**")
                    st.info(review_text)
                else:
                    st.warning("Please enter a review text!")
    
    with tab2:
        st.markdown("### Classify Multiple Reviews")
        
        batch_reviews = st.text_area(
            "Enter multiple reviews (one per line):",
            """This game is absolutely amazing! Best purchase ever!
Terrible game, waste of money and time.
Pretty good game, has some bugs but enjoyable overall.
Worst game I've ever played, don't buy this trash.
10/10 would definitely recommend to my friends!
Boring gameplay, repetitive missions, not worth it.""",
            height=200
        )
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔮 Classify All Reviews", type="primary", width='stretch'):
                reviews = [r.strip() for r in batch_reviews.split("\n") if r.strip()]
                
                if reviews:
                    # Vectorize
                    reviews_vec = vectorizer.transform(reviews)
                    
                    # Predict
                    predictions = model.predict(reviews_vec)
                    
                    # Get probabilities if available
                    if hasattr(model, 'predict_proba'):
                        probas = model.predict_proba(reviews_vec)
                    elif hasattr(model, 'decision_function'):
                        # For LinearSVC, use decision function and normalize
                        decisions = model.decision_function(reviews_vec)
                        # Convert to probability-like scores
                        probas_pos = 1 / (1 + np.exp(-decisions))
                        probas = np.column_stack([1 - probas_pos, probas_pos])
                    else:
                        probas = None
                    
                    # Display results
                    st.markdown("### 📊 Classification Results:")
                    
                    # Summary
                    positive_count = int(sum(predictions))
                    negative_count = len(predictions) - positive_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Reviews", len(reviews))
                    with col2:
                        st.metric("👍 Positive", positive_count)
                    with col3:
                        st.metric("👎 Negative", negative_count)
                    
                    st.markdown("---")
                    
                    # Individual results
                    for i, (review, pred) in enumerate(zip(reviews, predictions), 1):
                        sentiment = "👍 POSITIVE" if pred == 1 else "👎 NEGATIVE"
                        color = "positive-sentiment" if pred == 1 else "negative-sentiment"
                        
                        if probas is not None:
                            confidence = probas[i-1][pred] * 100
                            title = f"Review {i}: {sentiment} ({confidence:.1f}% confidence)"
                        else:
                            title = f"Review {i}: {sentiment}"
                        
                        st.markdown(
                            f'<div class="{color}"><strong>{title}</strong><br><em>{html.escape(review)}</em></div>',
                            unsafe_allow_html=True,
                        )
                else:
                    st.warning("Please enter at least one review!")

if __name__ == "__main__":
    main()