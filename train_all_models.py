"""Train the price and sentiment models and save what the Streamlit app needs.

Usage (from the repository root):
    python train_all_models.py
"""
import json
import os
import pickle

import matplotlib

matplotlib.use("Agg")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, KFold, RepeatedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.model_training import ModelComparator, SteamPricePredictor
from src.nlp_sentiment_classifier import SteamSentimentClassifier

PRICE_DATA = "data/steam_cleaned_data.csv"
REVIEW_DATA = "data/steam_reviews_cleaned_final.csv"

os.makedirs("models", exist_ok=True)


def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def rounded(value):
    return None if value is None else round(float(value), 4)


# ---------------------------------------------------------------- price model
print("Training price prediction model...")
predictor = SteamPricePredictor(PRICE_DATA)
X, y = predictor.prepare_features()
predictor.split_data(X, y)

comparator = ModelComparator(
    predictor.X_train,
    predictor.X_test,
    predictor.y_train,
    predictor.y_test,
    predictor.feature_names,
)
comparator.initialize_models()
comparator.scale_features()
comparator.train_and_evaluate()

price_name, price_model = comparator.get_best_model()

save_pickle(price_model, "models/price_model.pkl")
save_pickle(comparator.scaler, "models/price_scaler.pkl")
save_pickle(predictor.feature_names, "models/price_features.pkl")

# A single 80/20 split of 232 games is very noisy, so also report 5-fold CV
# repeated 10 times on the whole dataset, for every candidate model.
price_cv = {}
for name in comparator.results:
    pipeline = make_pipeline(StandardScaler(), clone(comparator.models[name]))
    scores = cross_val_score(
        pipeline, X, y, scoring="r2", n_jobs=-1,
        cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=42),
    )
    mae = -cross_val_score(
        pipeline, X, y, scoring="neg_mean_absolute_error", n_jobs=-1,
        cv=RepeatedKFold(n_splits=5, n_repeats=10, random_state=42),
    )
    price_cv[name] = {
        "cv_r2_mean": rounded(scores.mean()),
        "cv_r2_std": rounded(scores.std()),
        "cv_mae_mean": rounded(mae.mean()),
    }

# ------------------------------------------------------------ sentiment model
print("Training sentiment model...")
classifier = SteamSentimentClassifier(REVIEW_DATA)
classifier.prepare_data()
classifier.create_vectorizer()
classifier.initialize_models()
classifier.train_and_evaluate()

sentiment_name, sentiment_model = classifier.get_best_model()

save_pickle(sentiment_model, "models/sentiment_model.pkl")
save_pickle(classifier.vectorizer, "models/sentiment_vectorizer.pkl")

# Random review-level CV versus CV that keeps each game in one fold. The second
# one shows how the model does on reviews of games it never saw during training.
reviews = classifier.df
sentiment_cv = {}
for name in classifier.results:
    pipeline = make_pipeline(clone(classifier.vectorizer), clone(classifier.models[name]))
    random_f1 = cross_val_score(
        pipeline, reviews["cleaned_text"], reviews["sentiment"], scoring="f1", n_jobs=-1,
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
    )
    unseen_game_f1 = cross_val_score(
        pipeline, reviews["cleaned_text"], reviews["sentiment"], scoring="f1", n_jobs=-1,
        groups=reviews["app_id"], cv=GroupKFold(n_splits=5),
    )
    sentiment_cv[name] = {
        "cv_f1_mean": rounded(random_f1.mean()),
        "cv_f1_std": rounded(random_f1.std()),
        "unseen_game_cv_f1_mean": rounded(unseen_game_f1.mean()),
    }

# -------------------------------------------------------------------- metrics
metrics = {
    "price": {
        "best_model": price_name,
        "n_games": int(len(X)),
        "n_features": len(predictor.feature_names),
        "models": {
            name: {
                "test_r2": rounded(r["Test R²"]),
                "train_r2": rounded(r["Train R²"]),
                "test_rmse": rounded(r["RMSE"]),
                "test_mae": rounded(r["MAE"]),
                **price_cv[name],
            }
            for name, r in comparator.results.items()
        },
    },
    "sentiment": {
        "best_model": sentiment_name,
        "n_reviews": int(len(reviews)),
        "models": {
            name: {
                "test_accuracy": rounded(r["Test Accuracy"]),
                "train_accuracy": rounded(r["Train Accuracy"]),
                "test_precision": rounded(r["Precision"]),
                "test_recall": rounded(r["Recall"]),
                "test_f1": rounded(r["F1-Score"]),
                "test_roc_auc": rounded(r["ROC-AUC"]),
                **sentiment_cv[name],
            }
            for name, r in classifier.results.items()
        },
    },
}

with open("models/metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

print("All models trained and saved.")
