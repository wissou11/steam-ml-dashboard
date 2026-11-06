import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class SteamSentimentClassifier:
    """Complete NLP pipeline for Steam review sentiment classification."""
    
    def __init__(self, csv_path):
        """Initialize with cleaned reviews CSV."""
        self.df = pd.read_csv(csv_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.vectorizer = None
        self.models = {}
        self.results = {}
        
        print("=" * 80)
        print("STEAM SENTIMENT CLASSIFIER - INITIALIZATION")
        print("=" * 80)
        print(f"\n✓ Loaded {len(self.df):,} reviews")
        print(f"✓ Positive: {(self.df['sentiment'] == 1).sum():,} ({(self.df['sentiment'] == 1).mean()*100:.1f}%)")
        print(f"✓ Negative: {(self.df['sentiment'] == 0).sum():,} ({(self.df['sentiment'] == 0).mean()*100:.1f}%)")
    
    def prepare_data(self, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        print("\n" + "=" * 80)
        print("PREPARING DATA")
        print("=" * 80)
        
        X = self.df['cleaned_text']
        y = self.df['sentiment']
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\n✓ Train set: {len(self.X_train):,} reviews")
        print(f"  - Positive: {(self.y_train == 1).sum():,} ({(self.y_train == 1).mean()*100:.1f}%)")
        print(f"  - Negative: {(self.y_train == 0).sum():,} ({(self.y_train == 0).mean()*100:.1f}%)")
        
        print(f"\n✓ Test set: {len(self.X_test):,} reviews")
        print(f"  - Positive: {(self.y_test == 1).sum():,} ({(self.y_test == 1).mean()*100:.1f}%)")
        print(f"  - Negative: {(self.y_test == 0).sum():,} ({(self.y_test == 0).mean()*100:.1f}%)")
        
        return self
    
    def create_vectorizer(self, vectorizer_type='tfidf', max_features=5000, ngram_range=(1, 2)):
        """Create text vectorizer."""
        print("\n" + "=" * 80)
        print("CREATING VECTORIZER")
        print("=" * 80)
        
        if vectorizer_type == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                strip_accents='unicode',
                stop_words='english'
            )
            print(f"✓ TF-IDF Vectorizer created")
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                min_df=2,
                max_df=0.95,
                strip_accents='unicode',
                stop_words='english'
            )
            print(f"✓ Count Vectorizer created")
        
        print(f"  - Max features: {max_features}")
        print(f"  - N-gram range: {ngram_range}")
        print(f"  - Stop words: English")
        
        # Fit and transform
        print("\n🔄 Vectorizing text...")
        self.X_train_vec = self.vectorizer.fit_transform(self.X_train)
        self.X_test_vec = self.vectorizer.transform(self.X_test)
        
        print(f"✓ Train shape: {self.X_train_vec.shape}")
        print(f"✓ Test shape: {self.X_test_vec.shape}")
        print(f"✓ Vocabulary size: {len(self.vectorizer.vocabulary_):,} words")
        
        return self
    
    def initialize_models(self):
        """Initialize classification models."""
        print("\n" + "=" * 80)
        print("INITIALIZING MODELS")
        print("=" * 80)
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=42,
                C=1.0,
                class_weight='balanced'
            ),
            'Naive Bayes': MultinomialNB(alpha=1.0),
            'Linear SVM': LinearSVC(
                random_state=42,
                max_iter=2000,
                C=1.0,
                class_weight='balanced'
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=20,
                min_samples_split=10,
                n_jobs=-1,
                class_weight='balanced'
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=5,
                learning_rate=0.1
            )
        }
        
        print(f"✓ {len(self.models)} models initialized:")
        for name in self.models.keys():
            print(f"  • {name}")
        
        return self
    
    def train_and_evaluate(self):
        """Train and evaluate all models."""
        print("\n" + "=" * 80)
        print("TRAINING AND EVALUATION")
        print("=" * 80)
        
        for name, model in self.models.items():
            print(f"\n🔄 Training: {name}...")
            
            try:
                # Train
                model.fit(self.X_train_vec, self.y_train)
                
                # Predictions
                y_pred_train = model.predict(self.X_train_vec)
                y_pred_test = model.predict(self.X_test_vec)
                
                # Probabilities (if available)
                if hasattr(model, 'predict_proba'):
                    y_proba_test = model.predict_proba(self.X_test_vec)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_proba_test = model.decision_function(self.X_test_vec)
                else:
                    y_proba_test = y_pred_test
                
                # Metrics
                train_acc = accuracy_score(self.y_train, y_pred_train)
                test_acc = accuracy_score(self.y_test, y_pred_test)
                precision = precision_score(self.y_test, y_pred_test)
                recall = recall_score(self.y_test, y_pred_test)
                f1 = f1_score(self.y_test, y_pred_test)
                
                try:
                    auc = roc_auc_score(self.y_test, y_proba_test)
                except:
                    auc = None
                
                # Cross-validation
                cv_scores = cross_val_score(
                    model, self.X_train_vec[:5000], self.y_train[:5000], 
                    cv=3, scoring='accuracy', n_jobs=-1
                )
                
                self.results[name] = {
                    'Train Accuracy': train_acc,
                    'Test Accuracy': test_acc,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'ROC-AUC': auc,
                    'CV Accuracy': cv_scores.mean(),
                    'CV Std': cv_scores.std(),
                    'Overfitting': train_acc - test_acc,
                    'model': model,
                    'predictions': y_pred_test,
                    'probabilities': y_proba_test
                }
                
                print(f"  ✓ Accuracy: {test_acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
                
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
                continue
        
        return self
    
    def display_results(self):
        """Display results table."""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON RESULTS")
        print("=" * 80)
        
        results_df = pd.DataFrame(self.results).T.drop(['model', 'predictions', 'probabilities'], axis=1)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        print("\n" + results_df.to_string())
        
        # Save results
        results_df.to_csv('nlp_model_results.csv')
        print("\n✓ Results saved: nlp_model_results.csv")
        
        return self
    
    def plot_model_comparison(self):
        """Visualize model comparison."""
        print("\n" + "=" * 80)
        print("GENERATING COMPARISON PLOTS")
        print("=" * 80)
        
        results_df = pd.DataFrame(self.results).T.drop(['model', 'predictions', 'probabilities'], axis=1)
        results_df = results_df.sort_values('F1-Score', ascending=False)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Accuracy comparison
        x = np.arange(len(results_df))
        width = 0.35
        axes[0, 0].bar(x - width/2, results_df['Train Accuracy'], width, 
                      label='Train', color='skyblue', alpha=0.8)
        axes[0, 0].bar(x + width/2, results_df['Test Accuracy'], width, 
                      label='Test', color='orange', alpha=0.8)
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_title('Train vs Test Accuracy')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(results_df.index, rotation=45, ha='right')
        axes[0, 0].legend()
        axes[0, 0].set_ylim(0.5, 1.0)
        axes[0, 0].grid(axis='y', alpha=0.3)
        
        # Precision, Recall, F1
        x = np.arange(len(results_df))
        width = 0.25
        axes[0, 1].bar(x - width, results_df['Precision'], width, 
                      label='Precision', color='#4CAF50', alpha=0.8)
        axes[0, 1].bar(x, results_df['Recall'], width, 
                      label='Recall', color='#2196F3', alpha=0.8)
        axes[0, 1].bar(x + width, results_df['F1-Score'], width, 
                      label='F1-Score', color='#FF9800', alpha=0.8)
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Precision, Recall, F1-Score')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(results_df.index, rotation=45, ha='right')
        axes[0, 1].legend()
        axes[0, 1].set_ylim(0.5, 1.0)
        axes[0, 1].grid(axis='y', alpha=0.3)
        
        # ROC-AUC
        valid_auc = results_df[results_df['ROC-AUC'].notna()]
        axes[1, 0].barh(valid_auc.index, valid_auc['ROC-AUC'], color='purple', alpha=0.7)
        axes[1, 0].set_xlabel('ROC-AUC Score')
        axes[1, 0].set_title('ROC-AUC by Model')
        axes[1, 0].set_xlim(0.5, 1.0)
        for i, v in enumerate(valid_auc['ROC-AUC']):
            axes[1, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
        
        # Overfitting check
        axes[1, 1].barh(results_df.index, results_df['Overfitting'], 
                       color=['red' if x > 0.05 else 'green' for x in results_df['Overfitting']])
        axes[1, 1].set_xlabel('Overfitting (Train Acc - Test Acc)')
        axes[1, 1].set_title('Overfitting Analysis')
        axes[1, 1].axvline(x=0.05, color='orange', linestyle='--', label='Threshold')
        axes[1, 1].legend()
        for i, v in enumerate(results_df['Overfitting']):
            axes[1, 1].text(v + 0.002, i, f'{v:.3f}', va='center')
        
        plt.tight_layout()
        plt.savefig('nlp_model_comparison.png', dpi=300, bbox_inches='tight')
        print("✓ Comparison plot saved: nlp_model_comparison.png")
        plt.show()
        
        return self
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for top 3 models."""
        print("\n" + "=" * 80)
        print("GENERATING CONFUSION MATRICES")
        print("=" * 80)
        
        # Get top 3 models by F1-Score
        sorted_models = sorted(
            self.results.items(),
            key=lambda x: x[1]['F1-Score'],
            reverse=True
        )[:3]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, result) in enumerate(sorted_models):
            cm = confusion_matrix(self.y_test, result['predictions'])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'])
            axes[idx].set_title(f'{name}\nF1-Score: {result["F1-Score"]:.4f}')
            axes[idx].set_ylabel('True Label')
            axes[idx].set_xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("✓ Confusion matrices saved: confusion_matrices.png")
        plt.show()
        
        return self
    
    def plot_roc_curves(self):
        """Plot ROC curves."""
        print("\n" + "=" * 80)
        print("GENERATING ROC CURVES")
        print("=" * 80)
        
        plt.figure(figsize=(10, 8))
        
        for name, result in self.results.items():
            if result['ROC-AUC'] is not None:
                try:
                    fpr, tpr, _ = roc_curve(self.y_test, result['probabilities'])
                    plt.plot(fpr, tpr, label=f"{name} (AUC = {result['ROC-AUC']:.3f})", linewidth=2)
                except:
                    continue
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Sentiment Classification')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("✓ ROC curves saved: roc_curves.png")
        plt.show()
        
        return self
    
    def show_classification_report(self):
        """Show detailed classification report for best model."""
        print("\n" + "=" * 80)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 80)
        
        best_name = max(self.results, key=lambda x: self.results[x]['F1-Score'])
        best_pred = self.results[best_name]['predictions']
        
        print(f"\n🏆 Best Model: {best_name}")
        print(f"F1-Score: {self.results[best_name]['F1-Score']:.4f}")
        print("\n" + "="*80)
        
        report = classification_report(
            self.y_test, best_pred, 
            target_names=['Negative', 'Positive'],
            digits=4
        )
        print(report)
        
        return self
    
    def analyze_top_features(self, top_n=20):
        """Analyze most important features for each class."""
        print("\n" + "=" * 80)
        print("TOP PREDICTIVE FEATURES")
        print("=" * 80)
        
        # Use Logistic Regression coefficients
        if 'Logistic Regression' in self.results:
            model = self.results['Logistic Regression']['model']
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get coefficients
            coef = model.coef_[0]
            
            # Top positive (indicates positive sentiment)
            top_positive_idx = np.argsort(coef)[-top_n:][::-1]
            top_positive_features = [(feature_names[i], coef[i]) for i in top_positive_idx]
            
            # Top negative (indicates negative sentiment)
            top_negative_idx = np.argsort(coef)[:top_n]
            top_negative_features = [(feature_names[i], coef[i]) for i in top_negative_idx]
            
            print(f"\n🟢 Top {top_n} words indicating POSITIVE sentiment:")
            for i, (word, score) in enumerate(top_positive_features, 1):
                print(f"  {i:2d}. {word:30s} (score: {score:+.4f})")
            
            print(f"\n🔴 Top {top_n} words indicating NEGATIVE sentiment:")
            for i, (word, score) in enumerate(top_negative_features, 1):
                print(f"  {i:2d}. {word:30s} (score: {score:+.4f})")
            
            # Visualize
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Positive words
            words_pos = [w for w, s in top_positive_features]
            scores_pos = [s for w, s in top_positive_features]
            axes[0].barh(words_pos, scores_pos, color='green', alpha=0.7)
            axes[0].set_xlabel('Coefficient')
            axes[0].set_title('Top Words for POSITIVE Sentiment')
            axes[0].invert_yaxis()
            
            # Negative words
            words_neg = [w for w, s in top_negative_features]
            scores_neg = [s for w, s in top_negative_features]
            axes[1].barh(words_neg, scores_neg, color='red', alpha=0.7)
            axes[1].set_xlabel('Coefficient')
            axes[1].set_title('Top Words for NEGATIVE Sentiment')
            axes[1].invert_yaxis()
            
            plt.tight_layout()
            plt.savefig('top_features.png', dpi=300, bbox_inches='tight')
            print("\n✓ Feature importance plot saved: top_features.png")
            plt.show()
        
        return self
    
    def predict_new_reviews(self, reviews):
        """Predict sentiment for new reviews."""
        print("\n" + "=" * 80)
        print("PREDICTING NEW REVIEWS")
        print("=" * 80)
        
        best_name = max(self.results, key=lambda x: self.results[x]['F1-Score'])
        best_model = self.results[best_name]['model']
        
        # Vectorize
        X_new = self.vectorizer.transform(reviews)
        
        # Predict
        predictions = best_model.predict(X_new)
        
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_new)
        else:
            probabilities = None
        
        print(f"\n🔮 Using model: {best_name}\n")
        
        for i, (review, pred) in enumerate(zip(reviews, predictions), 1):
            sentiment = "👍 POSITIVE" if pred == 1 else "👎 NEGATIVE"
            if probabilities is not None:
                conf = probabilities[i-1][pred] * 100
                print(f"{i}. {sentiment} ({conf:.1f}% confidence)")
            else:
                print(f"{i}. {sentiment}")
            print(f"   Review: {review[:100]}...")
            print()
        
        return predictions
    
    def get_best_model(self):
        """Return best model and save it."""
        best_name = max(self.results, key=lambda x: self.results[x]['F1-Score'])
        best_model = self.results[best_name]['model']
        
        print("\n" + "=" * 80)
        print("BEST MODEL SUMMARY")
        print("=" * 80)
        print(f"\n🏆 Best Model: {best_name}")
        print(f"\n📊 Performance Metrics:")
        for metric in ['Test Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
            value = self.results[best_name][metric]
            if value is not None:
                print(f"  • {metric}: {value:.4f}")
        
        return best_name, best_model