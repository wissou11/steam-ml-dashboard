# steam_review_cleaner.py

import pandas as pd
import numpy as np
import re
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from functools import wraps
from abc import ABC, abstractmethod
import logging
from typing import List, Dict, Optional, Tuple
import os

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# ==================== DECORATORS ====================

def timing_decorator(func):
    """Decorator to measure execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed = end_time - start_time
        logging.info(f"{func.__name__} executed in {elapsed:.2f} seconds")
        return result
    return wrapper

def validation_decorator(required_columns=None):
    """Decorator to validate required columns."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if required_columns and hasattr(self, 'df'):
                missing = [col for col in required_columns if col not in self.df.columns]
                if missing:
                    raise DataValidationError(f"Missing required columns: {missing}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def plot_decorator(save_fig=True, filename=None):
    """Decorator for plotting functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if save_fig and plt.get_fignums():
                if filename:
                    fname = filename
                else:
                    fname = f"{func.__name__}.png"
                plt.tight_layout()
                plt.savefig(fname, dpi=300, bbox_inches='tight')
                logging.info(f"Plot saved as {fname}")
            return result
        return wrapper
    return decorator

def error_handler_decorator(default_return=None):
    """Decorator to handle errors gracefully."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                if default_return is not None:
                    return default_return
                raise
        return wrapper
    return decorator

# ==================== EXCEPTIONS ====================

class ReviewCleaningError(Exception):
    """Base exception for review cleaning errors."""
    pass

class DataValidationError(ReviewCleaningError):
    """Raised when data validation fails."""
    pass

class TextCleaningError(ReviewCleaningError):
    """Raised during text cleaning."""
    pass

class JSONExtractionError(ReviewCleaningError):
    """Raised during JSON extraction."""
    pass

class DatasetBalanceError(ReviewCleaningError):
    """Raised during dataset balancing."""
    pass

# ==================== INTERFACES ====================

class ITextCleaner(ABC):
    """Interface for text cleaning strategies."""
    
    @abstractmethod
    def clean(self, text: str) -> str:
        pass

class IReviewExtractor(ABC):
    """Interface for review extraction strategies."""
    
    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class IDatasetBalancer(ABC):
    """Interface for dataset balancing strategies."""
    
    @abstractmethod
    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# ==================== TEXT CLEANING STRATEGIES ====================

class RegexTextCleaner(ITextCleaner):
    """Text cleaner using regex patterns."""
    
    def __init__(self):
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> List[Tuple[str, str]]:
        """Compile regex patterns for cleaning."""
        return [
            (r'\[/?[a-z0-9]+\]', ''),  # Steam BBCode tags
            (r'http\S+|www\.\S+', ''),  # URLs
            (r'\S+@\S+', ''),  # Email addresses
            (r'@\w+', ''),  # Mentions
            (r'#(\w+)', r'\1'),  # Hashtags (keep text)
            (r'([!?.]){3,}', r'\1\1'),  # Excessive punctuation
            (r'(\w)\1{3,}', r'\1\1'),  # Character repetition
            (r'[^a-z0-9\s.,!?\'-]', ' '),  # Special characters
            (r'[\r\n\t]+', ' '),  # Newlines and tabs
            (r'\s+', ' '),  # Extra whitespace
        ]
    
    def clean(self, text: str) -> str:
        """Clean text using regex patterns."""
        if not isinstance(text, str):
            return ""
        
        try:
            # Convert to lowercase
            text = text.lower()
            
            # Apply all regex patterns
            for pattern, replacement in self.patterns:
                text = re.sub(pattern, replacement, text)
            
            # Strip leading/trailing whitespace
            text = text.strip()
            
            return text
            
        except Exception as e:
            raise TextCleaningError(f"Error cleaning text: {e}")

class AdvancedTextCleaner(RegexTextCleaner):
    """Advanced text cleaner with additional features."""
    
    def clean(self, text: str) -> str:
        """Clean text with advanced features."""
        if not isinstance(text, str):
            return ""
        
        try:
            # Use parent cleaning
            text = super().clean(text)
            
            # Additional cleaning
            if text:
                # Remove extra spaces around punctuation
                text = re.sub(r'\s+([.,!?])', r'\1', text)
                text = re.sub(r'([.,!?])\s+', r'\1 ', text)
                
                # Fix common contractions
                contractions = {
                    r"don't": "do not",
                    r"can't": "cannot",
                    r"won't": "will not",
                    r"i'm": "i am",
                    r"you're": "you are",
                    r"it's": "it is",
                    r"that's": "that is",
                    r"what's": "what is",
                    r"where's": "where is",
                    r"there's": "there is",
                    r"he's": "he is",
                    r"she's": "she is",
                }
                
                for pattern, replacement in contractions.items():
                    text = re.sub(pattern, replacement, text)
            
            return text
            
        except Exception as e:
            raise TextCleaningError(f"Error in advanced cleaning: {e}")

# ==================== REVIEW EXTRACTOR ====================

class JSONReviewExtractor(IReviewExtractor):
    """Extracts reviews from JSON column."""
    
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract reviews from JSON column."""
        if 'review_texts' not in df.columns:
            raise JSONExtractionError("'review_texts' column not found")
        
        all_reviews = []
        
        for _, row in df.iterrows():
            try:
                if pd.isna(row['review_texts']):
                    continue
                
                reviews = json.loads(row['review_texts'])
                
                for review in reviews:
                    all_reviews.append({
                        'app_id': row.get('app_id', 0),
                        'game_name': row.get('name', 'Unknown'),
                        'review_text': review.get('review_text', ''),
                        'sentiment': 1 if review.get('voted_up') else 0,
                        'votes_helpful': review.get('votes_helpful', 0),
                        'votes_funny': review.get('votes_funny', 0),
                        'timestamp': review.get('timestamp_created', 0)
                    })
            except json.JSONDecodeError:
                continue
            except Exception as e:
                logging.warning(f"Error extracting review: {e}")
                continue
        
        if not all_reviews:
            raise JSONExtractionError("No reviews extracted")
        
        return pd.DataFrame(all_reviews)

# ==================== DATASET BALANCERS ====================

class UndersampleBalancer(IDatasetBalancer):
    """Balances dataset by undersampling."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def balance(self, df: pd.DataFrame, target_ratio: float = 1.0) -> pd.DataFrame:
        """Balance dataset by undersampling majority class."""
        try:
            pos_df = df[df['sentiment'] == 1]
            neg_df = df[df['sentiment'] == 0]
            
            pos_count = len(pos_df)
            neg_count = len(neg_df)
            
            if target_ratio == 1.0:
                # Equal balance
                min_count = min(pos_count, neg_count)
                pos_sample = pos_df.sample(min_count, random_state=self.random_state)
                neg_sample = neg_df.sample(min_count, random_state=self.random_state)
            else:
                # Custom ratio
                if pos_count > neg_count:
                    target_pos = int(neg_count * target_ratio)
                    pos_sample = pos_df.sample(target_pos, random_state=self.random_state)
                    neg_sample = neg_df
                else:
                    target_neg = int(pos_count * target_ratio)
                    pos_sample = pos_df
                    neg_sample = neg_df.sample(target_neg, random_state=self.random_state)
            
            balanced_df = pd.concat([pos_sample, neg_sample])
            balanced_df = balanced_df.sample(frac=1, random_state=self.random_state)
            
            return balanced_df.reset_index(drop=True)
            
        except Exception as e:
            raise DatasetBalanceError(f"Undersampling failed: {e}")

class OversampleBalancer(IDatasetBalancer):
    """Balances dataset by oversampling."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
    
    def balance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Balance dataset by oversampling minority class."""
        try:
            pos_df = df[df['sentiment'] == 1]
            neg_df = df[df['sentiment'] == 0]
            
            pos_count = len(pos_df)
            neg_count = len(neg_df)
            
            # Oversample minority class
            if pos_count > neg_count:
                neg_sample = neg_df.sample(pos_count, replace=True, 
                                         random_state=self.random_state)
                balanced_df = pd.concat([pos_df, neg_sample])
            else:
                pos_sample = pos_df.sample(neg_count, replace=True, 
                                         random_state=self.random_state)
                balanced_df = pd.concat([pos_sample, neg_df])
            
            balanced_df = balanced_df.sample(frac=1, random_state=self.random_state)
            
            return balanced_df.reset_index(drop=True)
            
        except Exception as e:
            raise DatasetBalanceError(f"Oversampling failed: {e}")

# ==================== MAIN CLEANER ====================

class SteamReviewCleaner:
    """Complete pipeline for cleaning Steam reviews for NLP classification."""
    
    def __init__(self, csv_path: str):
        """Initialize with the cleaned Steam dataset."""
        self.csv_path = csv_path
        self.df = None
        self.reviews_df = None
        self.cleaned_reviews_df = None
        self.text_cleaner = AdvancedTextCleaner()
        self.review_extractor = JSONReviewExtractor()
        self.balancer = UndersampleBalancer()
        self._load_data()
    
    @timing_decorator
    @error_handler_decorator(default_return=None)
    def _load_data(self):
        """Load data from CSV."""
        try:
            self.df = pd.read_csv(self.csv_path)
            logging.info(f"Data loaded successfully. Shape: {self.df.shape}")
        except FileNotFoundError:
            raise DataValidationError(f"File not found: {self.csv_path}")
        except Exception as e:
            raise DataValidationError(f"Error loading data: {str(e)}")
    
    @timing_decorator
    @validation_decorator(required_columns=['review_texts'])
    def extract_reviews(self) -> 'SteamReviewCleaner':
        """Extract reviews from JSON column."""
        print("=" * 80)
        print("EXTRACTING REVIEWS FROM DATASET")
        print("=" * 80)
        
        try:
            self.reviews_df = self.review_extractor.extract(self.df)
            
            print(f"\n📊 Extraction Results:")
            print(f"  ✓ Total reviews extracted: {len(self.reviews_df):,}")
            print(f"  ✓ Positive reviews: {self.reviews_df['sentiment'].sum():,}")
            print(f"  ✓ Negative reviews: {(self.reviews_df['sentiment'] == 0).sum():,}")
            print(f"  ✓ Average length: {self.reviews_df['review_text'].str.len().mean():.0f} chars")
            
            return self
            
        except JSONExtractionError as e:
            logging.error(f"Review extraction failed: {e}")
            raise
    
    @timing_decorator
    def clean_all_reviews(self, min_length: int = 10, max_length: int = 5000) -> 'SteamReviewCleaner':
        """Clean all reviews with filtering."""
        print("\n" + "=" * 80)
        print("CLEANING REVIEW TEXTS")
        print("=" * 80)
        
        if self.reviews_df is None:
            raise DataValidationError("No reviews to clean. Run extract_reviews() first.")
        
        df = self.reviews_df.copy()
        
        print(f"\n📊 Initial statistics:")
        print(f"  Total reviews: {len(df):,}")
        print(f"  Avg length: {df['review_text'].str.len().mean():.0f} chars")
        
        # Apply cleaning
        print("\n🧹 Applying text cleaning...")
        df['cleaned_text'] = df['review_text'].apply(self.text_cleaner.clean)
        
        # Calculate metrics
        df['text_length'] = df['cleaned_text'].str.len()
        df['word_count'] = df['cleaned_text'].str.split().str.len()
        
        # Filter by length
        initial_count = len(df)
        df = df[(df['text_length'] >= min_length) & (df['text_length'] <= max_length)]
        removed_length = initial_count - len(df)
        if removed_length > 0:
            print(f"✓ Removed {removed_length:,} reviews (length <{min_length} or >{max_length} chars)")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['cleaned_text'])
        removed_duplicates = initial_count - len(df)
        if removed_duplicates > 0:
            print(f"✓ Removed {removed_duplicates:,} duplicate reviews")
        
        # Remove empty reviews
        initial_count = len(df)
        df = df[df['cleaned_text'].str.strip().str.len() > 0]
        removed_empty = initial_count - len(df)
        if removed_empty > 0:
            print(f"✓ Removed {removed_empty:,} empty reviews")
        
        self.cleaned_reviews_df = df
        
        print(f"\n📊 Final statistics:")
        print(f"  Total reviews: {len(df):,}")
        pos_count = df['sentiment'].sum()
        neg_count = (df['sentiment'] == 0).sum()
        print(f"  Positive: {pos_count:,} ({pos_count/len(df)*100:.1f}%)")
        print(f"  Negative: {neg_count:,} ({neg_count/len(df)*100:.1f}%)")
        print(f"  Avg length: {df['text_length'].mean():.0f} chars")
        print(f"  Avg word count: {df['word_count'].mean():.0f} words")
        
        return self
    
    @timing_decorator
    @plot_decorator(save_fig=True, filename='review_cleaning_analysis.png')
    def analyze_cleaning_impact(self) -> 'SteamReviewCleaner':
        """Visualize the impact of cleaning."""
        print("\n" + "=" * 80)
        print("ANALYZING CLEANING IMPACT")
        print("=" * 80)
        
        if self.reviews_df is None or self.cleaned_reviews_df is None:
            raise DataValidationError("No data to analyze")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Before vs After length distribution
        axes[0, 0].hist(self.reviews_df['review_text'].str.len(), 
                       bins=50, alpha=0.5, label='Before', color='red', edgecolor='black')
        axes[0, 0].hist(self.cleaned_reviews_df['text_length'], 
                       bins=50, alpha=0.5, label='After', color='green', edgecolor='black')
        axes[0, 0].set_xlabel('Text Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Text Length Distribution: Before vs After Cleaning')
        axes[0, 0].legend()
        axes[0, 0].set_xlim(0, 1000)
        
        # Word count distribution
        axes[0, 1].hist(self.cleaned_reviews_df['word_count'], 
                       bins=50, color='steelblue', edgecolor='black')
        axes[0, 1].set_xlabel('Word Count')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Word Count Distribution (After Cleaning)')
        axes[0, 1].set_xlim(0, 200)
        
        # Sentiment distribution
        sentiment_counts = self.cleaned_reviews_df['sentiment'].value_counts()
        axes[1, 0].bar(['Negative', 'Positive'], 
                      [sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)],
                      color=['#ff6b6b', '#51cf66'], edgecolor='black')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Sentiment Distribution')
        for i, v in enumerate([sentiment_counts.get(0, 0), sentiment_counts.get(1, 0)]):
            axes[1, 0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')
        
        # Length by sentiment
        self.cleaned_reviews_df.boxplot(column='text_length', by='sentiment', ax=axes[1, 1])
        axes[1, 1].set_xlabel('Sentiment (0=Negative, 1=Positive)')
        axes[1, 1].set_ylabel('Text Length (characters)')
        axes[1, 1].set_title('Text Length by Sentiment')
        plt.suptitle('')
        
        plt.tight_layout()
        print("✓ Analysis plot generated")
        plt.show()
        
        return self
    
    @timing_decorator
    def show_examples(self, n_examples: int = 5) -> 'SteamReviewCleaner':
        """Show examples of before/after cleaning."""
        print("\n" + "=" * 80)
        print("CLEANING EXAMPLES")
        print("=" * 80)
        
        if self.cleaned_reviews_df is None:
            raise DataValidationError("No cleaned reviews to show")
        
        # Get random samples
        pos_sample = self.cleaned_reviews_df[
            self.cleaned_reviews_df['sentiment'] == 1
        ].sample(min(n_examples, (self.cleaned_reviews_df['sentiment'] == 1).sum()))
        
        neg_sample = self.cleaned_reviews_df[
            self.cleaned_reviews_df['sentiment'] == 0
        ].sample(min(n_examples, (self.cleaned_reviews_df['sentiment'] == 0).sum()))
        
        samples = pd.concat([pos_sample, neg_sample])
        
        for _, row in samples.iterrows():
            sentiment = "👍 POSITIVE" if row['sentiment'] == 1 else "👎 NEGATIVE"
            
            print(f"\n{'-'*80}")
            print(f"{sentiment} | Game: {row['game_name']}")
            print(f"{'-'*80}")
            print(f"BEFORE:")
            # Find original text
            original = self._find_original_text(row['app_id'], row['cleaned_text'])
            if original:
                print(f"  {original[:300]}..." if len(original) > 300 else f"  {original}")
            else:
                print("  Original text not found")
            
            print(f"\nAFTER ({row['text_length']} chars, {row['word_count']} words):")
            print(f"  {row['cleaned_text'][:300]}..." if len(row['cleaned_text']) > 300 else f"  {row['cleaned_text']}")
        
        return self
    
    def _find_original_text(self, app_id: int, cleaned_text: str) -> Optional[str]:
        """Find original text for a cleaned review."""
        if self.reviews_df is None:
            return None
        
        try:
            # Try to find matching review
            mask = (self.reviews_df['app_id'] == app_id)
            candidates = self.reviews_df[mask]
            
            if len(candidates) == 0:
                return None
            
            # Simple matching by checking if cleaned text is in original
            for _, row in candidates.iterrows():
                if cleaned_text[:50].lower() in row['review_text'].lower():
                    return row['review_text']
            
            return None
            
        except Exception:
            return None
    
    @timing_decorator
    def get_text_statistics(self) -> 'SteamReviewCleaner':
        """Get detailed statistics about the cleaned text."""
        print("\n" + "=" * 80)
        print("DETAILED TEXT STATISTICS")
        print("=" * 80)
        
        if self.cleaned_reviews_df is None:
            raise DataValidationError("No cleaned data for statistics")
        
        df = self.cleaned_reviews_df
        
        print("\n📊 Overall Statistics:")
        print(f"  Total reviews: {len(df):,}")
        print(f"  Unique reviews: {df['cleaned_text'].nunique():,}")
        print(f"  Avg length: {df['text_length'].mean():.1f} ± {df['text_length'].std():.1f} chars")
        print(f"  Avg words: {df['word_count'].mean():.1f} ± {df['word_count'].std():.1f} words")
        print(f"  Median length: {df['text_length'].median():.0f} chars")
        
        # By sentiment
        print("\n📊 By Sentiment:")
        for sentiment, label in [(0, "Negative"), (1, "Positive")]:
            subset = df[df['sentiment'] == sentiment]
            if len(subset) > 0:
                print(f"\n  {label} Reviews:")
                print(f"    Count: {len(subset):,}")
                print(f"    Avg length: {subset['text_length'].mean():.1f} chars")
                print(f"    Avg words: {subset['word_count'].mean():.1f} words")
        
        # Length quantiles
        print("\n📊 Length Quantiles:")
        quantiles = df['text_length'].quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        for q, val in quantiles.items():
            print(f"  {int(q*100)}th percentile: {val:.0f} chars")
        
        return self
    
    @timing_decorator
    def balance_dataset(self, method: str = 'undersample', target_ratio: float = 1.0) -> 'SteamReviewCleaner':
        """Balance positive/negative reviews."""
        print("\n" + "=" * 80)
        print("BALANCING DATASET")
        print("=" * 80)
        
        if self.cleaned_reviews_df is None:
            raise DataValidationError("No cleaned data to balance")
        
        df = self.cleaned_reviews_df.copy()
        
        pos_count = (df['sentiment'] == 1).sum()
        neg_count = (df['sentiment'] == 0).sum()
        
        print(f"\nBefore balancing:")
        print(f"  Positive: {pos_count:,}")
        print(f"  Negative: {neg_count:,}")
        print(f"  Ratio: {pos_count/neg_count:.2f}:1")
        
        try:
            if method == 'undersample':
                self.cleaned_reviews_df = self.balancer.balance(df, target_ratio)
            elif method == 'oversample':
                oversampler = OversampleBalancer()
                self.cleaned_reviews_df = oversampler.balance(df)
            else:
                raise DatasetBalanceError(f"Unknown balancing method: {method}")
            
            df_balanced = self.cleaned_reviews_df
            pos_balanced = (df_balanced['sentiment'] == 1).sum()
            neg_balanced = (df_balanced['sentiment'] == 0).sum()
            
            print(f"\nAfter balancing ({method}):")
            print(f"  Positive: {pos_balanced:,}")
            print(f"  Negative: {neg_balanced:,}")
            print(f"  Total: {len(df_balanced):,}")
            print(f"  Ratio: {pos_balanced/neg_balanced:.2f}:1")
            
            return self
            
        except DatasetBalanceError as e:
            logging.error(f"Balancing failed: {e}")
            raise
    
    @timing_decorator
    def export_cleaned_data(self, output_file: str = 'steam_reviews_cleaned.csv') -> pd.DataFrame:
        """Export cleaned reviews to CSV."""
        print("\n" + "=" * 80)
        print("EXPORTING CLEANED DATA")
        print("=" * 80)
        
        if self.cleaned_reviews_df is None:
            raise DataValidationError("No cleaned data to export")
        
        # Select relevant columns
        export_df = self.cleaned_reviews_df[[
            'app_id', 'game_name', 'cleaned_text', 'sentiment', 
            'text_length', 'word_count', 'votes_helpful', 'votes_funny'
        ]].copy()
        
        # Save to CSV
        export_df.to_csv(output_file, index=False)
        
        print(f"\n✅ Data exported to: {output_file}")
        print(f"📊 Statistics:")
        print(f"  Total reviews: {len(export_df):,}")
        print(f"  Columns: {', '.join(export_df.columns)}")
        print(f"  File size: {export_df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        
        # Show data preview
        print(f"\n📋 Data Preview:")
        print(export_df.head(3).to_string())
        
        return export_df
    
    def run_pipeline(self, min_length: int = 20, max_length: int = 2000) -> pd.DataFrame:
        """Run complete cleaning pipeline."""
        print("=" * 80)
        print("STEAM REVIEWS - NLP TEXT CLEANING PIPELINE")
        print("=" * 80)
        
        try:
            # Step 1: Extract reviews
            self.extract_reviews()
            
            # Step 2: Clean reviews
            self.clean_all_reviews(min_length=min_length, max_length=max_length)
            
            # Step 3: Show statistics
            self.get_text_statistics()
            
            # Step 4: Show examples
            self.show_examples(n_examples=3)
            
            # Step 5: Analyze impact
            self.analyze_cleaning_impact()
            
            # Step 6: Balance dataset (optional)
            self.balance_dataset(method='undersample')
            
            # Step 7: Export data
            cleaned_df = self.export_cleaned_data('steam_reviews_cleaned_nlp.csv')
            
            print("\n" + "=" * 80)
            print("✅ CLEANING PIPELINE COMPLETED!")
            print("=" * 80)
            
            return cleaned_df
            
        except Exception as e:
            logging.error(f"Pipeline failed: {e}")
            raise