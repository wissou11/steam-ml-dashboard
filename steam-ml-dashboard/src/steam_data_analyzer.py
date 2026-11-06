# steam_data_analyzer.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import json
import warnings
from functools import wraps
from abc import ABC, abstractmethod
import logging
from datetime import datetime
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

def validation_decorator(required_fields=None):
    """Decorator to validate data before processing."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if required_fields:
                missing = [field for field in required_fields 
                          if field not in self.df.columns]
                if missing:
                    raise DataValidationError(f"Missing required fields: {missing}")
            return func(self, *args, **kwargs)
        return wrapper
    return decorator

def plot_decorator(save_fig=True):
    """Decorator for plotting functions."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            if save_fig and plt.get_fignums():
                plt.tight_layout()
                plt.show()
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

class DataAnalysisError(Exception):
    """Base exception for data analysis errors."""
    pass

class DataValidationError(DataAnalysisError):
    """Raised when data validation fails."""
    pass

class FeatureEngineeringError(DataAnalysisError):
    """Raised during feature engineering."""
    pass

class VisualizationError(DataAnalysisError):
    """Raised during visualization generation."""
    pass

class PCAAnalysisError(DataAnalysisError):
    """Raised during PCA analysis."""
    pass

# ==================== INTERFACES ====================

class IDataAnalyzer(ABC):
    """Interface for data analyzers."""
    
    @abstractmethod
    def clean_data(self):
        pass
    
    @abstractmethod
    def analyze(self):
        pass

class IDataCleaner(ABC):
    """Interface for data cleaning strategies."""
    
    @abstractmethod
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

class IFeatureEngineer(ABC):
    """Interface for feature engineering."""
    
    @abstractmethod
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        pass

# ==================== DATA CLEANING STRATEGIES ====================

class BasicDataCleaner(IDataCleaner):
    """Basic data cleaning strategy."""
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform basic cleaning operations."""
        df_clean = df.copy()
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates(subset=['app_id'])
        
        # Handle free games
        df_clean.loc[df_clean['is_free'] == True, 'price_final'] = 0
        df_clean.loc[df_clean['is_free'] == True, 'price_initial'] = 0
        
        # Remove games with no price info
        df_clean = df_clean[~((df_clean['price_final'].isna()) & 
                            (df_clean['is_free'] == False))]
        
        return df_clean

class AdvancedDataCleaner(IDataCleaner):
    """Advanced data cleaning with feature engineering."""
    
    def __init__(self, current_year: int = 2024):
        self.current_year = current_year
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Perform advanced cleaning and feature engineering."""
        df_clean = df.copy()
        
        # Convert release_date to datetime
        df_clean['release_date'] = pd.to_datetime(df_clean['release_date'], errors='coerce')
        df_clean['release_year'] = df_clean['release_date'].dt.year
        df_clean['release_month'] = df_clean['release_date'].dt.month
        
        # Remove upcoming games
        df_clean = df_clean[df_clean['coming_soon'] == False]
        
        # Remove invalid release dates
        df_clean = df_clean[(df_clean['release_year'] >= 1990) & 
                          (df_clean['release_year'] <= self.current_year)]
        
        # Calculate game age
        df_clean['game_age_years'] = self.current_year - df_clean['release_year']
        
        # Fill missing values
        df_clean = self._fill_missing_values(df_clean)
        
        # Create engineered features
        df_clean = self._engineer_features(df_clean)
        
        # Remove outliers
        df_clean = self._remove_outliers(df_clean)
        
        return df_clean
    
    def _fill_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in the dataframe."""
        # Fill review metrics
        review_cols = ['total_positive', 'total_negative', 'total_reviews', 
                      'review_score', 'recommendations']
        for col in review_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0)
        
        # Fill metacritic with median
        if 'metacritic_score' in df.columns:
            df['metacritic_score'] = df['metacritic_score'].fillna(
                df['metacritic_score'].median())
        
        return df
    
    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create new engineered features."""
        # Positive ratio
        df['positive_ratio'] = df.apply(
            lambda x: x['total_positive'] / x['total_reviews'] 
            if x['total_reviews'] > 0 else 0, axis=1
        )
        
        # Platform count
        df['platform_count'] = df['windows'].astype(int) + df['mac'].astype(int) + df['linux'].astype(int)
        
        # Developer count
        df['developer_count'] = df['developers'].apply(
            lambda x: len(str(x).split(',')) if pd.notna(x) else 1
        )
        
        return df
    
    def _remove_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove outliers from the data."""
        # Remove expensive games (> $200)
        df = df[df['price_final'] <= 200]
        
        # Remove extreme description lengths
        if 'detailed_description_length' in df.columns:
            q99 = df['detailed_description_length'].quantile(0.99)
            df = df[df['detailed_description_length'] <= q99]
        
        return df

# ==================== FEATURE ENGINEERING ====================

class SteamFeatureEngineer(IFeatureEngineer):
    """Feature engineer for Steam data."""
    
    def __init__(self):
        self.numeric_features = None
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data."""
        try:
            df_engineered = df.copy()
            
            # Additional feature engineering can be added here
            if 'price_initial' in df_engineered.columns and 'price_final' in df_engineered.columns:
                df_engineered['price_difference'] = df_engineered['price_initial'] - df_engineered['price_final']
            
            self.numeric_features = self._identify_numeric_features(df_engineered)
            return df_engineered
            
        except Exception as e:
            raise FeatureEngineeringError(f"Feature engineering failed: {str(e)}")
    
    def _identify_numeric_features(self, df: pd.DataFrame) -> list:
        """Identify numeric features for analysis."""
        return df.select_dtypes(include=[np.number]).columns.tolist()

# ==================== VISUALIZATION COMPONENTS ====================

class DataVisualizer:
    """Handles data visualization."""
    
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    @plot_decorator(save_fig=True)
    def plot_price_distribution(self, df: pd.DataFrame):
        """Plot price distribution visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Price histogram
        axes[0, 0].hist(df[df['price_final'] > 0]['price_final'], bins=50, 
                       edgecolor='black', alpha=0.7)
        axes[0, 0].set_xlabel('Price ($)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Price Distribution (Excluding Free Games)')
        
        # Log price
        axes[0, 1].hist(np.log1p(df['price_final']), bins=50, 
                       edgecolor='black', color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Log(Price + 1)')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Log-Transformed Price Distribution')
        
        # Free vs Paid
        price_category = df['is_free'].value_counts()
        axes[1, 0].bar(['Paid', 'Free'], price_category.values, 
                      color=['#1f77b4', '#ff7f0e'], alpha=0.7)
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Free vs Paid Games')
        
        # Price by platform
        if 'platform_count' in df.columns:
            df.boxplot(column='price_final', by='platform_count', ax=axes[1, 1])
            axes[1, 1].set_xlabel('Number of Platforms')
            axes[1, 1].set_ylabel('Price ($)')
            axes[1, 1].set_title('Price by Platform Availability')
        
        plt.suptitle('')
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/price_analysis.png', dpi=300, bbox_inches='tight')
        
        return fig
    
    @plot_decorator(save_fig=True)
    def plot_correlation_matrix(self, df: pd.DataFrame, target_col: str = 'price_final'):
        """Plot correlation matrix."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        corr_matrix = df[numeric_cols].corr()
        
        # Price correlations
        price_corr = corr_matrix[target_col].sort_values(ascending=False)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Full correlation matrix
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True, linewidths=1,
                   cbar_kws={"shrink": 0.8}, ax=axes[0])
        axes[0].set_title('Feature Correlation Matrix', fontsize=14)
        
        # Price correlations
        price_corr_abs = price_corr.abs().sort_values(ascending=True)
        price_corr_abs[1:16].plot(kind='barh', color='steelblue', ax=axes[1])
        axes[1].set_xlabel('Absolute Correlation with Price')
        axes[1].set_title('Top 15 Features by Correlation with Price')
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/correlation_analysis.png', dpi=300, bbox_inches='tight')
        
        return fig, corr_matrix
    
    @plot_decorator(save_fig=True)
    def plot_pca_results(self, pca, explained_var, feature_names):
        """Plot PCA results."""
        cumulative_var = np.cumsum(explained_var)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Individual variance
        axes[0].bar(range(1, len(explained_var) + 1), explained_var, 
                   alpha=0.8, color='steelblue')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Explained Variance by Component')
        
        # Cumulative variance
        axes[1].plot(range(1, len(cumulative_var) + 1), cumulative_var, 
                    'o-', linewidth=2, markersize=8)
        axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% Variance')
        axes[1].axhline(y=0.9, color='g', linestyle='--', label='90% Variance')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Explained Variance')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{self.save_dir}/pca_variance.png', dpi=300, bbox_inches='tight')
        
        return fig

# ==================== MAIN ANALYZER ====================

class SteamDataAnalyzer(IDataAnalyzer):
    """Complete analyzer for Steam game data."""
    
    def __init__(self, csv_path: str, current_year: int = 2024):
        self.csv_path = csv_path
        self.current_year = current_year
        self.df = None
        self.df_cleaned = None
        self.data_cleaner = AdvancedDataCleaner(current_year)
        self.feature_engineer = SteamFeatureEngineer()
        self.visualizer = DataVisualizer()
        self.load_data()
    
    @timing_decorator
    def load_data(self):
        """Load data from CSV with automatic header detection."""
        try:
            # Test read to detect headers
            df_test = pd.read_csv(self.csv_path, nrows=1)
            
            if df_test.columns[0].isdigit():
                logging.info("No header detected. Adding column names...")
                column_names = [
                    'app_id', 'name', 'type', 'is_free', 'price_initial', 'price_final',
                    'discount_percent', 'release_date', 'coming_soon', 'developers',
                    'detailed_description_length', 'about_game_length', 'short_description',
                    'windows', 'mac', 'linux', 'categories', 'category_count',
                    'genres', 'genre_count', 'metacritic_score', 'recommendations',
                    'dlc_count', 'achievements_total', 'review_score', 'review_score_desc',
                    'total_positive', 'total_negative', 'total_reviews', 'review_texts'
                ]
                self.df = pd.read_csv(self.csv_path, names=column_names)
            else:
                self.df = pd.read_csv(self.csv_path)
            
            logging.info(f"Data loaded successfully. Shape: {self.df.shape}")
            
        except FileNotFoundError:
            raise DataValidationError(f"File not found: {self.csv_path}")
        except Exception as e:
            raise DataValidationError(f"Error loading data: {str(e)}")
    
    @timing_decorator
    @validation_decorator(required_fields=['app_id', 'price_final'])
    def clean_data(self) -> pd.DataFrame:
        """Clean and preprocess the dataset."""
        print("=" * 80)
        print("DATA CLEANING AND PREPROCESSING")
        print("=" * 80)
        
        initial_shape = self.df.shape
        print(f"\nInitial shape: {initial_shape}")
        
        # Clean data
        self.df_cleaned = self.data_cleaner.clean(self.df)
        
        # Engineer additional features
        self.df_cleaned = self.feature_engineer.engineer_features(self.df_cleaned)
        
        final_shape = self.df_cleaned.shape
        print(f"\nFinal shape after cleaning: {final_shape}")
        print(f"Rows removed: {initial_shape[0] - final_shape[0]}")
        
        # Show missing values
        missing = self.df_cleaned.isnull().sum()
        if missing.any():
            print("\nMissing values per column:")
            print(missing[missing > 0])
        
        return self.df_cleaned
    
    @timing_decorator
    def exploratory_analysis(self):
        """Perform exploratory data analysis."""
        if self.df_cleaned is None:
            raise DataValidationError("No cleaned data available. Run clean_data() first.")
        
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)
        
        # Price analysis
        self.visualizer.plot_price_distribution(self.df_cleaned)
        
        # Review analysis
        self._analyze_reviews()
        
        # Temporal analysis
        self._analyze_temporal_patterns()
        
        # Summary statistics
        self._print_summary_statistics()
        
        print("✅ Exploratory analysis completed")
    
    def _analyze_reviews(self):
        """Analyze review patterns."""
        df = self.df_cleaned
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total reviews distribution
        axes[0, 0].hist(np.log1p(df['total_reviews']), bins=50, 
                       edgecolor='black', color='purple', alpha=0.7)
        axes[0, 0].set_xlabel('Log(Total Reviews + 1)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Review Count Distribution')
        
        # Positive ratio distribution
        axes[0, 1].hist(df['positive_ratio'], bins=50, 
                       edgecolor='black', color='green', alpha=0.7)
        axes[0, 1].set_xlabel('Positive Review Ratio')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Positive Review Ratio Distribution')
        
        # Reviews vs Price
        reviews_data = df[df['total_reviews'] > 0]
        if len(reviews_data) > 0:
            sample = reviews_data.sample(min(5000, len(reviews_data)))
            axes[1, 0].scatter(np.log1p(sample['total_reviews']), 
                              sample['price_final'], alpha=0.5, s=10)
            axes[1, 0].set_xlabel('Log(Total Reviews + 1)')
            axes[1, 0].set_ylabel('Price ($)')
            axes[1, 0].set_title('Reviews vs Price')
        
        # Metacritic vs Price
        if 'metacritic_score' in df.columns:
            valid_meta = df[df['metacritic_score'] > 0]
            if len(valid_meta) > 0:
                sample_meta = valid_meta.sample(min(5000, len(valid_meta)))
                axes[1, 1].scatter(sample_meta['metacritic_score'], 
                                  sample_meta['price_final'], alpha=0.5, s=10, color='red')
                axes[1, 1].set_xlabel('Metacritic Score')
                axes[1, 1].set_ylabel('Price ($)')
                axes[1, 1].set_title('Metacritic Score vs Price')
        
        plt.tight_layout()
        plt.savefig(f'{self.visualizer.save_dir}/reviews_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _analyze_temporal_patterns(self):
        """Analyze temporal patterns."""
        df = self.df_cleaned
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Games released per year
        if 'release_year' in df.columns:
            games_per_year = df['release_year'].value_counts().sort_index()
            axes[0, 0].plot(games_per_year.index, games_per_year.values, linewidth=2)
            axes[0, 0].set_xlabel('Year')
            axes[0, 0].set_ylabel('Number of Games')
            axes[0, 0].set_title('Games Released Over Time')
        
        # Average price over years
        if 'release_year' in df.columns:
            avg_price_year = df.groupby('release_year')['price_final'].mean()
            axes[0, 1].plot(avg_price_year.index, avg_price_year.values, 
                           linewidth=2, color='green')
            axes[0, 1].set_xlabel('Year')
            axes[0, 1].set_ylabel('Average Price ($)')
            axes[0, 1].set_title('Average Game Price Over Time')
        
        # Game age vs price
        if 'game_age_years' in df.columns:
            sample = df.sample(min(5000, len(df)))
            axes[1, 0].scatter(sample['game_age_years'], sample['price_final'], 
                              alpha=0.3, s=10)
            axes[1, 0].set_xlabel('Game Age (Years)')
            axes[1, 0].set_ylabel('Price ($)')
            axes[1, 0].set_title('Game Age vs Price')
        
        # DLC count vs price
        if 'dlc_count' in df.columns:
            sample = df.sample(min(5000, len(df)))
            axes[1, 1].scatter(sample['dlc_count'], sample['price_final'], 
                              alpha=0.3, s=10, color='orange')
            axes[1, 1].set_xlabel('DLC Count')
            axes[1, 1].set_ylabel('Price ($)')
            axes[1, 1].set_title('DLC Count vs Price')
        
        plt.tight_layout()
        plt.savefig(f'{self.visualizer.save_dir}/temporal_analysis.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def _print_summary_statistics(self):
        """Print summary statistics."""
        df = self.df_cleaned
        
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS")
        print("=" * 80)
        
        print("\n📊 PRICE STATISTICS:")
        print(df['price_final'].describe())
        
        if 'total_reviews' in df.columns:
            print("\n📊 REVIEW STATISTICS:")
            print(df[['total_reviews', 'positive_ratio', 'review_score']].describe())
    
    @timing_decorator
    def correlation_analysis(self) -> pd.DataFrame:
        """Analyze feature correlations."""
        if self.df_cleaned is None:
            raise DataValidationError("No cleaned data available.")
        
        print("\n" + "=" * 80)
        print("CORRELATION ANALYSIS")
        print("=" * 80)
        
        _, corr_matrix = self.visualizer.plot_correlation_matrix(self.df_cleaned)
        
        # Print top correlations with price
        price_corr = corr_matrix['price_final'].sort_values(ascending=False)
        print("\nTop 15 Features Correlated with Price:")
        for i, (feature, corr) in enumerate(price_corr.head(15).items(), 1):
            print(f"{i:2d}. {feature:30s}: {corr:7.3f}")
        
        return corr_matrix
    
    @timing_decorator
    def pca_analysis(self, n_components: int = 10):
        """Perform Principal Component Analysis."""
        if self.df_cleaned is None:
            raise DataValidationError("No cleaned data available.")
        
        print("\n" + "=" * 80)
        print("PRINCIPAL COMPONENT ANALYSIS")
        print("=" * 80)
        
        try:
            # Select features for PCA
            feature_cols = [
                'discount_percent', 'detailed_description_length',
                'about_game_length', 'platform_count', 'category_count', 
                'genre_count', 'metacritic_score', 'recommendations', 
                'dlc_count', 'achievements_total', 'total_positive', 
                'total_negative', 'total_reviews', 'positive_ratio', 
                'game_age_years', 'developer_count'
            ]
            
            # Filter to existing columns
            feature_cols = [col for col in feature_cols if col in self.df_cleaned.columns]
            
            # Prepare data
            X = self.df_cleaned[feature_cols].copy()
            X = X.fillna(X.median())
            
            # Standardize
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit PCA
            pca = PCA(n_components=min(n_components, len(feature_cols)))
            X_pca = pca.fit_transform(X_scaled)
            
            # Explained variance
            explained_var = pca.explained_variance_ratio_
            
            # Plot results
            self.visualizer.plot_pca_results(pca, explained_var, feature_cols)
            
            # Print component information
            print("\nPCA Component Information:")
            for i, var in enumerate(explained_var, 1):
                print(f"PC{i}: {var:.4f} ({np.sum(explained_var[:i]):.4f} cumulative)")
            
            # Component loadings
            loadings = pd.DataFrame(
                pca.components_.T,
                columns=[f'PC{i+1}' for i in range(pca.n_components_)],
                index=feature_cols
            )
            
            print("\nTop Feature Loadings for First 3 Components:")
            for i in range(min(3, pca.n_components_)):
                print(f"\nPC{i+1} ({explained_var[i]:.2%} of variance):")
                top_features = loadings[f'PC{i+1}'].abs().sort_values(ascending=False).head(5)
                for feat, loading in top_features.items():
                    print(f"  • {feat:30s}: {loading:7.3f}")
            
            return pca, loadings
            
        except Exception as e:
            raise PCAAnalysisError(f"PCA analysis failed: {str(e)}")
    
    @timing_decorator
    def generate_feature_recommendations(self):
        """Generate feature recommendations for modeling."""
        if self.df_cleaned is None:
            raise DataValidationError("No cleaned data available.")
        
        print("\n" + "=" * 80)
        print("FEATURE RECOMMENDATIONS")
        print("=" * 80)
        
        # Calculate correlations
        numeric_cols = self.df_cleaned.select_dtypes(include=[np.number]).columns
        correlations = self.df_cleaned[numeric_cols].corrwith(
            self.df_cleaned['price_final']).abs().sort_values(ascending=False)
        
        # Categorize features
        strong_features = correlations[correlations > 0.15].drop('price_final')
        moderate_features = correlations[(correlations > 0.05) & (correlations <= 0.15)]
        weak_features = correlations[correlations <= 0.05]
        
        print("\n🟢 STRONGLY RECOMMENDED FEATURES (|correlation| > 0.15):")
        for feat, corr in strong_features.items():
            print(f"  • {feat:30s}: {corr:.3f}")
        
        print("\n🟡 MODERATELY USEFUL FEATURES (0.05 < |correlation| < 0.15):")
        for feat, corr in moderate_features.items():
            print(f"  • {feat:30s}: {corr:.3f}")
        
        print("\n🔴 WEAK FEATURES (|correlation| < 0.05):")
        for feat, corr in weak_features.head(10).items():
            print(f"  • {feat:30s}: {corr:.3f}")
        
        # Save recommendations
        recommendations = {
            'strong_features': strong_features.index.tolist(),
            'moderate_features': moderate_features.index.tolist(),
            'weak_features': weak_features.index.tolist()
        }
        
        return recommendations
    
    @timing_decorator
    @error_handler_decorator(default_return=None)
    def extract_reviews_for_nlp(self, output_file: str = 'steam_reviews_for_nlp.csv'):
        """Extract review texts for NLP analysis."""
        if self.df_cleaned is None or 'review_texts' not in self.df_cleaned.columns:
            logging.warning("No review texts available for extraction")
            return None
        
        print("\n" + "=" * 80)
        print("EXTRACTING REVIEWS FOR NLP")
        print("=" * 80)
        
        all_reviews = []
        
        for _, row in self.df_cleaned.iterrows():
            try:
                if pd.isna(row['review_texts']):
                    continue
                
                reviews = json.loads(row['review_texts'])
                
                for review in reviews:
                    all_reviews.append({
                        'app_id': row['app_id'],
                        'game_name': row.get('name', 'Unknown'),
                        'review_text': review.get('review_text', ''),
                        'sentiment': 1 if review.get('voted_up') else 0,
                        'votes_helpful': review.get('votes_helpful', 0),
                        'votes_funny': review.get('votes_funny', 0),
                        'timestamp': review.get('timestamp_created', 0)
                    })
            except Exception as e:
                continue
        
        if not all_reviews:
            print("No reviews extracted")
            return None
        
        reviews_df = pd.DataFrame(all_reviews)
        reviews_df = reviews_df[reviews_df['review_text'].str.len() > 10]
        
        print(f"\n📊 Reviews Extracted:")
        print(f"  Total reviews: {len(reviews_df):,}")
        print(f"  Positive reviews: {reviews_df['sentiment'].sum():,}")
        print(f"  Negative reviews: {(reviews_df['sentiment'] == 0).sum():,}")
        print(f"  Avg review length: {reviews_df['review_text'].str.len().mean():.0f} chars")
        
        # Save to file
        reviews_df.to_csv(output_file, index=False)
        print(f"\n✅ Reviews saved to: {output_file}")
        
        return reviews_df
    
    @timing_decorator
    def save_cleaned_data(self, output_file: str = 'steam_cleaned_data.csv'):
        """Save cleaned data to CSV."""
        if self.df_cleaned is None:
            raise DataValidationError("No cleaned data to save.")
        
        self.df_cleaned.to_csv(output_file, index=False)
        print(f"✅ Cleaned data saved to: {output_file}")
        return output_file
    
    def analyze(self):
        """Complete analysis pipeline."""
        print("\n" + "=" * 80)
        print("STEAM DATA ANALYSIS PIPELINE")
        print("=" * 80)
        
        try:
            # Step 1: Clean data
            self.clean_data()
            
            # Step 2: EDA
            self.exploratory_analysis()
            
            # Step 3: Correlation analysis
            self.correlation_analysis()
            
            # Step 4: PCA analysis
            self.pca_analysis()
            
            # Step 5: Feature recommendations
            recommendations = self.generate_feature_recommendations()
            
            # Step 6: Extract reviews for NLP
            self.extract_reviews_for_nlp()
            
            # Step 7: Save cleaned data
            self.save_cleaned_data()
            
            print("\n" + "=" * 80)
            print("✅ ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 80)
            
            return recommendations
            
        except Exception as e:
            logging.error(f"Analysis pipeline failed: {e}")
            raise