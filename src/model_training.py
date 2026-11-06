import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ModelTrainingException(Exception):
    """Custom exception for model training errors."""
    pass


class SteamPricePredictor:
    """Complete pipeline for Steam game price prediction."""
    
    def __init__(self, csv_path):
        """Initialize with cleaned data CSV."""
        self.df = pd.read_csv(csv_path)
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        
    def prepare_features(self, target='price_final', feature_selection='balanced'):
        """Prepare features for modeling based on correlation analysis.
        
        Args:
            target: Target variable name
            feature_selection: 'strong', 'balanced', or 'all'
                - 'strong': Only features with |correlation| > 0.15
                - 'balanced': Strong + moderate features (|correlation| > 0.05)
                - 'all': All available features
        """
        print("=" * 80)
        print("FEATURE PREPARATION")
        print("=" * 80)
        
        df = self.df.copy()
        
        # Remove rows with missing target
        df = df[df[target].notna()]
        
        # Define feature sets based on correlation analysis
        # ⚠️ REMOVED price_initial - it's data leakage (too correlated with target)
        strong_features = [
            'discount_percent',   # 0.312
            'release_year',       # 0.265
            'game_age_years',     # 0.265
            'positive_ratio',     # 0.244
            'review_score',       # 0.234
            'genre_count',        # 0.221
            'platform_count',     # 0.194
        ]
        
        moderate_features = [
            'achievements_total', # 0.090
            'metacritic_score',   # 0.082
            'developer_count',    # 0.076
            'release_month',      # 0.066
            'total_negative',     # 0.060
            'category_count',     # 0.058
        ]
        
        # Weak features kept for completeness (only in 'all' mode)
        weak_features = [
            'about_game_length',
            'detailed_description_length',
            'recommendations',
            'total_reviews',
            'total_positive',
            'dlc_count',
            'windows', 'mac', 'linux'
        ]
        
        # Select features based on mode
        if feature_selection == 'strong':
            feature_cols = strong_features
            print("\n📊 Feature Selection: STRONG ONLY (|correlation| > 0.15)")
        elif feature_selection == 'balanced':
            feature_cols = strong_features + moderate_features
            print("\n📊 Feature Selection: STRONG + MODERATE (|correlation| > 0.05)")
        else:  # 'all'
            feature_cols = strong_features + moderate_features + weak_features
            print("\n📊 Feature Selection: ALL FEATURES")
        
        # Filter to existing columns
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Feature Engineering - Create new features
        print("\n🔧 Engineering new features...")
        
        # Reviews per year (engagement rate)
        if 'total_reviews' in df.columns and 'game_age_years' in df.columns:
            df['reviews_per_year'] = df['total_reviews'] / (df['game_age_years'] + 1)
            feature_cols.append('reviews_per_year')
            print("  ✓ reviews_per_year created")
        
        # Price per DLC (without using price_initial to avoid leakage)
        if 'dlc_count' in df.columns:
            df['has_dlc'] = (df['dlc_count'] > 0).astype(int)
            feature_cols.append('has_dlc')
            print("  ✓ has_dlc created")
        
        # Multi-platform indicator
        if all(col in df.columns for col in ['windows', 'mac', 'linux']):
            df['is_multiplatform'] = ((df['windows'].astype(int) + 
                                       df['mac'].astype(int) + 
                                       df['linux'].astype(int)) > 1).astype(int)
            feature_cols.append('is_multiplatform')
            print("  ✓ is_multiplatform created")
        
        # Review quality score (combines volume and sentiment)
        if 'total_reviews' in df.columns and 'positive_ratio' in df.columns:
            df['review_quality_score'] = np.log1p(df['total_reviews']) * df['positive_ratio']
            feature_cols.append('review_quality_score')
            print("  ✓ review_quality_score created")
        
        # Content richness (has achievements, has DLC, etc.)
        content_features = []
        if 'achievements_total' in df.columns:
            df['has_achievements'] = (df['achievements_total'] > 0).astype(int)
            content_features.append('has_achievements')
        if 'dlc_count' in df.columns:
            content_features.append('has_dlc')
        
        if content_features:
            feature_cols.extend([f for f in content_features if f not in feature_cols])
            print(f"  ✓ Content indicators created: {', '.join(content_features)}")
        
        # Handle missing values
        for col in feature_cols:
            if col in df.columns and df[col].isna().sum() > 0:
                df[col] = df[col].fillna(df[col].median())
        
        # Prepare X and y
        X = df[feature_cols]
        y = df[target]
        
        # Remove any remaining NaN rows
        mask = ~(X.isna().any(axis=1) | y.isna())
        X = X[mask]
        y = y[mask]
        
        self.feature_names = feature_cols
        
        print(f"\n✓ Features prepared: {len(feature_cols)} features")
        print(f"✓ Total samples: {len(X)}")
        print(f"✓ Target variable: {target}")
        print(f"\n📋 Final feature list:")
        
        # Group features by category
        strong_used = [f for f in feature_cols if f in strong_features]
        moderate_used = [f for f in feature_cols if f in moderate_features]
        engineered = [f for f in feature_cols if f not in strong_features + moderate_features + weak_features]
        
        if strong_used:
            print(f"\n  🟢 Strong features ({len(strong_used)}):")
            for feat in strong_used:
                print(f"     • {feat}")
        
        if moderate_used:
            print(f"\n  🟡 Moderate features ({len(moderate_used)}):")
            for feat in moderate_used:
                print(f"     • {feat}")
        
        if engineered:
            print(f"\n  🔧 Engineered features ({len(engineered)}):")
            for feat in engineered:
                print(f"     • {feat}")
        
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into train and test sets."""
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"\n✓ Data split:")
        print(f"  - Train set: {len(self.X_train)} samples")
        print(f"  - Test set: {len(self.X_test)} samples")
        
        return self
    
    def get_feature_importance(self, model, model_name):
        """Extract feature importance from trained model."""
        importances = None
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                # Linear models
                importances = np.abs(model.coef_)
            
            if importances is not None:
                feat_imp = pd.DataFrame({
                    'feature': self.feature_names,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                print(f"\n📊 Top 10 Features for {model_name}:")
                print(feat_imp.head(10).to_string(index=False))
                
                return feat_imp
        except Exception as e:
            print(f"  ⚠ Could not extract feature importance: {str(e)}")
        
        return None


class ModelComparator:
    """Class to train and compare multiple models."""
    
    def __init__(self, X_train, X_test, y_train, y_test, feature_names=None):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        self.scaler = StandardScaler()
        
    def initialize_models(self):
        """Initialize models to compare."""
        try:
            self.models = {
                'Linear Regression': LinearRegression(),
                'Ridge Regression': Ridge(alpha=1.0),
                'Lasso Regression': Lasso(alpha=0.1, max_iter=5000),
                'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, max_iter=5000),
                'Decision Tree': DecisionTreeRegressor(
                    random_state=42, 
                    max_depth=8,  # Reduced to prevent overfitting
                    min_samples_split=20,
                    min_samples_leaf=10
                ),
                'Random Forest': RandomForestRegressor(
                    n_estimators=100, 
                    random_state=42, 
                    max_depth=10,  # Reduced to prevent overfitting
                    min_samples_split=20,
                    min_samples_leaf=10,
                    n_jobs=-1
                ),
                'Gradient Boosting': GradientBoostingRegressor(
                    n_estimators=100, 
                    random_state=42,
                    max_depth=5,  # Reduced to prevent overfitting
                    learning_rate=0.1,
                    min_samples_split=20,
                    min_samples_leaf=10
                ),
                'SVR': SVR(kernel='rbf', C=10.0, gamma='scale', epsilon=0.1)
            }
            print(f"✓ {len(self.models)} models initialized")
            return self
        except Exception as e:
            raise ModelTrainingException(f"Model initialization error: {str(e)}")
    
    def scale_features(self):
        """Normalize features."""
        try:
            self.X_train_scaled = self.scaler.fit_transform(self.X_train)
            self.X_test_scaled = self.scaler.transform(self.X_test)
            print("✓ Features normalized")
            return self
        except Exception as e:
            raise ModelTrainingException(f"Normalization error: {str(e)}")
    
    def train_and_evaluate(self):
        """Train and evaluate all models."""
        print("\n" + "="*80)
        print("MODEL TRAINING AND EVALUATION")
        print("="*80)
        
        for name, model in self.models.items():
            try:
                print(f"\n🔄 Training: {name}...")
                
                # Train model
                model.fit(self.X_train_scaled, self.y_train)
                
                # Predictions
                y_pred_train = model.predict(self.X_train_scaled)
                y_pred_test = model.predict(self.X_test_scaled)
                
                # Metrics
                train_r2 = r2_score(self.y_train, y_pred_train)
                test_r2 = r2_score(self.y_test, y_pred_test)
                test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
                test_mae = mean_absolute_error(self.y_test, y_pred_test)
                
                # MAPE (Mean Absolute Percentage Error) - handle division by zero
                non_zero_mask = self.y_test > 0.1  # Avoid division by very small numbers
                if non_zero_mask.sum() > 0:
                    mape = np.mean(np.abs((self.y_test[non_zero_mask] - y_pred_test[non_zero_mask]) / 
                                          self.y_test[non_zero_mask])) * 100
                else:
                    mape = np.nan
                
                # Cross-validation (on sample for speed)
                cv_sample_size = min(1000, len(self.X_train_scaled))
                cv_scores = cross_val_score(
                    model, 
                    self.X_train_scaled[:cv_sample_size], 
                    self.y_train[:cv_sample_size], 
                    cv=3, 
                    scoring='r2', 
                    n_jobs=-1
                )
                cv_mean = cv_scores.mean()
                cv_std = cv_scores.std()
                
                self.results[name] = {
                    'Train R²': train_r2,
                    'Test R²': test_r2,
                    'RMSE': test_rmse,
                    'MAE': test_mae,
                    'MAPE': mape,
                    'CV R² Mean': cv_mean,
                    'CV R² Std': cv_std,
                    'Overfitting': train_r2 - test_r2,
                    'model': model,
                    'predictions': y_pred_test
                }
                
                print(f"  ✓ Test R²: {test_r2:.4f} | RMSE: ${test_rmse:.2f} | MAE: ${test_mae:.2f} | MAPE: {mape:.2f}%")
                
            except Exception as e:
                print(f"  ✗ Error with {name}: {str(e)}")
                continue
        
        return self
    
    def display_results(self):
        """Display results as a table."""
        try:
            results_df = pd.DataFrame(self.results).T.drop(['model', 'predictions'], axis=1)
            results_df = results_df.sort_values('Test R²', ascending=False)
            
            print("\n" + "="*80)
            print("COMPARATIVE MODEL RESULTS")
            print("="*80)
            print(results_df.to_string())
            
            # Save results
            results_df.to_csv('model_comparison_results.csv')
            print("\n✓ Results saved: model_comparison_results.csv")
            
            return self
        except Exception as e:
            raise ModelTrainingException(f"Results display error: {str(e)}")
    
    def plot_model_comparison(self):
        """Visualize model comparison."""
        try:
            results_df = pd.DataFrame(self.results).T.drop(['model', 'predictions'], axis=1)
            results_df = results_df.sort_values('Test R²', ascending=False)
            
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # R² Score
            axes[0, 0].barh(results_df.index, results_df['Test R²'], color='steelblue')
            axes[0, 0].set_xlabel('R² Score')
            axes[0, 0].set_title('Test R² Score by Model (higher is better)')
            axes[0, 0].axvline(x=0, color='red', linestyle='--', linewidth=0.5)
            for i, v in enumerate(results_df['Test R²']):
                axes[0, 0].text(v + 0.01, i, f'{v:.3f}', va='center')
            
            # RMSE
            axes[0, 1].barh(results_df.index, results_df['RMSE'], color='coral')
            axes[0, 1].set_xlabel('RMSE ($)')
            axes[0, 1].set_title('RMSE by Model (lower is better)')
            for i, v in enumerate(results_df['RMSE']):
                axes[0, 1].text(v + 0.5, i, f'${v:.2f}', va='center')
            
            # MAE and MAPE
            x = np.arange(len(results_df))
            width = 0.35
            ax2 = axes[1, 0].twinx()
            axes[1, 0].bar(x - width/2, results_df['MAE'], width, label='MAE ($)', color='lightgreen', alpha=0.8)
            ax2.bar(x + width/2, results_df['MAPE'], width, label='MAPE (%)', color='orange', alpha=0.8)
            axes[1, 0].set_ylabel('MAE ($)', color='green')
            ax2.set_ylabel('MAPE (%)', color='orange')
            axes[1, 0].set_title('MAE and MAPE by Model')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels(results_df.index, rotation=45, ha='right')
            axes[1, 0].legend(loc='upper left')
            ax2.legend(loc='upper right')
            
            # Train vs Test R² (Overfitting check)
            x = np.arange(len(results_df))
            width = 0.35
            axes[1, 1].bar(x - width/2, results_df['Train R²'], width, label='Train R²', color='skyblue')
            axes[1, 1].bar(x + width/2, results_df['Test R²'], width, label='Test R²', color='darkblue')
            axes[1, 1].set_ylabel('R² Score')
            axes[1, 1].set_title('Train vs Test R² (Overfitting Check)')
            axes[1, 1].set_xticks(x)
            axes[1, 1].set_xticklabels(results_df.index, rotation=45, ha='right')
            axes[1, 1].legend()
            axes[1, 1].axhline(y=0, color='red', linestyle='--', linewidth=0.5)
            
            plt.tight_layout()
            plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
            print("✓ Chart saved: model_comparison.png")
            plt.show()
            
            return self
        except Exception as e:
            raise ModelTrainingException(f"Comparison chart error: {str(e)}")
    
    def plot_predictions(self):
        """Plot actual vs predicted values for best models."""
        try:
            # Get top 3 models
            sorted_models = sorted(
                self.results.items(), 
                key=lambda x: x[1]['Test R²'], 
                reverse=True
            )[:3]
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for idx, (name, result) in enumerate(sorted_models):
                y_pred = result['predictions']
                
                axes[idx].scatter(self.y_test, y_pred, alpha=0.5, s=20)
                axes[idx].plot([self.y_test.min(), self.y_test.max()], 
                              [self.y_test.min(), self.y_test.max()], 
                              'r--', lw=2, label='Perfect Prediction')
                axes[idx].set_xlabel('Actual Price ($)')
                axes[idx].set_ylabel('Predicted Price ($)')
                axes[idx].set_title(f'{name}\nR² = {result["Test R²"]:.4f}')
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('prediction_plots.png', dpi=300, bbox_inches='tight')
            print("✓ Prediction plots saved: prediction_plots.png")
            plt.show()
            
            return self
        except Exception as e:
            raise ModelTrainingException(f"Prediction plot error: {str(e)}")
    
    def plot_residuals(self):
        """Plot residuals for best model."""
        try:
            best_name = max(self.results, key=lambda x: self.results[x]['Test R²'])
            y_pred = self.results[best_name]['predictions']
            residuals = self.y_test - y_pred
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 5))
            
            # Residual plot
            axes[0].scatter(y_pred, residuals, alpha=0.5, s=20)
            axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
            axes[0].set_xlabel('Predicted Price ($)')
            axes[0].set_ylabel('Residuals ($)')
            axes[0].set_title(f'Residual Plot - {best_name}')
            axes[0].grid(True, alpha=0.3)
            
            # Residual distribution
            axes[1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
            axes[1].axvline(x=0, color='r', linestyle='--', lw=2)
            axes[1].set_xlabel('Residuals ($)')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Residual Distribution')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('residual_analysis.png', dpi=300, bbox_inches='tight')
            print("✓ Residual analysis saved: residual_analysis.png")
            plt.show()
            
            return self
        except Exception as e:
            raise ModelTrainingException(f"Residual plot error: {str(e)}")
    
    def get_feature_importance(self):
        """Get feature importance from best tree-based model."""
        try:
            # Find best tree-based model
            tree_models = ['Random Forest', 'Gradient Boosting', 'Decision Tree']
            best_tree_model = None
            best_r2 = -np.inf
            
            for name in tree_models:
                if name in self.results and self.results[name]['Test R²'] > best_r2:
                    best_r2 = self.results[name]['Test R²']
                    best_tree_model = name
            
            if best_tree_model and self.feature_names:
                model = self.results[best_tree_model]['model']
                
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    feat_imp = pd.DataFrame({
                        'feature': self.feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    print(f"\n" + "="*80)
                    print(f"FEATURE IMPORTANCE - {best_tree_model}")
                    print("="*80)
                    print(feat_imp.to_string(index=False))
                    
                    # Plot
                    plt.figure(figsize=(10, 8))
                    plt.barh(feat_imp['feature'][:15], feat_imp['importance'][:15])
                    plt.xlabel('Importance')
                    plt.title(f'Top 15 Feature Importance - {best_tree_model}')
                    plt.gca().invert_yaxis()
                    plt.tight_layout()
                    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
                    print("\n✓ Feature importance plot saved: feature_importance.png")
                    plt.show()
                    
                    return feat_imp
            
        except Exception as e:
            print(f"⚠ Could not extract feature importance: {str(e)}")
        
        return None
    
    def get_best_model(self):
        """Return the best model based on Test R²."""
        try:
            best_model_name = max(self.results, key=lambda x: self.results[x]['Test R²'])
            best_model = self.results[best_model_name]['model']
            best_score = self.results[best_model_name]['Test R²']
            
            print(f"\n" + "="*80)
            print(f"🏆 BEST MODEL: {best_model_name}")
            print(f"   Test R²: {best_score:.4f}")
            print(f"   RMSE: ${self.results[best_model_name]['RMSE']:.2f}")
            print(f"   MAE: ${self.results[best_model_name]['MAE']:.2f}")
            print(f"   MAPE: {self.results[best_model_name]['MAPE']:.2f}%")
            print("="*80)
            
            return best_model_name, best_model
        except Exception as e:
            raise ModelTrainingException(f"Best model selection error: {str(e)}")