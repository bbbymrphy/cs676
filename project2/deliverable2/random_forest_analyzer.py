"""

Comprehensive Random Forest Analysis Script
Performs complete machine learning analysis including:
- Data loading and exploration
- Feature engineering and preprocessing
- Model training (Classification & Regression)
- Hyperparameter tuning
- Feature importance analysis
- Model evaluation and visualization
- Prediction on new data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Models
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)

# Visualization
from sklearn.tree import plot_tree

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)


class RandomForestAnalyzer:
    """Complete Random Forest analysis pipeline for classification and regression"""

    def __init__(self, task='auto', random_state=42):
        """
        Initialize the analyzer

        Parameters:
        -----------
        task : str
            'classification', 'regression', or 'auto' (auto-detect based on target)
        random_state : int
            Random seed for reproducibility
        """
        self.task = task
        self.random_state = random_state
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        self.model = None
        self.predictions = None
        self.probabilities = None
        self.feature_importance = None
        self.metrics = {}
        self.preprocessing_info = {}
        self.label_encoder = None
        self.scaler = None

    def load_data(self, data, target_column=None):
        """Load and prepare data"""
        print("=" * 80)
        print("LOADING DATA")
        print("=" * 80)

        if isinstance(data, str):
            self.data = pd.read_csv(data)
            print(f"Loaded data from: {data}")
        else:
            self.data = data.copy()

        print(f"Data shape: {self.data.shape}")
        print(f"\nColumns: {list(self.data.columns)}")
        print(f"\nFirst few rows:\n{self.data.head()}")

        # Auto-detect target if not specified
        if target_column is None:
            # Use last column as target
            target_column = self.data.columns[-1]
            print(f"\nAuto-detected target column: {target_column}")

        self.target_name = target_column

        # Separate features and target
        self.X = self.data.drop(columns=[target_column])
        self.y = self.data[target_column]
        self.feature_names = list(self.X.columns)

        # Auto-detect task type
        if self.task == 'auto':
            unique_values = self.y.nunique()
            if unique_values < 20 or self.y.dtype == 'object':
                self.task = 'classification'
            else:
                self.task = 'regression'
            print(f"\nAuto-detected task: {self.task}")

        print(f"\nTask type: {self.task}")
        print(f"Features: {len(self.feature_names)}")
        print(f"Target: {self.target_name}")
        print(f"Target unique values: {self.y.nunique()}")

        return self

    def generate_sample_data(self, n_samples=1000, n_features=10, task='classification'):
        """Generate sample data for testing"""
        print("=" * 80)
        print("GENERATING SAMPLE DATA")
        print("=" * 80)

        self.task = task

        from sklearn.datasets import make_classification, make_regression

        if task == 'classification':
            X, y = make_classification(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(2, n_features // 2),
                n_redundant=max(1, n_features // 4),
                n_classes=2,
                random_state=self.random_state
            )
            feature_names = [f'feature_{i}' for i in range(n_features)]
            self.data = pd.DataFrame(X, columns=feature_names)
            self.data['target'] = y

        else:  # regression
            X, y = make_regression(
                n_samples=n_samples,
                n_features=n_features,
                n_informative=max(2, n_features // 2),
                noise=10,
                random_state=self.random_state
            )
            feature_names = [f'feature_{i}' for i in range(n_features)]
            self.data = pd.DataFrame(X, columns=feature_names)
            self.data['target'] = y

        self.X = self.data.drop(columns=['target'])
        self.y = self.data['target']
        self.feature_names = list(self.X.columns)
        self.target_name = 'target'

        print(f"Generated {n_samples} samples with {n_features} features")
        print(f"Task: {task}")

        return self

    def explore_data(self):
        """Explore and visualize the dataset"""
        print("\n" + "=" * 80)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 80)

        # Summary statistics
        print("\nDataset Info:")
        print(f"  Samples: {len(self.data)}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Target: {self.target_name}")

        print("\nFeature Statistics:")
        print(self.X.describe())

        print("\nTarget Statistics:")
        print(self.y.describe())

        print("\nMissing Values:")
        missing = self.data.isnull().sum()
        if missing.sum() > 0:
            print(missing[missing > 0])
        else:
            print("No missing values")

        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Target distribution
        if self.task == 'classification':
            axes[0, 0].bar(self.y.value_counts().index, self.y.value_counts().values)
            axes[0, 0].set_title('Target Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Class')
            axes[0, 0].set_ylabel('Count')
        else:
            axes[0, 0].hist(self.y, bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].set_title('Target Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Value')
            axes[0, 0].set_ylabel('Frequency')

        # Feature correlation heatmap (top features)
        n_features_to_show = min(10, len(self.feature_names))
        corr = self.X.iloc[:, :n_features_to_show].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', ax=axes[0, 1],
                   cbar_kws={'label': 'Correlation'})
        axes[0, 1].set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')

        # Missing values plot
        missing_data = self.data.isnull().sum()
        if missing_data.sum() > 0:
            missing_data[missing_data > 0].plot(kind='bar', ax=axes[1, 0])
            axes[1, 0].set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
            axes[1, 0].set_ylabel('Count')
        else:
            axes[1, 0].text(0.5, 0.5, 'No Missing Values',
                          ha='center', va='center', transform=axes[1, 0].transAxes, fontsize=14)
            axes[1, 0].set_title('Missing Values', fontsize=14, fontweight='bold')

        # Data types
        dtype_counts = self.X.dtypes.value_counts()
        axes[1, 1].pie(dtype_counts.values, labels=dtype_counts.index, autopct='%1.1f%%')
        axes[1, 1].set_title('Feature Data Types', fontsize=14, fontweight='bold')

        plt.tight_layout()
        plt.savefig('rf_01_exploratory_analysis.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: rf_01_exploratory_analysis.png")
        plt.show()

        return self

    def preprocess_data(self, handle_missing='mean', scale=True, encode_categorical=True):
        """Preprocess features"""
        print("\n" + "=" * 80)
        print("DATA PREPROCESSING")
        print("=" * 80)

        X_processed = self.X.copy()

        # Handle missing values
        if X_processed.isnull().sum().sum() > 0:
            print(f"\nHandling missing values with strategy: {handle_missing}")
            if handle_missing in ['mean', 'median']:
                imputer = SimpleImputer(strategy=handle_missing)
                numeric_cols = X_processed.select_dtypes(include=[np.number]).columns
                X_processed[numeric_cols] = imputer.fit_transform(X_processed[numeric_cols])

            self.preprocessing_info['imputer'] = imputer

        # Encode categorical variables
        categorical_cols = X_processed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0 and encode_categorical:
            print(f"\nEncoding categorical features: {list(categorical_cols)}")
            # Use label encoding for simplicity
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            self.preprocessing_info['categorical_cols'] = categorical_cols

        # Encode target if classification
        if self.task == 'classification' and self.y.dtype == 'object':
            print("\nEncoding target variable")
            self.label_encoder = LabelEncoder()
            self.y = pd.Series(self.label_encoder.fit_transform(self.y), index=self.y.index)
            print(f"Classes: {self.label_encoder.classes_}")

        # Scale features
        if scale:
            print("\nScaling features")
            self.scaler = StandardScaler()
            X_processed = pd.DataFrame(
                self.scaler.fit_transform(X_processed),
                columns=X_processed.columns,
                index=X_processed.index
            )
            self.preprocessing_info['scaler'] = self.scaler

        self.X = X_processed
        print("\nPreprocessing complete!")
        print(f"Final feature shape: {self.X.shape}")

        return self

    def split_data(self, test_size=0.2, stratify=True):
        """Split data into training and testing sets"""
        print("\n" + "=" * 80)
        print("TRAIN-TEST SPLIT")
        print("=" * 80)

        stratify_param = self.y if (stratify and self.task == 'classification') else None

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=self.random_state,
            stratify=stratify_param
        )

        print(f"\nTrain size: {len(self.X_train)} ({(1-test_size)*100:.0f}%)")
        print(f"Test size: {len(self.X_test)} ({test_size*100:.0f}%)")

        if self.task == 'classification':
            print(f"\nTrain class distribution:\n{self.y_train.value_counts()}")
            print(f"\nTest class distribution:\n{self.y_test.value_counts()}")

        return self

    def train_model(self, n_estimators=100, max_depth=None, min_samples_split=2,
                   min_samples_leaf=1, max_features='sqrt', **kwargs):
        """Train Random Forest model"""
        print("\n" + "=" * 80)
        print("TRAINING RANDOM FOREST MODEL")
        print("=" * 80)

        print(f"\nTask: {self.task}")
        print(f"Parameters:")
        print(f"  n_estimators: {n_estimators}")
        print(f"  max_depth: {max_depth}")
        print(f"  min_samples_split: {min_samples_split}")
        print(f"  min_samples_leaf: {min_samples_leaf}")
        print(f"  max_features: {max_features}")

        if self.task == 'classification':
            self.model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
                **kwargs
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=self.random_state,
                **kwargs
            )

        print("\nTraining model...")
        self.model.fit(self.X_train, self.y_train)
        print("Training complete!")

        # Get predictions
        self.predictions = self.model.predict(self.X_test)

        if self.task == 'classification':
            self.probabilities = self.model.predict_proba(self.X_test)

        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)

        print("\nTop 10 Important Features:")
        print(self.feature_importance.head(10))

        return self

    def evaluate_model(self):
        """Evaluate model performance"""
        print("\n" + "=" * 80)
        print("MODEL EVALUATION")
        print("=" * 80)

        if self.task == 'classification':
            self._evaluate_classification()
        else:
            self._evaluate_regression()

        return self

    def _evaluate_classification(self):
        """Evaluate classification model"""
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, self.predictions)

        # Handle binary vs multiclass
        average = 'binary' if len(np.unique(self.y)) == 2 else 'weighted'

        precision = precision_score(self.y_test, self.predictions, average=average, zero_division=0)
        recall = recall_score(self.y_test, self.predictions, average=average, zero_division=0)
        f1 = f1_score(self.y_test, self.predictions, average=average, zero_division=0)

        self.metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }

        print("\nClassification Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(self.y_test, self.predictions))

        # Confusion Matrix
        cm = confusion_matrix(self.y_test, self.predictions)

        # Visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Confusion Matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # ROC Curve (for binary classification)
        if len(np.unique(self.y)) == 2:
            fpr, tpr, _ = roc_curve(self.y_test, self.probabilities[:, 1])
            roc_auc = auc(fpr, tpr)

            axes[1].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC curve (AUC = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('ROC Curve', fontsize=14, fontweight='bold')
            axes[1].legend(loc="lower right")
            axes[1].grid(True, alpha=0.3)

            self.metrics['roc_auc'] = roc_auc
        else:
            axes[1].text(0.5, 0.5, 'ROC Curve\n(Binary Classification Only)',
                        ha='center', va='center', transform=axes[1].transAxes, fontsize=12)

        plt.tight_layout()
        plt.savefig('rf_02_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: rf_02_evaluation.png")
        plt.show()

    def _evaluate_regression(self):
        """Evaluate regression model"""
        # Calculate metrics
        mse = mean_squared_error(self.y_test, self.predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(self.y_test, self.predictions)
        r2 = r2_score(self.y_test, self.predictions)

        # MAPE
        mape = mean_absolute_percentage_error(self.y_test, self.predictions) * 100

        self.metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape
        }

        print("\nRegression Metrics:")
        print(f"  MSE:   {mse:.4f}")
        print(f"  RMSE:  {rmse:.4f}")
        print(f"  MAE:   {mae:.4f}")
        print(f"  R²:    {r2:.4f}")
        print(f"  MAPE:  {mape:.2f}%")

        # Visualizations
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Actual vs Predicted
        axes[0].scatter(self.y_test, self.predictions, alpha=0.5)
        axes[0].plot([self.y_test.min(), self.y_test.max()],
                    [self.y_test.min(), self.y_test.max()],
                    'r--', lw=2)
        axes[0].set_xlabel('Actual Values')
        axes[0].set_ylabel('Predicted Values')
        axes[0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)

        # Residuals
        residuals = self.y_test - self.predictions
        axes[1].scatter(self.predictions, residuals, alpha=0.5)
        axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1].set_xlabel('Predicted Values')
        axes[1].set_ylabel('Residuals')
        axes[1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('rf_02_evaluation.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: rf_02_evaluation.png")
        plt.show()

    def plot_feature_importance(self, top_n=20):
        """Plot feature importance"""
        print("\n" + "=" * 80)
        print("FEATURE IMPORTANCE")
        print("=" * 80)

        top_features = self.feature_importance.head(top_n)

        plt.figure(figsize=(12, max(6, len(top_features) * 0.3)))
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()

        plt.savefig('rf_03_feature_importance.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: rf_03_feature_importance.png")
        plt.show()

        return self

    def cross_validate(self, cv=5):
        """Perform cross-validation"""
        print("\n" + "=" * 80)
        print("CROSS-VALIDATION")
        print("=" * 80)

        print(f"\nPerforming {cv}-fold cross-validation...")

        if self.task == 'classification':
            scoring = 'accuracy'
        else:
            scoring = 'r2'

        scores = cross_val_score(self.model, self.X_train, self.y_train,
                                cv=cv, scoring=scoring)

        print(f"\nCross-validation scores ({scoring}):")
        for i, score in enumerate(scores, 1):
            print(f"  Fold {i}: {score:.4f}")

        print(f"\nMean: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")

        self.metrics[f'cv_{scoring}_mean'] = scores.mean()
        self.metrics[f'cv_{scoring}_std'] = scores.std()

        return self

    def hyperparameter_tuning(self, param_grid=None, cv=5, n_jobs=-1):
        """Perform hyperparameter tuning with GridSearchCV"""
        print("\n" + "=" * 80)
        print("HYPERPARAMETER TUNING")
        print("=" * 80)

        if param_grid is None:
            # Default parameter grid
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

        print(f"\nSearching over {len(param_grid)} parameters...")
        print(f"Parameter grid: {param_grid}")

        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=cv,
            scoring='accuracy' if self.task == 'classification' else 'r2',
            n_jobs=n_jobs,
            verbose=1
        )

        grid_search.fit(self.X_train, self.y_train)

        print("\nBest parameters:")
        for param, value in grid_search.best_params_.items():
            print(f"  {param}: {value}")

        print(f"\nBest score: {grid_search.best_score_:.4f}")

        # Update model with best parameters
        self.model = grid_search.best_estimator_

        # Re-evaluate
        self.predictions = self.model.predict(self.X_test)
        if self.task == 'classification':
            self.probabilities = self.model.predict_proba(self.X_test)

        return self

    def plot_tree(self, tree_index=0, max_depth=3):
        """Visualize a single decision tree from the forest"""
        print("\n" + "=" * 80)
        print(f"VISUALIZING TREE {tree_index}")
        print("=" * 80)

        plt.figure(figsize=(20, 10))
        plot_tree(
            self.model.estimators_[tree_index],
            feature_names=self.feature_names,
            filled=True,
            max_depth=max_depth,
            fontsize=10
        )
        plt.title(f'Decision Tree {tree_index} (max_depth={max_depth})',
                 fontsize=14, fontweight='bold')
        plt.tight_layout()

        plt.savefig(f'rf_04_tree_{tree_index}.png', dpi=300, bbox_inches='tight')
        print(f"\nSaved plot: rf_04_tree_{tree_index}.png")
        plt.show()

        return self

    def predict_new_data(self, new_data):
        """Make predictions on new data"""
        print("\n" + "=" * 80)
        print("MAKING PREDICTIONS")
        print("=" * 80)

        if isinstance(new_data, str):
            new_data = pd.read_csv(new_data)

        # Apply same preprocessing
        if self.scaler:
            new_data = pd.DataFrame(
                self.scaler.transform(new_data),
                columns=new_data.columns
            )

        predictions = self.model.predict(new_data)

        if self.task == 'classification':
            probabilities = self.model.predict_proba(new_data)

            if self.label_encoder:
                predictions = self.label_encoder.inverse_transform(predictions)

            results = pd.DataFrame({
                'prediction': predictions,
                'probability': probabilities.max(axis=1)
            })
        else:
            results = pd.DataFrame({
                'prediction': predictions
            })

        print(f"\nMade predictions for {len(new_data)} samples")
        print(f"\nFirst few predictions:\n{results.head()}")

        return results

    def generate_report(self, filename='random_forest_report.txt'):
        """Generate comprehensive analysis report"""
        print("\n" + "=" * 80)
        print("GENERATING REPORT")
        print("=" * 80)

        with open(filename, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("RANDOM FOREST ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")

            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write(f"Task: {self.task}\n")
            f.write(f"Total samples: {len(self.data)}\n")
            f.write(f"Number of features: {len(self.feature_names)}\n")
            f.write(f"Target variable: {self.target_name}\n\n")

            # Model info
            f.write("MODEL INFORMATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Algorithm: Random Forest {self.task.title()}\n")
            f.write(f"Number of trees: {self.model.n_estimators}\n")
            f.write(f"Max depth: {self.model.max_depth}\n")
            f.write(f"Random state: {self.random_state}\n\n")

            # Performance metrics
            f.write("PERFORMANCE METRICS\n")
            f.write("-" * 80 + "\n")
            for metric, value in self.metrics.items():
                f.write(f"  {metric}: {value:.4f}\n")
            f.write("\n")

            # Top features
            f.write("TOP 10 IMPORTANT FEATURES\n")
            f.write("-" * 80 + "\n")
            for idx, row in self.feature_importance.head(10).iterrows():
                f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
            f.write("\n")

        print(f"\nReport saved: {filename}")

        return self


def main():
    """Main execution function with example usage"""
    print("\n" + "=" * 80)
    print("COMPREHENSIVE RANDOM FOREST ANALYSIS")
    print("=" * 80)

    # Initialize analyzer
    analyzer = RandomForestAnalyzer(task='auto', random_state=42)

    # Option 1: Generate sample data for classification
    print("\nGenerating sample classification data...")
    analyzer.generate_sample_data(n_samples=1000, n_features=20, task='classification')

    # Option 2: Load your own data (uncomment and modify)
    # analyzer.load_data('your_data.csv', target_column='target')

    # Run complete analysis pipeline
    (analyzer
        .explore_data()
        .preprocess_data(handle_missing='mean', scale=True)
        .split_data(test_size=0.2)
        .train_model(n_estimators=100, max_depth=10)
        .evaluate_model()
        .plot_feature_importance(top_n=15)
        .cross_validate(cv=5)
        .plot_tree(tree_index=0, max_depth=3)
        .generate_report('random_forest_report.txt')
    )

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - rf_01_exploratory_analysis.png")
    print("  - rf_02_evaluation.png")
    print("  - rf_03_feature_importance.png")
    print("  - rf_04_tree_0.png")
    print("  - random_forest_report.txt")


if __name__ == "__main__":
    main()