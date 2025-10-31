"""
Neural Network Model for URL Credibility Prediction
Feed-forward neural network trained on URL and content features
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import json
from typing import Tuple, Dict, List
import os
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")


class CredibilityDataset(Dataset):
    """PyTorch Dataset for URL credibility data"""

    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class CredibilityNN(nn.Module):
    """
    Feed-forward Neural Network for URL Credibility Classification

    Architecture:
    - Input layer: All extracted features
    - Hidden layers: 3 layers with dropout for regularization
    - Output layer: 3 classes (low, medium, high credibility)
    """

    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 128, 64], num_classes: int = 3, dropout: float = 0.3):
        super(CredibilityNN, self).__init__()

        layers = []

        # Input layer
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(prev_dim, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CredibilityPredictor:
    """
    Wrapper class for training and using the credibility neural network
    """

    def __init__(self, model_path: str = "models/credibility_model.pth"):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.model_path = model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

    def prepare_data(self, df: pd.DataFrame, target_col: str = 'credibility_class') -> Tuple:
        """
        Prepare data for training

        Args:
            df: DataFrame with features and labels
            target_col: Column name for target labels

        Returns:
            X_train, X_val, X_test, y_train, y_val, y_test
        """
        # Identify feature columns (exclude metadata and target)
        exclude_cols = ['url', 'query', 'credibility_score', 'credibility_class',
                       'credibility_label', target_col]
        self.feature_names = [col for col in df.columns if col not in exclude_cols]

        # Extract features and labels
        X = df[self.feature_names].values
        y = df[target_col].values

        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

        # Check if we have enough samples for stratified split
        class_counts = pd.Series(y).value_counts()
        min_samples = class_counts.min()

        if min_samples < 2:
            print(f"⚠️  Warning: Class imbalance detected. Minimum samples in a class: {min_samples}")
            print(f"   Using simple split without stratification")
            print(f"   Class distribution: {class_counts.to_dict()}")

            # Simple split without stratification
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42
            )
        else:
            # Stratified split when we have enough samples
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.3, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )

        # Scale features
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        X_test = self.scaler.transform(X_test)

        print(f"📊 Data prepared:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Validation samples: {len(X_val)}")
        print(f"   Test samples: {len(X_test)}")
        print(f"   Features: {len(self.feature_names)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, df: pd.DataFrame, epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.001, target_col: str = 'credibility_class'):
        """
        Train the neural network

        Args:
            df: Training DataFrame
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate for optimizer
            target_col: Column name for target labels
        """
        # Check dataset size
        print(f"\n📊 Dataset Overview:")
        print(f"   Total samples: {len(df)}")
        if target_col in df.columns:
            class_dist = df[target_col].value_counts().sort_index()
            print(f"   Class distribution:")
            for cls, count in class_dist.items():
                print(f"      Class {cls}: {count} samples")

            # Warn if imbalanced
            min_samples = class_dist.min()
            if min_samples < 10:
                print(f"\n⚠️  Warning: Low sample count detected!")
                print(f"   For better results, aim for at least 30 samples per class")
                print(f"   Current minimum: {min_samples} samples")

        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test = self.prepare_data(df, target_col)

        # Create datasets
        train_dataset = CredibilityDataset(X_train, y_train)
        val_dataset = CredibilityDataset(X_val, y_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        # Initialize model
        input_dim = len(self.feature_names)
        self.model = CredibilityNN(input_dim).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        best_val_acc = 0.0
        train_losses = []
        val_accuracies = []

        print(f"\n🚀 Starting training for {epochs} epochs...")

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0

            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            self.model.eval()
            val_predictions = []
            val_labels = []

            with torch.no_grad():
                for features, labels in val_loader:
                    features = features.to(self.device)
                    outputs = self.model(features)
                    _, predicted = torch.max(outputs, 1)
                    val_predictions.extend(predicted.cpu().numpy())
                    val_labels.extend(labels.numpy())

            val_acc = accuracy_score(val_labels, val_predictions)
            val_accuracies.append(val_acc)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_train_loss:.4f}, Val Acc: {val_acc:.4f}")

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.save_model()

        print(f"\n✅ Training complete! Best validation accuracy: {best_val_acc:.4f}")

        # Final evaluation on test set
        test_dataset = CredibilityDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        test_predictions = []
        test_labels_list = []

        self.model.eval()
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.to(self.device)
                outputs = self.model(features)
                _, predicted = torch.max(outputs, 1)
                test_predictions.extend(predicted.cpu().numpy())
                test_labels_list.extend(labels.numpy())

        test_acc = accuracy_score(test_labels_list, test_predictions)
        print(f"\n📊 Test Set Performance:")
        print(f"   Accuracy: {test_acc:.4f}")

        # Only show classification report if we have samples from multiple classes
        unique_classes = len(set(test_labels_list))
        if unique_classes > 1:
            print(f"\n   Classification Report:")
            try:
                print(classification_report(test_labels_list, test_predictions,
                                            target_names=['Low', 'Medium', 'High'],
                                            zero_division=0))
            except Exception as e:
                print(f"   (Classification report unavailable: {str(e)})")
        else:
            print(f"\n   Note: Test set contains only {unique_classes} class(es)")

        return {
            'train_losses': train_losses,
            'val_accuracies': val_accuracies,
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        }

    def predict(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data

        Args:
            features: Feature array (n_samples, n_features)

        Returns:
            predictions: Class predictions (0=low, 1=medium, 2=high)
            probabilities: Class probabilities
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")

        self.model.eval()

        # Scale features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)

        with torch.no_grad():
            outputs = self.model(features_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)

        return predictions, probabilities

    def predict_url(self, url: str, feature_extractor, heuristic_scores: Dict = None) -> Dict:
        """
        Predict credibility for a single URL

        Args:
            url: URL to analyze
            feature_extractor: URLFeatureExtractor instance
            heuristic_scores: Optional existing heuristic scores

        Returns:
            Dictionary with prediction results
        """
        # Extract features
        features_dict = feature_extractor.extract_all_features(url, heuristic_scores)

        # Create feature vector in correct order
        feature_vector = np.array([features_dict.get(name, 0) for name in self.feature_names])
        feature_vector = feature_vector.reshape(1, -1)

        # Handle NaN
        feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)

        # Predict
        predictions, probabilities = self.predict(feature_vector)

        label_map = {0: 'low', 1: 'medium', 2: 'high'}

        return {
            'url': url,
            'predicted_class': int(predictions[0]),
            'predicted_label': label_map[predictions[0]],
            'confidence': float(probabilities[0][predictions[0]]),
            'probabilities': {
                'low': float(probabilities[0][0]),
                'medium': float(probabilities[0][1]),
                'high': float(probabilities[0][2])
            }
        }

    def save_model(self, path: str = None):
        """Save model, scaler, and feature names"""
        if path is None:
            path = self.model_path

        # Save model state
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'feature_names': self.feature_names,
            'input_dim': len(self.feature_names)
        }, path)

        # Save scaler
        scaler_path = path.replace('.pth', '_scaler.pkl')
        joblib.dump(self.scaler, scaler_path)

        print(f"💾 Model saved to {path}")

    def load_model(self, path: str = None):
        """Load trained model"""
        if path is None:
            path = self.model_path

        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load checkpoint
        checkpoint = torch.load(path, map_location=self.device)

        # Reconstruct model
        input_dim = checkpoint['input_dim']
        self.feature_names = checkpoint['feature_names']
        self.model = CredibilityNN(input_dim).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        # Load scaler
        scaler_path = path.replace('.pth', '_scaler.pkl')
        self.scaler = joblib.load(scaler_path)

        print(f"✅ Model loaded from {path}")


if __name__ == "__main__":
    # Example usage
    print("Neural Network Model for URL Credibility")
    print("=" * 50)
    print("\nThis module provides:")
    print("1. CredibilityNN - The neural network architecture")
    print("2. CredibilityPredictor - Training and prediction wrapper")
    print("\nUse train_model.py to train the model on your dataset")
