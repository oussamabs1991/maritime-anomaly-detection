"""
Ensemble learning models for vessel classification
"""
import gc
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from ..config import config
from .base_models import TraditionalMLModels
from .deep_learning import DeepLearningModels


class MetaFeatureGenerator:
    """Generate meta-features from base model predictions with consistent dimensionality"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.expected_models = ["rf", "lgb", "cnn", "lstm"]  # Define expected models
        self.max_classes = None

    def create_meta_features(
        self,
        models: Dict,
        X_features: np.ndarray,
        X_sequences: np.ndarray,
        expected_classes: int = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Create meta-features from base model predictions with consistent dimensionality

        Args:
            models: Dictionary of trained base models
            X_features: Feature data for traditional ML models
            X_sequences: Sequence data for deep learning models
            expected_classes: Expected number of classes for consistency

        Returns:
            Tuple of (meta_features_array, feature_names)
        """
        n_samples = len(X_features)

        # Determine number of classes
        if expected_classes is not None:
            num_classes = expected_classes
        elif self.max_classes is not None:
            num_classes = self.max_classes
        else:
            # Detect from first available model
            num_classes = None
            for name, model in models.items():
                try:
                    if name in ["cnn", "lstm"]:
                        if (
                            X_sequences is not None
                            and len(X_sequences) > 0
                            and hasattr(model, "output_shape")
                        ):
                            test_pred = model.predict(X_sequences[:1], verbose=0)
                            num_classes = test_pred.shape[1]
                            break
                    else:
                        if (
                            X_features is not None
                            and len(X_features) > 0
                            and hasattr(model, "n_classes_")
                        ):
                            num_classes = model.n_classes_
                            break
                except Exception:
                    continue

            if num_classes is None:
                num_classes = 6  # Default fallback

            self.max_classes = num_classes

        logger.info(f"Creating meta-features for {num_classes} classes from {len(models)} models")

        # Initialize meta-features with consistent shape
        total_features = len(self.expected_models) * num_classes
        meta_features_array = np.zeros((n_samples, total_features))
        meta_feature_names = []

        # Generate feature names for all expected models
        for model_name in self.expected_models:
            for i in range(num_classes):
                meta_feature_names.append(f"{model_name}_class_{i}")

        # Fill in predictions for available models
        feature_idx = 0
        for model_name in self.expected_models:
            if model_name in models:
                model = models[model_name]
                try:
                    if model_name in ["cnn", "lstm"]:
                        # Deep learning models - use sequence data
                        if X_sequences is not None and len(X_sequences) > 0:
                            logger.info(f"Generating predictions from {model_name} model")
                            preds = model.predict(X_sequences, verbose=0)
                        else:
                            logger.warning(f"No sequence data for {model_name} - using zeros")
                            preds = np.zeros((n_samples, num_classes))
                    else:
                        # Traditional ML models - use feature data
                        if X_features is not None and len(X_features) > 0:
                            logger.info(f"Generating predictions from {model_name} model")
                            preds = model.predict_proba(X_features)
                        else:
                            logger.warning(f"No feature data for {model_name} - using zeros")
                            preds = np.zeros((n_samples, num_classes))

                    # Ensure consistent number of classes
                    if preds.shape[1] != num_classes:
                        logger.warning(
                            f"Model {model_name} has {preds.shape[1]} classes, "
                            f"expected {num_classes}"
                        )
                        if preds.shape[1] < num_classes:
                            # Pad with zeros
                            padding = np.zeros((preds.shape[0], num_classes - preds.shape[1]))
                            preds = np.hstack([preds, padding])
                        else:
                            # Truncate
                            preds = preds[:, :num_classes]

                    # Assign to meta-features array
                    meta_features_array[:, feature_idx : feature_idx + num_classes] = preds

                except Exception as e:
                    logger.error(f"Error generating meta-features from {model_name}: {e}")
                    # Leave as zeros
            else:
                logger.warning(f"Model {model_name} not available - using zeros")

            feature_idx += num_classes

        logger.info(f"Generated consistent meta-features shape: {meta_features_array.shape}")

        return meta_features_array, meta_feature_names


class StackingEnsemble:
    """Stacking ensemble with robust cross-validation"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config
        self.dl_models = DeepLearningModels(config_obj)
        self.ml_models = TraditionalMLModels(config_obj)
        self.meta_generator = MetaFeatureGenerator(config_obj)
        self.base_models = None
        self.meta_learner = None
        self.label_encoder = None

    def train_base_models(
        self,
        X_features: np.ndarray,
        y: np.ndarray,
        X_features_val: np.ndarray,
        y_val: np.ndarray,
        X_sequences: np.ndarray,
        X_sequences_val: np.ndarray,
        class_weights: Dict,
    ) -> Dict:
        """
        Train base models (both traditional ML and deep learning)
        """
        models = {}

        logger.info("Training base models")

        # Train traditional ML models (always try these)
        if X_features is not None and len(X_features) > 0:
            try:
                traditional_models = self.ml_models.train_all_traditional_models(X_features, y)
                models.update(traditional_models)
                logger.info(f"Successfully trained {len(traditional_models)} traditional ML models")
            except Exception as e:
                logger.error(f"Failed to train traditional ML models: {e}")

        # Train deep learning models (may fail due to small data or other issues)
        if X_sequences is not None and len(X_sequences) > 0:
            try:
                num_classes = len(np.unique(y))
                dl_models = self.dl_models.create_and_train_models(
                    X_sequences, y, X_sequences_val, y_val, num_classes, class_weights
                )
                if dl_models:
                    models.update(dl_models)
                    logger.info(f"Successfully trained {len(dl_models)} deep learning models")
                else:
                    logger.warning("No deep learning models were trained successfully")
            except Exception as e:
                logger.error(f"Failed to train deep learning models: {e}")

        logger.info(f"Total trained models: {len(models)} - {list(models.keys())}")

        return models

    def cross_validate_ensemble(
        self, X_features: np.ndarray, y: np.ndarray, X_sequences: np.ndarray, n_folds: int = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        Perform cross-validation to generate out-of-fold meta-features with consistent
        dimensionality
        """
        n_folds = n_folds or (3 if self.config.TEST_MODE else self.config.CV_FOLDS)

        logger.info(f"Starting {n_folds}-fold cross-validation for ensemble")

        # Determine maximum number of classes across all data
        unique_classes = np.unique(y)
        max_classes = len(unique_classes)
        self.meta_generator.max_classes = max_classes

        logger.info(f"Maximum classes detected: {max_classes}")

        # Initialize out-of-fold predictions array with consistent size
        expected_models = self.meta_generator.expected_models
        total_meta_features = len(expected_models) * max_classes
        oof_meta_features = np.zeros((len(X_features), total_meta_features))

        # Create stratified k-fold splitter
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.config.RANDOM_STATE)

        final_base_models = None

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_features, y)):
            logger.info(f"Processing fold {fold + 1}/{n_folds}")

            # Split data for this fold
            X_feat_train = X_features[train_idx]
            X_feat_val = X_features[val_idx]
            y_train_fold = y[train_idx]
            y_val_fold = y[val_idx]
            X_seq_train = X_sequences[train_idx] if X_sequences is not None else None
            X_seq_val = X_sequences[val_idx] if X_sequences is not None else None

            # Compute class weights for this fold
            fold_classes = np.unique(y_train_fold)
            fold_weights = compute_class_weight("balanced", classes=fold_classes, y=y_train_fold)
            fold_class_weights = dict(zip(fold_classes, fold_weights))

            # Train base models for this fold
            fold_models = self.train_base_models(
                X_feat_train,
                y_train_fold,
                X_feat_val,
                y_val_fold,
                X_seq_train,
                X_seq_val,
                fold_class_weights,
            )

            # Generate meta-features for validation set with consistent dimensions
            if fold_models:
                meta_val, _ = self.meta_generator.create_meta_features(
                    fold_models, X_feat_val, X_seq_val, expected_classes=max_classes
                )

                # Store validation predictions in consistent format
                oof_meta_features[val_idx] = meta_val
            else:
                logger.warning(f"No models trained in fold {fold + 1}")

            # Keep models from last fold for final ensemble
            if fold_models:
                final_base_models = fold_models

            # Clear memory
            del fold_models
            if hasattr(tf.keras, "backend"):
                tf.keras.backend.clear_session()
            gc.collect()

        if final_base_models is None:
            logger.error("No models were trained successfully in any fold")
            return np.zeros((len(X_features), 0)), {}

        logger.info(
            f"Cross-validation complete. OOF meta-features shape: {oof_meta_features.shape}"
        )

        return oof_meta_features, final_base_models

    def fit(
        self, X_features: np.ndarray, y: np.ndarray, X_sequences: np.ndarray = None
    ) -> "StackingEnsemble":
        """
        Fit the stacking ensemble with robust cross-validation
        """
        logger.info("Fitting stacking ensemble with robust cross-validation")

        # Encode labels if they're strings
        if y.dtype == "object" or isinstance(y[0], str):
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(y)
        else:
            y_encoded = y

        # Generate out-of-fold meta-features
        oof_meta_features, self.base_models = self.cross_validate_ensemble(
            X_features, y_encoded, X_sequences
        )

        if oof_meta_features.shape[1] == 0:
            raise ValueError("No meta-features generated. Cannot train ensemble.")

        # Filter out all-zero columns (models that never trained successfully)
        non_zero_cols = np.any(oof_meta_features != 0, axis=0)
        if np.any(non_zero_cols):
            oof_meta_features = oof_meta_features[:, non_zero_cols]
            logger.info(f"Filtered meta-features to shape: {oof_meta_features.shape}")

        # Train meta-learner on out-of-fold predictions
        logger.info("Training meta-learner")
        self.meta_learner = self.ml_models.train_meta_learner(oof_meta_features, y_encoded)

        if self.meta_learner is None:
            raise ValueError("Failed to train meta-learner")

        logger.info("Stacking ensemble training complete")

        return self

    def predict(self, X_features: np.ndarray, X_sequences: np.ndarray = None) -> np.ndarray:
        """
        Make predictions using the stacking ensemble
        """
        if self.base_models is None or self.meta_learner is None:
            raise ValueError("Ensemble must be fitted before making predictions")

        logger.info("Making ensemble predictions")

        # Generate meta-features from base models
        meta_features, _ = self.meta_generator.create_meta_features(
            self.base_models,
            X_features,
            X_sequences,
            expected_classes=self.meta_generator.max_classes,
        )

        # Filter to match training meta-features
        non_zero_cols = np.any(meta_features != 0, axis=0)
        if np.any(non_zero_cols):
            meta_features = meta_features[:, non_zero_cols]

        # Make final predictions using meta-learner
        predictions = self.meta_learner.predict(meta_features)

        # Decode labels if necessary
        if self.label_encoder is not None:
            predictions = self.label_encoder.inverse_transform(predictions)

        return predictions

    def predict_proba(self, X_features: np.ndarray, X_sequences: np.ndarray = None) -> np.ndarray:
        """
        Get prediction probabilities from the ensemble
        """
        if self.base_models is None or self.meta_learner is None:
            raise ValueError("Ensemble must be fitted before making predictions")

        # Generate meta-features from base models
        meta_features, _ = self.meta_generator.create_meta_features(
            self.base_models,
            X_features,
            X_sequences,
            expected_classes=self.meta_generator.max_classes,
        )

        # Filter to match training meta-features
        non_zero_cols = np.any(meta_features != 0, axis=0)
        if np.any(non_zero_cols):
            meta_features = meta_features[:, non_zero_cols]

        # Get probabilities from meta-learner
        probabilities = self.meta_learner.predict_proba(meta_features)

        return probabilities

    def get_base_model_importance(self) -> pd.DataFrame:
        """
        Get the importance of base models in the meta-learner
        """
        if self.meta_learner is None:
            return pd.DataFrame()

        # Get meta-learner coefficients
        if hasattr(self.meta_learner, "coef_"):
            coef = np.abs(self.meta_learner.coef_)

            # Create feature names based on expected models
            feature_names = []
            max_classes = self.meta_generator.max_classes or 6
            for model_name in self.meta_generator.expected_models:
                for i in range(max_classes):
                    feature_names.append(f"{model_name}_class_{i}")

            # Create importance DataFrame
            if len(coef.shape) > 1:
                # Multi-class case
                n_features = min(len(feature_names), coef.shape[1])
                importance_df = pd.DataFrame(
                    coef.T[:n_features],
                    index=feature_names[:n_features],
                    columns=[f"class_{i}" for i in range(coef.shape[0])],
                )
                importance_df["mean_importance"] = importance_df.mean(axis=1)
            else:
                # Binary case
                n_features = min(len(feature_names), len(coef))
                importance_df = pd.DataFrame(
                    {"importance": coef[:n_features], "feature": feature_names[:n_features]}
                ).set_index("feature")
                importance_df["mean_importance"] = importance_df["importance"]

            return importance_df.sort_values("mean_importance", ascending=False)

        return pd.DataFrame()


def test_ensemble():
    """Test the stacking ensemble with sample data"""
    # Create sample data
    n_samples = 100
    n_features = 10
    sequence_length = 20
    n_seq_features = 4
    n_classes = 3

    np.random.seed(42)
    X_features = np.random.randn(n_samples, n_features)
    X_sequences = np.random.randn(n_samples, sequence_length, n_seq_features)
    y = np.random.randint(0, n_classes, n_samples)

    logger.info("Testing stacking ensemble")
    logger.info(f"Features shape: {X_features.shape}")
    logger.info(f"Sequences shape: {X_sequences.shape}")
    logger.info(f"Labels shape: {y.shape}")

    # Create and fit ensemble
    ensemble = StackingEnsemble()

    try:
        ensemble.fit(X_features, y, X_sequences)
        logger.info("Ensemble fitted successfully")

        # Test predictions
        predictions = ensemble.predict(X_features, X_sequences)
        probabilities = ensemble.predict_proba(X_features, X_sequences)

        logger.info(f"Predictions shape: {predictions.shape}")
        logger.info(f"Probabilities shape: {probabilities.shape}")

        # Test importance
        importance_df = ensemble.get_base_model_importance()
        if not importance_df.empty:
            logger.info("Base model importance:")
            logger.info(importance_df.head())

        return ensemble

    except Exception as e:
        logger.error(f"Ensemble test failed: {e}")
        return None


if __name__ == "__main__":
    test_ensemble()
