"""
Traditional machine learning models for vessel classification
"""
import numpy as np
import pandas as pd
from imblearn.ensemble import BalancedRandomForestClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.linear_model import LogisticRegression

from ..config import LGB_CONFIG, META_LEARNER_CONFIG, RF_CONFIG, config


class TraditionalMLModels:
    """Factory class for creating traditional ML models"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def create_random_forest(self, test_mode: bool = None) -> BalancedRandomForestClassifier:
        """
        Create Random Forest classifier with balanced sampling

        Args:
            test_mode: Whether to create simplified model for testing

        Returns:
            Random Forest classifier
        """
        test_mode = test_mode if test_mode is not None else self.config.TEST_MODE

        params = RF_CONFIG.params.copy()

        if test_mode:
            # Simplified parameters for testing
            params.update(
                {
                    "n_estimators": self.config.TEST_TREES,
                    "max_depth": self.config.RF_TEST_MAX_DEPTH,
                }
            )
        else:
            # Full parameters
            params.update(
                {
                    "n_estimators": self.config.N_ESTIMATORS,
                    "max_depth": self.config.RF_MAX_DEPTH,
                }
            )

        params["random_state"] = self.config.RANDOM_STATE

        logger.info(f"Creating Random Forest ({'TEST' if test_mode else 'FULL'} mode)")
        logger.info(f"Parameters: {params}")

        return BalancedRandomForestClassifier(**params)

    def create_lightgbm(self, test_mode: bool = None) -> LGBMClassifier:
        """
        Create LightGBM classifier

        Args:
            test_mode: Whether to create simplified model for testing

        Returns:
            LightGBM classifier
        """
        test_mode = test_mode if test_mode is not None else self.config.TEST_MODE

        params = LGB_CONFIG.params.copy()

        if test_mode:
            # Simplified parameters for testing
            params.update(
                {
                    "n_estimators": self.config.TEST_TREES,
                    "max_depth": self.config.LGB_TEST_MAX_DEPTH,
                    "learning_rate": self.config.LGB_TEST_LEARNING_RATE,
                }
            )
        else:
            # Full parameters
            params.update(
                {
                    "n_estimators": self.config.N_ESTIMATORS,
                    "max_depth": self.config.LGB_MAX_DEPTH,
                    "learning_rate": self.config.LGB_LEARNING_RATE,
                }
            )

        params["random_state"] = self.config.RANDOM_STATE

        logger.info(f"Creating LightGBM ({'TEST' if test_mode else 'FULL'} mode)")
        logger.info(f"Parameters: {params}")

        return LGBMClassifier(**params)

    def create_meta_learner(self) -> LogisticRegression:
        """
        Create meta-learner for ensemble stacking

        Returns:
            Logistic Regression meta-learner
        """
        params = META_LEARNER_CONFIG.params.copy()
        params["random_state"] = self.config.RANDOM_STATE

        logger.info("Creating meta-learner (Logistic Regression)")
        logger.info(f"Parameters: {params}")

        return LogisticRegression(**params)

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> BalancedRandomForestClassifier:
        """
        Train Random Forest model

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained Random Forest model
        """
        if len(X_train) == 0 or len(X_train) != len(y_train):
            logger.error("Invalid training data for Random Forest")
            return None

        logger.info(
            f"Training Random Forest with {len(X_train)} samples, {X_train.shape[1]} features"
        )

        try:
            rf_model = self.create_random_forest()
            rf_model.fit(X_train, y_train)

            logger.info("Random Forest training completed successfully")

            # Fixed: Compute feature importance separately
            importances = rf_model.feature_importances_
            top_5 = sorted(
                zip(range(len(importances)), importances), key=lambda x: x[1], reverse=True
            )[:5]
            logger.info(f"Feature importance (top 5): {top_5}")

            return rf_model

        except Exception as e:
            logger.error(f"Error training Random Forest: {e}")
            return None

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> LGBMClassifier:
        """
        Train LightGBM model

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Trained LightGBM model
        """
        if len(X_train) == 0 or len(X_train) != len(y_train):
            logger.error("Invalid training data for LightGBM")
            return None

        logger.info(f"Training LightGBM with {len(X_train)} samples, {X_train.shape[1]} features")

        try:
            lgb_model = self.create_lightgbm()
            lgb_model.fit(X_train, y_train)

            logger.info("LightGBM training completed successfully")

            # Fixed: Compute feature importance separately
            importances = lgb_model.feature_importances_
            top_5 = sorted(
                zip(range(len(importances)), importances), key=lambda x: x[1], reverse=True
            )[:5]
            logger.info(f"Feature importance (top 5): {top_5}")

            return lgb_model

        except Exception as e:
            logger.error(f"Error training LightGBM: {e}")
            return None

    def train_meta_learner(
        self, meta_features: np.ndarray, y_train: np.ndarray
    ) -> LogisticRegression:
        """
        Train meta-learner on base model predictions

        Args:
            meta_features: Meta-features from base models
            y_train: Training labels

        Returns:
            Trained meta-learner
        """
        if len(meta_features) == 0 or len(meta_features) != len(y_train):
            logger.error("Invalid meta-features for meta-learner")
            return None

        logger.info(
            f"Training meta-learner with {len(meta_features)} samples, "
            f"{meta_features.shape[1]} meta-features"
        )

        try:
            meta_model = self.create_meta_learner()
            meta_model.fit(meta_features, y_train)

            logger.info("Meta-learner training completed successfully")

            return meta_model

        except Exception as e:
            logger.error(f"Error training meta-learner: {e}")
            return None

    def train_all_traditional_models(self, X_train: np.ndarray, y_train: np.ndarray) -> dict:
        """
        Train all traditional ML models

        Args:
            X_train: Training features
            y_train: Training labels

        Returns:
            Dictionary of trained models
        """
        models = {}

        if len(X_train) == 0:
            logger.warning("No training data provided for traditional ML models")
            return models

        logger.info(f"Training traditional ML models with {len(X_train)} samples")

        # Random Forest
        rf_model = self.train_random_forest(X_train, y_train)
        if rf_model is not None:
            models["rf"] = rf_model

        # LightGBM
        lgb_model = self.train_lightgbm(X_train, y_train)
        if lgb_model is not None:
            models["lgb"] = lgb_model

        logger.info(f"Successfully trained {len(models)} traditional ML models")

        return models

    def get_feature_importance(self, models: dict, feature_names: list = None) -> pd.DataFrame:
        """
        Get feature importance from trained models

        Args:
            models: Dictionary of trained models
            feature_names: List of feature names

        Returns:
            DataFrame with feature importance
        """
        importance_data = []

        for model_name, model in models.items():
            if hasattr(model, "feature_importances_"):
                importances = model.feature_importances_

                for i, importance in enumerate(importances):
                    feature_name = feature_names[i] if feature_names else f"feature_{i}"
                    importance_data.append(
                        {"model": model_name, "feature": feature_name, "importance": importance}
                    )

        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            # Pivot for easier analysis
            importance_pivot = importance_df.pivot(
                index="feature", columns="model", values="importance"
            ).fillna(0)

            # Calculate mean importance across models
            importance_pivot["mean_importance"] = importance_pivot.mean(axis=1)

            return importance_pivot.sort_values("mean_importance", ascending=False)

        return pd.DataFrame()


def test_traditional_models():
    """Test traditional ML models with sample data"""
    # Create sample data
    n_samples = 100
    n_features = 10
    n_classes = 3

    np.random.seed(42)
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, n_classes, n_samples)

    # Create model factory
    ml_models = TraditionalMLModels()

    logger.info("Testing traditional ML models")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Training labels shape: {y_train.shape}")

    # Test individual models
    rf_model = ml_models.train_random_forest(X_train, y_train)
    if rf_model:
        logger.info("Random Forest trained successfully")
        logger.info(f"RF accuracy on training data: {rf_model.score(X_train, y_train):.3f}")

    lgb_model = ml_models.train_lightgbm(X_train, y_train)
    if lgb_model:
        logger.info("LightGBM trained successfully")
        logger.info(f"LGB accuracy on training data: {lgb_model.score(X_train, y_train):.3f}")

    # Test batch training
    models = ml_models.train_all_traditional_models(X_train, y_train)
    logger.info(f"Batch training completed: {list(models.keys())}")

    # Test feature importance
    feature_names = [f"feature_{i}" for i in range(n_features)]
    importance_df = ml_models.get_feature_importance(models, feature_names)
    if not importance_df.empty:
        logger.info("Feature importance analysis:")
        logger.info(importance_df.head())

    # Test meta-learner with sample meta-features
    n_base_models = 2
    n_meta_features = n_base_models * n_classes
    meta_features = np.random.rand(n_samples, n_meta_features)

    meta_model = ml_models.train_meta_learner(meta_features, y_train)
    if meta_model:
        logger.info("Meta-learner trained successfully")
        logger.info(f"Meta-learner accuracy: {meta_model.score(meta_features, y_train):.3f}")

    return models


if __name__ == "__main__":
    test_traditional_models()
