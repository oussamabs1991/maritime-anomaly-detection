"""
Deep learning models for vessel classification
"""
import numpy as np
from loguru import logger

# Try to import TensorFlow, handle gracefully if not available
try:
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.layers import (
        LSTM,
        Attention,
        Bidirectional,
        Conv1D,
        Dense,
        Dropout,
        Flatten,
        GlobalAveragePooling1D,
        Input,
        MaxPooling1D,
    )
    from tensorflow.keras.models import Model
    from tensorflow.keras.optimizers import Adam

    TENSORFLOW_AVAILABLE = True
    logger.info("TensorFlow is available")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(f"TensorFlow not available: {e}")
    logger.warning("Deep learning models will be disabled")

    # Create dummy classes to prevent import errors
    class Model:
        pass

    class EarlyStopping:
        pass


from loguru import logger

from ..config import CNN_CONFIG, LSTM_CONFIG, config


class DeepLearningModels:
    """Factory class for creating deep learning models"""

    def __init__(self, config_obj=None):
        self.config = config_obj or config

    def create_cnn_model(
        self, input_shape: tuple, num_classes: int, test_mode: bool = None
    ) -> Model:
        """
        Create 1D CNN model for sequence classification

        Args:
            input_shape: Shape of input sequences (sequence_length, features)
            num_classes: Number of output classes
            test_mode: Whether to create simplified model for testing

        Returns:
            Compiled Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available - cannot create CNN model")
            return None

        test_mode = test_mode if test_mode is not None else self.config.TEST_MODE

        logger.info(
            f"Creating CNN model for input shape {input_shape}, "
            f"{num_classes} classes ({'TEST' if test_mode else 'FULL'} mode)"
        )

        inputs = Input(shape=input_shape)

        if test_mode:
            # Simplified architecture for testing
            x = Conv1D(32, 3, activation="relu", padding="same")(inputs)
            x = MaxPooling1D(2)(x)
            x = Flatten()(x)
            x = Dense(16, activation="relu")(x)
        else:
            # Full architecture
            params = CNN_CONFIG.params

            # First convolutional block
            x = Conv1D(
                params["filters"][0],
                params["kernel_sizes"][0],
                activation=params["activation"],
                padding="same",
            )(inputs)
            x = MaxPooling1D(params["pool_size"])(x)

            # Second convolutional block
            x = Conv1D(
                params["filters"][1],
                params["kernel_sizes"][1],
                activation=params["activation"],
                padding="same",
            )(x)
            x = MaxPooling1D(params["pool_size"])(x)

            # Dense layers
            x = Flatten()(x)
            x = Dense(params["dense_units"], activation=params["activation"])(x)
            x = Dropout(params["dropout_rate"])(x)

        # Output layer
        outputs = Dense(num_classes, activation="softmax")(x)

        # Create and compile model
        model = Model(inputs, outputs, name="CNN_Classifier")
        model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def create_bi_lstm_attention_model(
        self, input_shape: tuple, num_classes: int, test_mode: bool = None
    ) -> Model:
        """
        Create Bidirectional LSTM with Attention model

        Args:
            input_shape: Shape of input sequences
            num_classes: Number of output classes
            test_mode: Whether to create simplified model for testing

        Returns:
            Compiled Keras model
        """
        if not TENSORFLOW_AVAILABLE:
            logger.error("TensorFlow not available - cannot create Bi-LSTM model")
            return None

        test_mode = test_mode if test_mode is not None else self.config.TEST_MODE

        logger.info(
            f"Creating Bi-LSTM Attention model for input shape {input_shape}, "
            f"{num_classes} classes ({'TEST' if test_mode else 'FULL'} mode)"
        )

        inputs = Input(shape=input_shape)

        if test_mode:
            # Lightweight model for testing
            x = Bidirectional(LSTM(16, return_sequences=False))(inputs)
            x = Dense(16, activation="relu")(x)
        else:
            # Full model with attention
            params = LSTM_CONFIG.params

            # Bidirectional LSTM layer
            x = Bidirectional(LSTM(params["lstm_units"], return_sequences=True))(inputs)

            # Attention mechanism
            if params.get("attention", True):
                attention = Attention()([x, x])
                x = GlobalAveragePooling1D()(attention)
            else:
                x = GlobalAveragePooling1D()(x)

            # Dense layers
            x = Dense(params["dense_units"], activation="relu")(x)
            x = Dropout(params["dropout_rate"])(x)

        # Output layer
        outputs = Dense(num_classes, activation="softmax")(x)

        # Create and compile model
        model = Model(inputs, outputs, name="BiLSTM_Attention_Classifier")
        model.compile(
            optimizer=Adam(learning_rate=self.config.LSTM_LEARNING_RATE),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def get_early_stopping_callback(self) -> EarlyStopping:
        """
        Get early stopping callback with configured parameters

        Returns:
            EarlyStopping callback
        """
        return EarlyStopping(
            monitor="val_loss",
            patience=self.config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1,
        )

    def train_cnn_model(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        class_weights: dict = None,
    ) -> Model:
        """
        Train CNN model with robust class weight handling
        """
        epochs = self.config.TEST_EPOCHS if self.config.TEST_MODE else self.config.EPOCHS

        logger.info(f"Training CNN model for {epochs} epochs")

        # Fix class weights to ensure proper mapping
        fixed_class_weights = None
        if class_weights:
            try:
                unique_classes = sorted(np.unique(y_train))
                num_classes = len(unique_classes)

                # Map class weights to 0-based indices
                fixed_class_weights = {}
                for idx in range(num_classes):
                    # Find corresponding original label
                    orig_label = unique_classes[idx]
                    if orig_label in class_weights:
                        fixed_class_weights[idx] = class_weights[orig_label]
                    else:
                        fixed_class_weights[idx] = 1.0

                logger.info(f"Fixed class weights: {fixed_class_weights}")

            except Exception as e:
                logger.warning(f"Error fixing class weights: {e}. Using None.")
                fixed_class_weights = None

        # Prepare training arguments
        fit_args = {
            "x": X_train,
            "y": y_train,
            "epochs": epochs,
            "batch_size": min(
                self.config.BATCH_SIZE, len(X_train)
            ),  # Adjust batch size for small datasets
            "verbose": 1,
            "class_weight": fixed_class_weights,
        }

        # Add validation data and callbacks if available
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_args["validation_data"] = (X_val, y_val)
            # Only add early stopping if we have enough validation data
            if len(X_val) >= 5:
                fit_args["callbacks"] = [self.get_early_stopping_callback()]

        try:
            model.fit(**fit_args)
            logger.info("CNN training completed successfully")
            return model
        except Exception as e:
            logger.error(f"Error training CNN model: {e}")
            return None

    def train_lstm_model(
        self,
        model: Model,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        class_weights: dict = None,
    ) -> Model:
        """
        Train Bi-LSTM with Attention model with robust class weight handling
        """
        epochs = self.config.TEST_EPOCHS if self.config.TEST_MODE else self.config.EPOCHS

        logger.info(f"Training Bi-LSTM model for {epochs} epochs")

        # Fix class weights similar to CNN
        fixed_class_weights = None
        if class_weights:
            try:
                unique_classes = sorted(np.unique(y_train))
                num_classes = len(unique_classes)

                fixed_class_weights = {}
                for idx in range(num_classes):
                    orig_label = unique_classes[idx]
                    if orig_label in class_weights:
                        fixed_class_weights[idx] = class_weights[orig_label]
                    else:
                        fixed_class_weights[idx] = 1.0

                logger.info(f"Fixed LSTM class weights: {fixed_class_weights}")

            except Exception as e:
                logger.warning(f"Error fixing LSTM class weights: {e}. Using None.")
                fixed_class_weights = None

        # Prepare training arguments
        fit_args = {
            "x": X_train,
            "y": y_train,
            "epochs": epochs,
            "batch_size": min(self.config.LSTM_BATCH_SIZE, len(X_train)),
            "verbose": 1,
            "class_weight": fixed_class_weights,
        }

        # Add validation data and callbacks if available
        if X_val is not None and y_val is not None and len(X_val) > 0:
            fit_args["validation_data"] = (X_val, y_val)
            if len(X_val) >= 5:
                fit_args["callbacks"] = [self.get_early_stopping_callback()]

        try:
            model.fit(**fit_args)
            logger.info("Bi-LSTM training completed successfully")
            return model
        except Exception as e:
            logger.error(f"Error training Bi-LSTM model: {e}")
            return None

    def create_and_train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        num_classes: int = None,
        class_weights: dict = None,
    ) -> dict:
        """
        Create and train both deep learning models

        Args:
            X_train: Training sequences
            y_train: Training labels
            X_val: Validation sequences
            y_val: Validation labels
            num_classes: Number of classes
            class_weights: Class weights

        Returns:
            Dictionary of trained models
        """
        models = {}

        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow not available - skipping deep learning models")
            return models

        if len(X_train) == 0:
            logger.warning("No training data provided for deep learning models")
            return models

        input_shape = (X_train.shape[1], X_train.shape[2])
        num_classes = num_classes or len(np.unique(y_train))

        logger.info(
            f"Training deep learning models with input shape {input_shape}, "
            f"{num_classes} classes"
        )

        # Validate input shapes
        if X_val is not None and len(X_val) > 0:
            if X_val.shape[1:] != input_shape:
                logger.error(
                    f"Validation data shape {X_val.shape[1:]} doesn't match "
                    f"training data shape {input_shape}"
                )
                X_val, y_val = None, None

        # CNN Model
        try:
            cnn_model = self.create_cnn_model(input_shape, num_classes)
            trained_cnn = self.train_cnn_model(
                cnn_model, X_train, y_train, X_val, y_val, class_weights
            )
            if trained_cnn is not None:
                models["cnn"] = trained_cnn
        except Exception as e:
            logger.error(f"Failed to create/train CNN model: {e}")

        # Bi-LSTM Model
        try:
            lstm_model = self.create_bi_lstm_attention_model(input_shape, num_classes)
            trained_lstm = self.train_lstm_model(
                lstm_model, X_train, y_train, X_val, y_val, class_weights
            )
            if trained_lstm is not None:
                models["lstm"] = trained_lstm
        except Exception as e:
            logger.error(f"Failed to create/train Bi-LSTM model: {e}")

        logger.info(f"Successfully trained {len(models)} deep learning models")

        return models


def test_deep_learning_models():
    """Test deep learning models with sample data"""
    # Create sample data
    batch_size = 10
    sequence_length = 20
    n_features = 4
    n_classes = 3

    X_train = np.random.randn(batch_size, sequence_length, n_features)
    y_train = np.random.randint(0, n_classes, batch_size)
    X_val = np.random.randn(5, sequence_length, n_features)
    y_val = np.random.randint(0, n_classes, 5)

    # Create model factory
    dl_models = DeepLearningModels()

    logger.info("Testing deep learning models")
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Training labels shape: {y_train.shape}")

    # Test model creation
    input_shape = (sequence_length, n_features)

    cnn = dl_models.create_cnn_model(input_shape, n_classes, test_mode=True)
    logger.info(f"CNN model created: {cnn.name}")
    logger.info(f"CNN parameters: {cnn.count_params()}")

    lstm = dl_models.create_bi_lstm_attention_model(input_shape, n_classes, test_mode=True)
    logger.info(f"LSTM model created: {lstm.name}")
    logger.info(f"LSTM parameters: {lstm.count_params()}")

    # Test training
    models = dl_models.create_and_train_models(X_train, y_train, X_val, y_val, n_classes)

    logger.info(f"Trained models: {list(models.keys())}")

    return models


if __name__ == "__main__":
    test_deep_learning_models()
