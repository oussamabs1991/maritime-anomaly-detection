"""
Maritime Anomaly Detection - Main Entry Point

This script provides the main interface for running the maritime vessel type
classification pipeline using AIS data.
"""
import argparse
import sys
from pathlib import Path

from loguru import logger

# Try to import from installed package first, fallback to development mode
try:
    from src.config import config, load_config_from_yaml
    from src.pipeline import MaritimePipeline
except ImportError:
    # Add src to path for development mode
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from src.config import config, load_config_from_yaml
    from src.pipeline import MaritimePipeline


def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Setup logging configuration"""
    # Remove default logger
    logger.remove()

    # Add console logger
    logger.add(
        sys.stderr,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
            "<level>{message}</level>"
        ),
        level=log_level,
        colorize=True,
    )

    # Add file logger if specified
    if log_file:
        logger.add(
            log_file,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | " "{name}:{function}:{line} - {message}"
            ),
            level=log_level,
            rotation="10 MB",
            retention="1 week",
        )


def main():
    """Main function to run the maritime anomaly detection pipeline"""

    parser = argparse.ArgumentParser(
        description="Maritime Vessel Type Classification using AIS Data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with ZIP file
  python main.py --zip-file data/AIS_2024_10_24.zip --csv-name AIS_2024_10_24.csv

  # Run in test mode
  python main.py --zip-file data/AIS_2024_10_24.zip --csv-name AIS_2024_10_24.csv --test-mode

  # Run with custom config
  python main.py --config config.yaml --zip-file data/AIS_2024_10_24.zip

  # Run smoke tests only
  python main.py --smoke-tests-only

  # Load and evaluate existing model
  python main.py --load-model models/ensemble_model.joblib --zip-file data/test.zip \\
    --csv-name test.csv
        """,
    )

    # Data arguments
    parser.add_argument("--zip-file", type=str, help="Path to ZIP file containing AIS data")

    parser.add_argument("--csv-name", type=str, help="Name of CSV file inside the ZIP")

    # Mode arguments
    parser.add_argument(
        "--test-mode", action="store_true", help="Run in test mode with sample data"
    )

    parser.add_argument(
        "--smoke-tests-only", action="store_true", help="Run smoke tests only and exit"
    )

    # Model arguments
    parser.add_argument("--load-model", type=str, help="Path to pre-trained model to load")

    parser.add_argument("--save-model", type=str, help="Path to save the trained model")

    # Configuration arguments
    parser.add_argument("--config", type=str, help="Path to YAML configuration file")

    # Logging arguments
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    parser.add_argument("--log-file", type=str, help="Path to log file")

    # Output arguments
    parser.add_argument(
        "--no-plots", action="store_true", help="Disable plot generation and saving"
    )

    parser.add_argument("--output-dir", type=str, help="Output directory for results")

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level, args.log_file)

    logger.info("Starting Maritime Anomaly Detection Pipeline")
    logger.info(f"Arguments: {vars(args)}")

    try:
        # Load configuration
        if args.config:
            if not Path(args.config).exists():
                logger.error(f"Config file not found: {args.config}")
                return 1
            config_obj = load_config_from_yaml(args.config)
            logger.info(f"Loaded configuration from: {args.config}")
        else:
            config_obj = config
            logger.info("Using default configuration")

        # Override config with command line arguments
        if args.test_mode:
            config_obj = config_obj.get_test_config()
            logger.info("Running in TEST mode")

        if args.no_plots:
            config_obj.SAVE_PLOTS = False

        if args.output_dir:
            config_obj.DATA_DIR = Path(args.output_dir)
            config_obj.PLOT_DIR = Path(args.output_dir) / "plots"
            config_obj.MODELS_DIR = Path(args.output_dir) / "models"

        # Create pipeline
        pipeline = MaritimePipeline(config_obj)

        # Run smoke tests if requested
        if args.smoke_tests_only:
            logger.info("Running smoke tests only")
            success = pipeline.run_smoke_tests()
            if success:
                logger.info("All smoke tests passed!")
                return 0
            else:
                logger.error("Smoke tests failed!")
                return 1

        # Validate required arguments for full pipeline
        if not args.load_model:
            if not args.zip_file:
                logger.error("--zip-file is required when not loading a pre-trained model")
                return 1

            if not args.csv_name:
                logger.error("--csv-name is required when not loading a pre-trained model")
                return 1

            if not Path(args.zip_file).exists():
                logger.error(f"ZIP file not found: {args.zip_file}")
                return 1

        # Load existing model or train new one
        if args.load_model:
            logger.info(f"Loading pre-trained model from: {args.load_model}")
            pipeline.load_model(args.load_model)

            # If data provided, evaluate the loaded model
            if args.zip_file and args.csv_name:
                logger.info("Evaluating loaded model on new data")

                # Load and prepare data
                df = pipeline.load_and_prepare_data(args.zip_file, args.csv_name)
                df = pipeline.preprocess_data(df)
                df = pipeline.apply_trajectory_smoothing(df)
                features_df, X, y = pipeline.extract_features(df)
                sequences = pipeline.prepare_sequence_data(df, features_df)

                # Scale features using existing scaler
                X_scaled = pipeline.scaler.transform(X)

                # Encode labels
                if hasattr(pipeline.label_encoder, "classes_"):
                    y_encoded = pipeline.label_encoder.transform(y)
                else:
                    y_encoded = y

                # Evaluate
                class_names = (
                    list(pipeline.label_encoder.classes_)
                    if hasattr(pipeline.label_encoder, "classes_")
                    else None
                )
                results = pipeline.evaluate_model(X_scaled, y_encoded, sequences, class_names)

                # Create visualizations
                predictions = pipeline.ensemble.predict(X_scaled, sequences)
                probabilities = pipeline.ensemble.predict_proba(X_scaled, sequences)
                pipeline.create_visualizations(y, predictions, probabilities, class_names)

                logger.info("Model evaluation completed")

        else:
            # Run complete pipeline
            logger.info("Starting complete training pipeline")

            # Run smoke tests first
            logger.info("Running smoke tests before pipeline")
            smoke_success = pipeline.run_smoke_tests()
            if not smoke_success:
                logger.warning("Some smoke tests failed, but continuing with pipeline")

            # Run pipeline
            results = pipeline.run_complete_pipeline(args.zip_file, args.csv_name)

            # Save model with custom path if specified
            if args.save_model:
                pipeline.save_model(args.save_model)
                logger.info(f"Model saved to custom path: {args.save_model}")

            # Log final results
            eval_results = results["evaluation_results"]
            basic_metrics = eval_results["basic_metrics"]

            logger.info("Pipeline Results Summary:")
            logger.info(f"  Accuracy: {basic_metrics['accuracy']:.4f}")
            logger.info(f"  F1 Score (weighted): {basic_metrics['f1_score']:.4f}")
            logger.info(f"  F1 Score (macro): {basic_metrics['f1_macro']:.4f}")
            logger.info(f"  Execution Time: {results['execution_time']/60:.2f} minutes")
            logger.info(f"  Data Shape: {results['data_shape']}")
            logger.info(f"  Features Shape: {results['features_shape']}")
            logger.info(f"  Model Path: {results['model_path']}")

        logger.info("Pipeline completed successfully!")
        return 0

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
