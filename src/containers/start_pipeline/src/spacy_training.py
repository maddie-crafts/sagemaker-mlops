import sys
import subprocess
import logging
import shutil
import json
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Install and Import Required Packages ---
def install_and_import(package, pip_args=None, import_name=None, spacy_download_model=False):
    if import_name is None:
        # Only install, no import
        logger.info(f"Installing package without importing: {package}")
        if spacy_download_model:
            subprocess.run([sys.executable, "-m", "spacy", "download", package], check=True)
        else:
            subprocess.run(pip_args or [sys.executable, "-m", "pip", "install", package], check=True)
        return

    import_name = import_name or package.split("==")[0].replace("-", "_")
    try:
        __import__(import_name)
    except ImportError:
        if spacy_download_model:
            logger.info(f"Downloading spaCy model: {package}")
            subprocess.run([sys.executable, "-m", "spacy", "download", package], check=True)
        else:
            logger.info(f"Installing missing package: {package}")
            subprocess.run(pip_args or [sys.executable, "-m", "pip", "install", package], check=True)
        __import__(import_name)

REQUIRED_PACKAGES = [
    ("spacy==3.7.0", None),
    ("torch", [sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"]),
    ("spacy-transformers", None),
    ("sentencepiece", None),
    ("cupy-cuda11x", [sys.executable, "-m", "pip", "install", "cupy-cuda11x", "--extra-index-url", "https://pypi.cupy.dev/simple"], None),
    ("en_core_web_trf", None, None, True),
]

for pkg in REQUIRED_PACKAGES:
    install_and_import(*pkg)

# === Utility Imports (assumed to be local ===
from utils import (
    load_model_and_predict,
    remove_duplicated_spans,
    evaluate_model
)

def train_model(train_path: str, validation_path: str, config_path: str = None, output_dir: str = "./output"):
    """Run spaCy training."""
    logger.info("Starting spaCy training...")
    if config_path is None:
        config_path = Path(__file__).parent / "config.cfg"
    else:
        config_path = Path(config_path)

    subprocess.run([
        "python", "-m", "spacy", "train", config_path,
        "--paths.train", train_path,
        "--paths.dev", validation_path,
        "--gpu-id", "0",
        "--output", output_dir
    ], check=True)
    logger.info("Training complete.")

def evaluate_model_on_test_data(model_dir: Path, test_path: Path, evaluation_output_path: Path):
    """Run evaluation using the trained model and save the results."""
    logger.info(f"Evaluating model using test data from {test_path}")
    test_data = pd.read_csv(test_path)

    if 'span' not in test_data.columns:
        test_data['span'] = None

    model_data = load_model_and_predict(str(model_dir), test_data)
    model_data = remove_duplicated_spans(model_data)

    report = evaluate_model(model_data, test_data)

    # Save evaluation report
    evaluation_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(evaluation_output_path, "w") as f:
        json.dump(report, f)
    
    logger.info(f"Evaluation complete. Report saved to {evaluation_output_path}")
    return report

def move_model_files(source_dir: Path, destination_dir: Path):
    """Move trained model files to the SageMaker output directory."""
    if not source_dir.exists():
        logger.error(f"Source directory '{source_dir}' does not exist.")
        return

    destination_dir.mkdir(parents=True, exist_ok=True)
    for file_path in source_dir.iterdir():
        shutil.move(str(file_path), str(destination_dir / file_path.name))
        logger.info(f"Moved '{file_path.name}' to '{destination_dir}'")

def main():
        
    train_path = Path("/opt/ml/input/data/train/train.spacy")
    validation_path = Path("/opt/ml/input/data/validation/validation.spacy")
    test_path = Path("/opt/ml/input/data/test/test.csv")
    model_output_dir = Path("./output/model-last")
    final_model_dir = Path("/opt/ml/model/output/model-last/")
    evaluation_path = Path("/opt/ml/processing/evaluation/evaluation.json")

    train_model(str(train_path), str(validation_path))
    report = evaluate_model_on_test_data(model_output_dir, test_path, evaluation_path)
    move_model_files(model_output_dir, final_model_dir)

if __name__ == "__main__":
    main()