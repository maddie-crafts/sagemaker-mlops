# SageMaker MLOps Pipeline: Span Categorization Model Training

This project sets up and runs a SageMaker ML pipeline for processing, training, evaluating, and registering a text classification model (SpanCat) focused on a selected category (e.g., *Organization*).

## Python script: `src/containers/start_pipeline/build_pipeline.py`

This script defines and runs a full SageMaker pipeline using custom modules for each step:

- **Preprocessing**
- **Training**
- **Evaluation**
- **Model Registration**
- **Evaluation Condition Check**

---