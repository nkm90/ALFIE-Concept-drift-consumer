# Loading and/or Deploying a trained model

- Once the AutoML tool is done, it will point you to a folder with the results/or you can find the results.zip from AutoDW

## Tabular model - Loading

- Once you have this folder (extract the zip if it is one), you can simply do

```
from autogluon.tabular import TabularPredictor
predictor = TabularPredictor.load(predictor_path)
```

- Now to predict something you can do, `predictor.predict(test_data)` or `predictor.predict(test_data, model = 'X')` for a specific model
- For more instructions on how to use this, please refer to the [AutoGluon documentation page](https://auto.gluon.ai/stable/tutorials/tabular/tabular-essentials.html#loading-a-trained-predictor)

## Tabular model - Deployment

## Deployment Considerations

When deploying an AutoGluon Tabular model to production, the saved predictor directory should be treated as an **immutable artifact**. The predictor already includes all preprocessing, feature transformations, and ensemble logic, so no separate pipeline code is required.

### Recommended Deployment Pattern

- Package the extracted predictor directory together with:
  - Your inference service code (e.g. FastAPI, Flask)
  - A pinned Python environment (e.g. `requirements.txt` or Docker image)

- Load the predictor **once at service startup**, not per request
- Reuse the in-memory predictor object for all inference calls

Example (FastAPI-style initialization):

```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor.load("/models/automl_predictor")
```

### Input Validation

At inference time:

- Inputs must be provided as a pandas DataFrame
- Feature columns must exactly match those used during training
- The target column must **not** be included

It is strongly recommended to:

- Validate column names and data types before prediction
- Reject or log requests with missing or extra columns

### Performance and Scaling

- AutoGluon predictors are CPU-optimized by default
- For high-throughput use cases:
  - Run multiple workers (e.g. `uvicorn --workers N`)
  - Consider batching predictions where possible

- For low-latency scenarios, avoid model reloading and disk access during requests

### Model Versioning and Updates

- Each trained predictor directory should be versioned (e.g. by dataset ID, timestamp, or hash)
- Deployments should reference models by versioned paths
- Rolling updates can be achieved by loading a new predictor and switching traffic

### Serialization and Portability

- The predictor directory is **not** framework-agnostic; it must be used with AutoGluon
- Python and AutoGluon versions should match (or be compatible) between training and deployment environments
- Containerized deployment (Docker) is recommended for reproducibility
