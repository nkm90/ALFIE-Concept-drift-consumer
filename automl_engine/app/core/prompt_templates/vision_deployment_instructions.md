# Loading and/or Deploying a trained Vision model

- Once the AutoML tool is done, it will point you to a folder with the results/or you can find the results.zip from AutoDW
- The zip contains:
  - `model.pt` — the trained model (a full pickled PyTorch module, not just weights)
  - `feature_mapping.json` — the label map (and auxiliary-feature state for multimodal models)
  - `vision_deployment_instructions.md` — this file

## Vision model - Loading

`model.pt` was saved with `torch.save(model)`, i.e. it pickles the whole
`nn.Module` wrapper defined in this engine (`app.ml_engine.model`). Loading
therefore requires:

- the `app.ml_engine.model` module to be importable (add the repo root to
  `PYTHONPATH` or install the engine package in the deployment image)
- `torch >= 2.6`, which defaults to `weights_only=True`; a full-module pickle
  must be loaded with `weights_only=False`
- only load artifacts you trust — pickled modules can execute arbitrary code

```python
import json
import torch

from app.ml_engine.model import ImageClassificationModel  # noqa: F401  (makes the pickle resolvable)

model = torch.load("model/model.pt", weights_only=False, map_location="cpu")
model.eval()

with open("model/feature_mapping.json") as f:
    feature_mapping = json.load(f)

id2label = feature_mapping["label_map"]["id2label"]
```

## Vision model - Inference

Preprocessing must match training: rebuild the Hugging Face image processor
from the backbone inside the checkpoint.

```python
from PIL import Image
from transformers import AutoImageProcessor

processor = AutoImageProcessor.from_pretrained(model.model.config._name_or_path)

image = Image.open("example.jpg").convert("RGB")
inputs = processor(images=image, return_tensors="pt")

with torch.no_grad():
    logits = model(pixel_values=inputs["pixel_values"])

label = id2label[str(logits.argmax(dim=-1).item())]
```

The same pattern applies to the other vision task types (`image_segmentation`,
`object_detection`, `video_classification`, `keypoint_detection`): build the
processor with `AutoImageProcessor.from_pretrained`, feed the tensor keys it
returns to the model, and post-process the raw HF output for the task.

### Multimodal (image + tabular) models

For `image_classification_multimodal`, `feature_mapping.json` additionally
contains an `auxiliary_features` section with the fitted preprocessing state
for the tabular columns: the `StandardScaler` statistics (`mean`, `scale`) for
`numeric_columns` and the `OrdinalEncoder` `categories` for
`categorical_columns`. Rebuild that state and pass the scaled/encoded tabular
vector alongside the image:

```python
aux = feature_mapping["auxiliary_features"]

# numeric columns: (value - mean) / scale ; categorical columns: index into categories
# order must follow aux["auxiliary_columns"] exactly

with torch.no_grad():
    logits = model(pixel_values=inputs["pixel_values"], aux_features=aux_tensor)
```

## Deployment Considerations

- Treat the extracted model directory as an **immutable artifact**
- Load the model **once at service startup**, not per request, and reuse the
  in-memory object for all inference calls
- Serve behind a pinned environment: Python/torch/transformers versions should
  match (or be compatible with) the training environment; Docker is recommended
- Validate inputs at inference time (image readable, correct mode; for
  multimodal: all expected auxiliary columns present and in training order)
- Version each trained artifact (e.g. by dataset ID, timestamp, or hash) and
  reference models by versioned paths; switch traffic for rolling updates
- For throughput, batch images per request or run multiple workers; for
  latency, keep the model warm and avoid disk access during requests
- GPU is optional: the checkpoints run on CPU, at lower throughput
