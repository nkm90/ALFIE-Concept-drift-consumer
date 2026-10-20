# Loading and/or Deploying a trained Audio model

- Once the AutoML tool is done, it will point you to a folder with the results/or you can find the results.zip from AutoDW
- The zip contains:
  - `model.pt` — the trained model (a full pickled PyTorch module, not just weights)
  - `feature_mapping.json` — the label map for `audio_classification`
  - `audio_deployment_instructions.md` — this file

## Audio model - Loading

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

from app.ml_engine.model import AudioClassificationModel  # noqa: F401  (makes the pickle resolvable)

model = torch.load("model/model.pt", weights_only=False, map_location="cpu")
model.eval()

with open("model/feature_mapping.json") as f:
    feature_mapping = json.load(f)

id2label = feature_mapping["label_map"]["id2label"]
```

## Audio model - Inference

Preprocessing must match training: audio was resampled to 16 kHz mono and
passed through the Hugging Face feature extractor of the underlying backbone
(e.g. `facebook/wav2vec2-base`).

```python
import torchaudio
from transformers import AutoFeatureExtractor

feature_extractor = AutoFeatureExtractor.from_pretrained(
    model.model.config._name_or_path
)

waveform, sample_rate = torchaudio.load("example.wav")
waveform = waveform.mean(dim=0)                      # to mono
if sample_rate != 16000:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)

inputs = feature_extractor(waveform.numpy(), sampling_rate=16000, return_tensors="pt")

with torch.no_grad():
    logits = model(input_values=inputs["input_values"])

label = id2label[str(logits.argmax(dim=-1).item())]
```

## Deployment Considerations

- Treat the extracted model directory as an **immutable artifact**
- Load the model **once at service startup**, not per request, and reuse the
  in-memory object for all inference calls
- Serve behind a pinned environment: Python/torch/transformers/torchaudio
  versions should match (or be compatible with) the training environment;
  Docker is recommended
- Validate inputs at inference time: audio must be decodable and resampled to
  16 kHz mono before feature extraction; reject or log files that fail
- Version each trained artifact (e.g. by dataset ID, timestamp, or hash) and
  reference models by versioned paths; switch traffic for rolling updates
- For throughput, batch clips per request or run multiple workers; for
  latency, keep the model warm and avoid disk access during requests
- GPU is optional: the checkpoints run on CPU, at lower throughput
