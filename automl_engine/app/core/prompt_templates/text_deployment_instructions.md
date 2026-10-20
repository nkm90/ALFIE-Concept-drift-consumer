# Loading and/or Deploying a trained Text model

- Once the AutoML tool is done, it will point you to a folder with the results/or you can find the results.zip from AutoDW
- The zip contains:
  - `model.pt` — the trained model (a full pickled PyTorch module, not just weights)
  - `feature_mapping.json` — the label map, plus the tokenizer vocabulary and
    the `hf_model_id` it was built from
  - `text_deployment_instructions.md` — this file

## Text model - Loading

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

from app.ml_engine.model import SequenceClassificationModel  # noqa: F401  (makes the pickle resolvable; import the class matching your task)

model = torch.load("model/model.pt", weights_only=False, map_location="cpu")
model.eval()

with open("model/feature_mapping.json") as f:
    feature_mapping = json.load(f)

id2label = feature_mapping["label_map"]["id2label"]
tokenizer_id = feature_mapping["tokenizer"]["hf_model_id"]
```

## Text model - Inference

Rebuild the tokenizer from the `hf_model_id` recorded in
`feature_mapping.json` so preprocessing matches training exactly.

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
```

### text_classification

```python
inputs = tokenizer("A sentence to classify", return_tensors="pt", truncation=True)

with torch.no_grad():
    logits = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

label = id2label[str(logits.argmax(dim=-1).item())]
```

### question_answering

The wrapper returns the raw HF QA output; extract the answer span with the
start/end logits:

```python
inputs = tokenizer(question, context, return_tensors="pt", truncation=True)

with torch.no_grad():
    output = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])

start = output.start_logits.argmax(dim=-1).item()
end = output.end_logits.argmax(dim=-1).item()
answer = tokenizer.decode(inputs["input_ids"][0, start : end + 1])
```

### causal_lm / masked_lm / seq2seq_lm

The training wrappers return the training **loss** only. For generation, call
the underlying Hugging Face model directly:

```python
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    generated = model.model.generate(**inputs, max_new_tokens=64)

text = tokenizer.decode(generated[0], skip_special_tokens=True)
```

## Deployment Considerations

- Treat the extracted model directory as an **immutable artifact**
- Load the model **once at service startup**, not per request, and reuse the
  in-memory object for all inference calls
- Serve behind a pinned environment: Python/torch/transformers versions should
  match (or be compatible with) the training environment; Docker is recommended
- Validate inputs at inference time: non-empty text, sensible length limits
  (`truncation=True` with the tokenizer's max length) before feeding the model
- Version each trained artifact (e.g. by dataset ID, timestamp, or hash) and
  reference models by versioned paths; switch traffic for rolling updates
- For throughput, batch sequences per request or run multiple workers; for
  latency, keep the model warm and avoid disk access during requests
- GPU is optional: the checkpoints run on CPU, at lower throughput
