# ALFIE Concept Drift Consumer

Standalone Kafka concept-drift worker for ALFIE.

Supports:

- Tabular classification/regression
- Image classification
- Semantic image segmentation
- River ADWIN drift detection
- AutoGluon tabular adaptive retraining
- AutoML Vision retraining
- Reserved holdout validation
- Source-vs-candidate model acceptance
- Data Warehouse drift reports
- Kafka completion events

## Dependencies

The consumer expects the following services to be running:

- Kafka
- ALFIE Data Warehouse API
- ALFIE AutoML Engine

The container must be attached to both the Data Warehouse/Kafka Docker
network and the AutoML Docker network.

## Configuration

Create local configuration files:

```powershell
Copy-Item ".env.cdd.example" ".env.cdd.local"
Copy-Item ".env.compose.example" ".env"