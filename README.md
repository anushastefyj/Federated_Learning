# Calibration-Aware Federated Learning for Ultra-Low-Parameter Air-Quality Models

This repository contains the codebase for reproducing the research project on federated learning for heterogeneous low-cost sensor networks, specifically tailored for TinyML deployment.

## Project Structure
- `data/`: Data loading, preprocessing, and Non-IID client split simulation. By default, generates synthetic sensor data.
- `models/`: PyTorch definitions for ultra-low parameter `TinyCNN` and `TinyMLP` models.
- `fl/`: Federated Learning implementation using the Flower (`flwr`) framework, including a custom `CalibrationAwareStrategy` and `FedProx`.
- `tinyml/`: Scripts for exporting the trained models to ONNX and converting Keras/TensorFlow models to TFLite (INT8 Quantization).
- `utils/`: Configuration and logging utilities.

## Requirements
To install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Code
The `main.py` script serves as the central entrypoint. You can run various baselines and federated strategies.

### Centralized Baseline
```bash
python main.py --mode centralized
```

### Local-Only Baseline
```bash
python main.py --mode local
```

### Federated Learning (FedAvg)
```bash
python main.py --mode fedavg --rounds 10 --clients 5
```

### Federated Learning (FedProx)
```bash
python main.py --mode fedprox --rounds 10 --clients 5
```

### Federated Learning (Calibration-Aware)
```bash
python main.py --mode calibration_aware --rounds 10 --clients 5
```

## Output
Results are logged in `outputs/results.csv`, and an ONNX export of the final model structure is saved to the `outputs/` directory.

## Customization
- **Data**: Replace the synthetic data logic in `data/preprocess.py` with your CPCB India / OpenAQ CSV paths.
- **TinyML**: See `tinyml/export.py` for how to create an equivalent TensorFlow model and apply INT8 quantization for TFLite Micro deployments.
