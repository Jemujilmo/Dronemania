# Dronemania Docker Build and Run Instructions

## Build Docker Image

```powershell
docker build -t dronemania:latest -f docker/Dockerfile .
```

## Run Containerized Autonomy Stack

```powershell
# Run with default config
docker run --rm dronemania:latest

# Run with custom config (mount config directory)
docker run --rm -v ${PWD}/config:/app/config dronemania:latest python src/autonomy/main.py --config /app/config/autonomy.yaml

# Run with GPU support (if needed for neural networks)
docker run --rm --gpus all dronemania:latest
```

## For Development (Interactive)

```powershell
# Run container interactively
docker run -it --rm -v ${PWD}:/app dronemania:latest /bin/bash

# Inside container, run:
python src/autonomy/main.py --config config/autonomy.yaml
```

## Verify Reproducibility

```powershell
# Run same config multiple times - should produce identical behavior
docker run --rm dronemania:latest > run1.log
docker run --rm dronemania:latest > run2.log
diff run1.log run2.log
```

## Notes

- Container runs offline (no external network calls)
- All dependencies pinned in requirements.txt
- Deterministic behavior enforced via environment variables
- Seeds set for reproducibility
