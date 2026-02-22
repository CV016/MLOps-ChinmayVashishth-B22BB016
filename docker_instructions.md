# Docker Image Build Instructions

This document explains how to build and run Docker images using the provided `Dockerfile` and `dockerfile.eval`.

---

# 1. Prerequisites

Ensure Docker is installed and working:

```bash
docker --version
```

If Docker is not installed, install it first.

---


# 2. Full Workflow Summary

```bash
# Build images
docker build -t pytorch-training .
docker build -t pytorch-eval -f dockerfile.eval .

# Run training
docker run -it pytorch-training

# Run evaluation
docker run -v $(pwd)/results:/results pytorch-eval
```

---

# Done

Your Docker environment is now ready for training and evaluation.
