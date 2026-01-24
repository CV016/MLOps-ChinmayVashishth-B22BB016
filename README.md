# Deep Learning Classification on MNIST and FashionMNIST

Comprehensive study comparing ResNet architectures, optimizers, and compute devices for image classification.

---

## Project Overview

This project evaluates deep learning models (ResNet-18, ResNet-50) and traditional machine learning (SVM) on MNIST and FashionMNIST datasets. It includes hyperparameter optimization across 64 configurations and CPU vs GPU performance analysis.

**Datasets:** MNIST, FashionMNIST  
**Models:** ResNet-18, ResNet-50, Support Vector Machine  
**Framework:** PyTorch, scikit-learn

---

## Best Performing Models

### MNIST
- **Model:** ResNet-18
- **Optimizer:** SGD (lr=0.001, momentum=0.9)
- **Epochs:** 5
- **Test Accuracy:** 99.33%
- **Validation Accuracy:** 99.5%

### FashionMNIST
- **Model:** ResNet-18
- **Optimizer:** Adam (lr=0.001)
- **Epochs:** 5
- **Test Accuracy:** 91.54%
- **Validation Accuracy:** 91.5%

---

## Question 1(a): Deep Learning Classification Results

### Experimental Configuration
- **Batch Size:** 16
- **Learning Rates:** 0.001, 0.0001
- **Epochs:** 3, 5
- **Pin Memory:** False, True
- **AMP:** True (all experiments)
- **Train-Val-Test Split:** 70%-10%-20%
- **Image Size:** 64×64

### MNIST Results Summary

| Epochs | Pin Memory | Optimizer | LR | ResNet-18 | ResNet-50 |
|--------|-----------|-----------|------|-----------|-----------|
| 3 | False | SGD | 0.001 | 98.89% | 98.70% |
| 3 | False | Adam | 0.0001 | 99.13% | 98.30% |
| 5 | False | SGD | 0.001 | **99.33%** | 99.16% |
| 5 | False | Adam | 0.001 | 99.07% | 99.03% |
| 5 | True | Adam | 0.0001 | 99.12% | 98.87% |

### FashionMNIST Results Summary

| Epochs | Pin Memory | Optimizer | LR | ResNet-18 | ResNet-50 |
|--------|-----------|-----------|------|-----------|-----------|
| 3 | False | SGD | 0.001 | 90.08% | 87.78% |
| 3 | False | Adam | 0.0001 | 89.75% | 86.48% |
| 5 | False | SGD | 0.001 | 90.59% | 89.68% |
| 5 | False | Adam | 0.0001 | 91.22% | 89.48% |
| 5 | True | Adam | 0.001 | **91.54%** | 90.80% |

### Key Findings
- ResNet-18 matches or exceeds ResNet-50 performance on both datasets
- SGD outperforms Adam on MNIST (99.33% vs 99.13%)
- Adam outperforms SGD on FashionMNIST (91.54% vs 90.59%)
- 5 epochs consistently improve accuracy by 0.3-0.5% over 3 epochs
- Learning rate 0.001 shows better results than 0.0001 for optimal models

---

## Question 1(b): Support Vector Machine Classification

### Configuration
- **Training Samples:** 10,000
- **Kernels:** Polynomial, RBF
- **Parameters:** gamma='scale', max_iter=1000

### Results

| Dataset | Kernel | Test Accuracy | Training Time (ms) |
|---------|--------|--------------|-------------------|
| MNIST | poly | 95.40% | 7710.46 |
| MNIST | **rbf** | **96.35%** | 7635.02 |
| FashionMNIST | poly | 80.67% | 7658.90 |
| FashionMNIST | **rbf** | **85.67%** | 7558.53 |

### Analysis
- RBF kernel outperforms polynomial kernel on both datasets
- ResNet models show significant advantage over SVM:
  - MNIST: +2.98% (99.33% vs 96.35%)
  - FashionMNIST: +5.87% (91.54% vs 85.67%)

---

## Question 2: CPU vs GPU Performance Comparison

### Configuration
- **Dataset:** FashionMNIST
- **Batch Size:** 16
- **Learning Rate:** 0.001
- **Epochs:** 1
- **Image Size:** 224×224

### Computational Complexity
- **ResNet-18:** 1.824 GFLOPs
- **ResNet-50:** 4.132 GFLOPs (2.27× more than ResNet-18)

### Performance Results

| Model | Optimizer | Device | Test Acc | Training Time (ms) | Speedup |
|-------|-----------|--------|----------|-------------------|---------|
| ResNet-18 | SGD | CPU | 86.56% | 3,269,881 | 1.00× |
| ResNet-18 | SGD | GPU | 83.38% | 121,377 | **26.94×** |
| ResNet-18 | Adam | CPU | 85.46% | 3,090,957 | 1.00× |
| ResNet-18 | Adam | GPU | 85.25% | 127,500 | **24.24×** |
| ResNet-50 | SGD | CPU | 76.81% | 10,360,177 | 1.00× |
| ResNet-50 | SGD | GPU | 76.50% | 275,000 | **37.67×** |
| ResNet-50 | Adam | CPU | 81.40% | 10,534,333 | 1.00× |
| ResNet-50 | Adam | GPU | 81.20% | 280,000 | **37.62×** |

### Key Insights
- GPU provides 24-27× speedup for ResNet-18
- GPU provides 37-38× speedup for ResNet-50
- Deeper models benefit more from GPU parallelization
- CPU and GPU produce nearly identical accuracy (within 0-3%)
- Training time reduced from hours to minutes with GPU

---

## Comparative Analysis

### Model Architecture
- ResNet-18 consistently matches or exceeds ResNet-50 performance
- Simpler architecture shows faster convergence and better generalization
- Lower computational cost without accuracy sacrifice

### Optimizer Selection
- Dataset-dependent optimal choice:
  - MNIST: SGD with momentum (99.33%)
  - FashionMNIST: Adam (91.54%)
- Adam adds 3-5% training time overhead for adaptive learning

### Training Dynamics
- MNIST shows rapid convergence (most learning in first 2 epochs)
- FashionMNIST benefits from full 5 epochs
- Minimal overfitting on MNIST, slight overfitting on FashionMNIST
- Training curves confirm stable learning without oscillations

### Deep Learning vs Traditional ML
- ResNet significantly outperforms SVM on both datasets
- Accuracy improvements justify higher computational costs
- Deep learning essential for complex image classification

### Hardware Considerations
- GPU acceleration essential for practical deep learning
- 30× average speedup transforms development workflow
- GPU makes deeper architectures practically feasible
- Critical for production-scale training and experimentation

---

## Recommendations

### For Image Classification Tasks
1. **Model:** ResNet-18 provides optimal accuracy-efficiency balance
2. **Optimizer:** Dataset-specific tuning (SGD for simple, Adam for complex)
3. **Learning Rate:** 0.001 for faster convergence
4. **Epochs:** 5+ for optimal performance
5. **Hardware:** GPU for training, 25-40× speedup over CPU

### Dataset-Specific
- **MNIST-like datasets:** ResNet-18 + SGD
- **Complex datasets:** ResNet-18 + Adam
- **Limited data:** Consider SVM for faster training
- **Production:** Always use GPU acceleration

---

## Repository Structure

```
├── training.ipynb                  # Main training notebook (Q1a, Q1b)
├── q2_cpu_gpu_performance.ipynb    # CPU vs GPU comparison (Q2)
├── best_models_training.ipynb      # Train best models with graphs
├── log1.txt                        # Complete training logs
├── report.txt                      # LaTeX report source
├── results/                        # CSV results files
├── models/                         # Saved model checkpoints
├── best_models/                    # Best model checkpoints
├── best_results/                   # Training curves for best models
└── q2_results/                     # Q2 performance visualizations
```

---

## Technical Details

### Environment
- **Framework:** PyTorch 2.x
- **Libraries:** torchvision, scikit-learn, numpy, pandas, matplotlib
- **GPU:** NVIDIA CUDA-enabled
- **CPU:** Intel Xeon processor

### Training Configuration
- **Loss Function:** CrossEntropyLoss
- **SGD Parameters:** momentum=0.9
- **Data Augmentation:** Resize to 64×64, Normalize
- **Mixed Precision:** Enabled (AMP) for all experiments

### Evaluation Metrics
- Test Accuracy
- Validation Accuracy
- Training Time (milliseconds)
- FLOPs (GigaFLOPs)
- Speedup Factor (CPU vs GPU)

---

## Results Reproducibility

All experiments are fully reproducible with provided notebooks and configurations. Deterministic results achieved through:
- Fixed random seeds
- Consistent data splits (70%-10%-20%)
- Documented hyperparameters
- Complete training logs

---

## Conclusion

This study demonstrates that ResNet-18 with dataset-specific optimizer tuning, trained on GPU for 5 epochs with learning rate 0.001, provides an excellent balance of accuracy, training speed, and computational efficiency for MNIST and FashionMNIST classification tasks. GPU acceleration is essential for practical deep learning, providing 25-40× speedup and making deeper architectures feasible for rapid experimentation and production deployment.

