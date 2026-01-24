# Standing Dead Tree Segmentation from Aerial Imagery

This project investigates automated segmentation of **standing dead trees (SDTs)** in aerial forest imagery to support forest health monitoring and environmental analysis. The work presents a comparative study of classical machine learning and modern deep learning–based segmentation approaches under a unified experimental pipeline.

The dataset consists of multispectral aerial images (RGB + Near-Infrared) with expert-annotated binary masks. Multiple models are implemented, trained, and evaluated to analyze trade-offs between segmentation accuracy, robustness, and computational efficiency.

---

## Dataset

- **Source**: Aerial Imagery for Standing Dead Tree Segmentation (Kaggle)
- **Samples**: 444 multispectral aerial images
- **Channels**: RGB + Near-Infrared (NIR)
- **Labels**: Binary masks (standing dead tree vs background)
- **Split**: 80% training / 20% validation

---

## Methods

Five segmentation approaches are evaluated:

### 1. SVM (Baseline)
- Traditional machine learning baseline using handcrafted pixel-level features (RGB + NIR)
- Linear SVM trained on balanced pixel samples
- Serves as a lower-bound reference for comparison

### 2. UNet
- Encoder–decoder convolutional neural network with skip connections
- Optimized using a combined Binary Cross-Entropy and Dice loss
- Strong recall performance with stable training behavior

### 3. DeepLabV3 (ResNet-50 Backbone)
- Deep semantic segmentation model with atrous convolutions
- Captures multi-scale spatial context effectively
- Achieves high precision and strong overall segmentation quality

### 4. Fast-SCNN
- Lightweight segmentation network designed for efficiency
- Trained with focal loss to mitigate class imbalance
- Offers fast convergence and low computational cost

### 5. ADA-Net
- Custom lightweight encoder–decoder architecture
- Designed to balance efficiency and segmentation quality
- Demonstrates competitive performance relative to larger models

---

## Experimental Setup

- **Input sizes**: 256–384 pixels (model-dependent)
- **Evaluation metrics**:
  - Pixel Accuracy
  - Precision / Recall
  - Dice Coefficient
  - Intersection over Union (IoU)
- **Training**:
  - Consistent data splits and evaluation pipeline across models
  - Best-performing checkpoints selected based on validation metrics

---

## Results Summary

Key observations from quantitative and qualitative evaluation:

- **DeepLabV3** achieved the strongest overall segmentation performance.
- **UNet** provided a robust balance between architectural simplicity and accuracy.
- **Fast-SCNN** demonstrated favorable efficiency–performance trade-offs.
- **ADA-Net** showed promising results as a compact segmentation model.
- **SVM** lagged behind deep learning approaches but served as a useful baseline.

No single model dominates across all metrics; model selection depends on accuracy requirements, computational constraints, and deployment scenarios.

---

## Repository Structure

├── dead_tree_segmentation.ipynb # Complete experimental pipeline
├── README.md
├── outputs/ # Evaluation results and visualizations
│ ├── confusion_matrices/
│ ├── training_curves/
│ └── qualitative_results/
└── models/ # (Optional) trained model checkpoints


*Note: The dataset is not included in this repository. Please refer to the Kaggle source for access.*

---

## Reproducibility

All trained models, evaluation results, and visualization artifacts are archived to support reproducibility and future extension.  
The notebook provides a modular and extensible pipeline that can be adapted to related remote sensing and environmental segmentation tasks.

---

## References

- Ronneberger et al., *U-Net: Convolutional Networks for Biomedical Image Segmentation*, 2015  
- Chen et al., *Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation*, 2018  
- Poudel et al., *Fast-SCNN: Fast Semantic Segmentation Network*, 2019  
- Aerial Imagery for Standing Dead Tree Segmentation, Kaggle