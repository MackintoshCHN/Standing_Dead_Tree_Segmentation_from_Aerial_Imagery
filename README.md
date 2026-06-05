# Standing Dead Tree Segmentation from Aerial Imagery

This project studies **standing dead tree segmentation** from aerial forest imagery using classical machine learning, deep learning, and lightweight segmentation models.

I built an end-to-end computer vision workflow that covers dataset preparation, model training, evaluation, visualization, cross-model comparison, and final project packaging. The goal is to understand how different segmentation model families behave on the same remote sensing task, especially when the target objects are sparse, small, and visually similar to surrounding vegetation.

## Project Overview

Standing dead tree detection is useful for forest monitoring, ecological assessment, and environmental management. In aerial imagery, this task is challenging because dead tree regions can be small, irregular, and difficult to distinguish from complex forest backgrounds.

In this project, I explored multiple segmentation approaches, including:

* **SVM** — a classical pixel-level machine learning baseline.
* **UNet** — a convolutional encoder-decoder segmentation model.
* **DeepLabV3** — a ResNet-50-based semantic segmentation model.
* **Fast-SCNN-style model** — a lightweight segmentation baseline.
* **ADA-Net-style model** — a custom compact encoder-decoder model.

The notebook compares these models using quantitative metrics, training curves, confusion matrices, and qualitative prediction visualizations.

## Project Goals

The main goals of this project are:

* **Build a complete segmentation workflow**

  * Prepare image-mask pairs.
  * Create dataset loaders.
  * Train multiple segmentation models.
  * Save checkpoints, logs, predictions, and visual outputs.

* **Compare different model families**

  * Classical machine learning baseline.
  * Standard deep learning segmentation architecture.
  * High-capacity semantic segmentation model.
  * Lightweight custom models.

* **Analyze model behaviour**

  * Track Dice score, IoU, pixel accuracy, precision, and recall.
  * Visualize training curves and prediction results.
  * Inspect common failure cases.

* **Prepare a reusable GitHub project**

  * Keep the notebook readable.
  * Organize outputs clearly.
  * Summarize practical trade-offs between accuracy and efficiency.
  * Preserve the workflow for future review and extension.

## Dataset

The dataset contains aerial forest imagery and pixel-level segmentation masks.

The notebook uses:

* **RGB images**
* **NRG images**
* **Binary segmentation masks**

The dataset archive is expected to be available in Google Drive when running the notebook in Colab. The notebook extracts the archive into the Colab workspace and prepares the required file paths for training and evaluation.

> Note: The dataset itself may not be included in this repository if it is too large or restricted. To reproduce the workflow, place the dataset archive in the expected Google Drive path or update the dataset paths in the notebook.

## Notebook Structure

The main notebook is:

```text
Standing Dead Tree Segmentation from Aerial Imagery.ipynb
```

The workflow is organized into the following parts:

* **Part 1. Environment Setup**

  * Mount Google Drive.
  * Extract the dataset.
  * Inspect the dataset folder structure.
  * Visualize sample RGB, NIR-style, and mask data.

* **Part 2. Dataset Splitting and File List Preparation**

  * Match RGB, NRG, and mask files.
  * Create train-validation file lists.
  * Define custom PyTorch dataset and DataLoader components.

* **Part 3. Individual Model Training and Evaluation**

  * Train and evaluate SVM, UNet, DeepLabV3, Fast-SCNN-style, and ADA-Net-style models.
  * Save model checkpoints, logs, predictions, and visualizations.
  * Generate confusion matrices and prediction examples.

* **Part 4. Comparative Analysis of Segmentation Models**

  * Compare final logged metrics.
  * Compare training curves across neural network models.
  * Discuss segmentation quality, model complexity, and efficiency trade-offs.

* **Part 5. Final Packaging and Project Archival**

  * Package generated logs, checkpoints, predictions, and visual outputs into a ZIP archive.

## Models Implemented

### SVM Baseline

The SVM model is used as a classical pixel-level baseline. It uses RGB and NRG-derived features to classify pixels as background or tree.

This model provides a simple reference point for comparing traditional machine learning with deep learning models.

### UNet

UNet is implemented as a convolutional encoder-decoder model with skip connections. It is used as a standard deep learning baseline for binary segmentation.

The model is configured to support RGB+NIR-style input and outputs a single-channel probability map.

### DeepLabV3

DeepLabV3 is implemented with a ResNet-50 backbone. It uses atrous convolution to capture wider spatial context and is trained as a semantic segmentation model.

In the current notebook, DeepLabV3 provides the strongest overall segmentation performance among the main deep learning models.

### Fast-SCNN-Style Model

The Fast-SCNN-style model is a lightweight segmentation baseline. It uses a compact downsampling structure and a simple classifier head.

This model is mainly used to study the trade-off between training speed, inference efficiency, and segmentation quality.

### ADA-Net-Style Model

The ADA-Net-style model is a custom compact encoder-decoder model. It uses a lightweight convolutional encoder and a transposed-convolution decoder for binary segmentation.

This model provides an additional efficiency-oriented comparison point against UNet, DeepLabV3, and Fast-SCNN-style models.

## Evaluation Metrics

The project uses the following metrics where available:

* **Pixel Accuracy**

  * Measures the overall proportion of correctly classified pixels.

* **Dice Score**

  * Measures overlap between predicted tree regions and ground-truth masks.

* **Intersection-over-Union (IoU)**

  * Measures region-level overlap between prediction and target.

* **Precision**

  * Measures how many predicted tree pixels are correct.

* **Recall**

  * Measures how many true tree pixels are detected.

For this task, I interpret **Dice** and **IoU** more carefully than pixel accuracy because the masks are highly imbalanced and background pixels dominate the image.

## Main Findings

From the current notebook results:

* **DeepLabV3** gives the strongest overall segmentation performance.
* **UNet** serves as a useful standard encoder-decoder baseline.
* **ADA-Net-style model** provides a reasonable lightweight comparison point.
* **Fast-SCNN-style model** highlights the speed-quality trade-off but struggles with sparse tree regions.
* **SVM** is useful as a simple classical baseline but is limited in complex aerial scenes.

The qualitative results show that the task remains challenging. Common issues include:

* Missing small or sparse tree regions.
* Predicting extra foreground in complex background areas.
* Confusion in dense vegetation or visually noisy scenes.

## Important Notes on Comparison

I do not treat the results as a perfectly controlled benchmark.

Some models use different:

* Input settings.
* Training splits.
* Evaluation routines.
* Loss functions.
* Model capacities.

Therefore, the comparison should be understood as a **diagnostic project-level study**, not a strictly controlled leaderboard. The purpose is to understand model behaviour and practical trade-offs across different segmentation approaches.

## Repository Structure

A typical project structure is:

```text
.
├── README.md
├── Standing Dead Tree Segmentation from Aerial Imagery.ipynb
├── outputs/
│   ├── svm_LinearSVC/
│   ├── unet/
│   ├── deeplabv3/
│   ├── fastscnn/
│   └── adanet/
├── checkpoints/
│   ├── unet_best.pth
│   ├── deeplabv3_best.pth
│   ├── fastscnn_best.pth
│   └── adanet_best.pth
├── training_log_svm.csv
├── training_log_unet.csv
├── training_log_deeplabv3.csv
├── training_log_fastscnn.csv
└── training_log_adanet.csv
```

Some folders and files are generated after running the notebook and may not be included in the repository.

## Requirements

The notebook is designed to run in Google Colab with GPU support.

Main dependencies include:

```text
Python
PyTorch
Torchvision
OpenCV
NumPy
Pandas
Matplotlib
Scikit-learn
Pillow
tqdm
```

## How to Run

1. Open the notebook in Google Colab.
2. Enable GPU runtime.
3. Mount Google Drive.
4. Place the dataset archive in the expected Google Drive path, or update the dataset path in the notebook.
5. Run the notebook from top to bottom.
6. Review generated logs, plots, predictions, and model checkpoints.
7. Use the final packaging step to download the project outputs as a ZIP archive.

## Outputs

The notebook generates:

* Training logs.
* Model checkpoints.
* Prediction masks.
* Confusion matrices.
* Training curve plots.
* Final comparison charts.
* Qualitative prediction examples.
* A packaged ZIP archive of the project workspace.

## Project Status

This project is complete as a comparative segmentation study. Future improvements could include:

* Using a fully unified train-validation-test split across all models.
* Standardizing all models under the same input modality.
* Adding stronger data augmentation.
* Testing pretrained segmentation backbones with better domain adaptation.
* Improving lightweight architectures for sparse foreground segmentation.
* Running repeated experiments with fixed seeds for more reliable comparison.

## Author

Implemented and organized by **Zhihao Wu** as a computer vision and remote sensing segmentation project for GitHub portfolio documentation and future project extension.
