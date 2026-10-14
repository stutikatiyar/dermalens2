# DermaLens 🧠🩺

<p align="center">
  <b>A Domain-Aware Dermatology Image Enhancement & Quality-Aware Validation Framework</b><br>
  <i>Adaptive Image Restoration • Multi-Metric Distortion Validation • Downstream AI Analysis</i>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/React-61DAFB?style=for-the-badge&logo=react&logoColor=black" alt="React">
  <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV">
  <img src="https://img.shields.io/badge/Tailwind_CSS-38B2AC?style=for-the-badge&logo=tailwind-css&logoColor=white" alt="Tailwind CSS">
</p>

---

## 📑 Table of Contents

1. [Executive Summary](#-executive-summary)
2. [Problem Statement & Domain Challenges](#-problem-statement--domain-challenges)
    - [The Teledermatology Degradation Problem](#the-teledermatology-degradation-problem)
    - [Why Traditional Preprocessing Fails](#why-traditional-preprocessing-fails)
3. [The DermaLens Paradigm](#-the-dermalens-paradigm)
    - [Validation-First Enhancement](#validation-first-enhancement)
    - [The Safety Fallback Guarantee](#the-safety-fallback-guarantee)
4. [System Architecture](#-system-architecture)
    - [High-Level Data Flow](#high-level-data-flow)
    - [Component Architecture Breakdown](#component-architecture-breakdown)
5. [Deep Dive: Pipeline Execution Stages](#-deep-dive-pipeline-execution-stages)
    - [Stage 1: Image Acquisition & Pre-Check](#stage-1-image-acquisition--pre-check)
    - [Stage 2: Multi-Dimensional Quality Profiling](#stage-2-multi-dimensional-quality-profiling)
    - [Stage 3: Adaptive Candidate Generation](#stage-3-adaptive-candidate-generation)
    - [Stage 4: Validation & Boundary Filtering](#stage-4-validation--boundary-filtering)
    - [Stage 5: Candidate Selection & Safety Fallback](#stage-5-candidate-selection--safety-fallback)
    - [Stage 6: Downstream Classifier Evaluation](#stage-6-downstream-classifier-evaluation)
6. [Mathematical & Algorithmic Foundations](#-mathematical--algorithmic-foundations)
    - [Variance of Laplacian (Blur Metrics)](#variance-of-laplacian-blur-metrics)
    - [LAB Color Space Illumination Analysis](#lab-color-space-illumination-analysis)
    - [Contrast Limited Adaptive Histogram Equalization (CLAHE)](#contrast-limited-adaptive-histogram-equalization-clahe)
    - [Structural Similarity Index Measure (SSIM)](#structural-similarity-index-measure-ssim)
    - [Color Delta Metric ($\Delta E^*_{\text{ab}}$)](#color-delta-metric-\delta-e_textab)
7. [Deep Learning Restoration Models](#-deep-learning-restoration-models)
    - [Real-ESRGAN (General $4\times$ Generative Restoration)](#real-esrgan-general-4\times-generative-restoration)
    - [EDSR (Enhanced Deep Residual Networks $2\times$)](#edsr-enhanced-deep-residual-networks-2\times)
    - [ISIC ResNet18 Abnormality Classification](#isic-resnet18-abnormality-classification)
8. [Comprehensive Tech Stack](#-comprehensive-tech-stack)
9. [Directory Structure & Module Reference](#-directory-structure--module-reference)
10. [API Reference & Schema Documentation](#-api-reference--schema-documentation)
    - [`POST /api/upload`](#post-apiupload)
    - [`GET /api/health`](#get-apihealth)
11. [Installation & Setup Guide](#-installation--setup-guide)
    - [Prerequisites](#prerequisites)
    - [Backend Setup](#backend-setup)
    - [Frontend Setup](#frontend-setup)
12. [Evaluation Metrics & Benchmark Protocols](#-evaluation-metrics--benchmark-protocols)
13. [Future Roadmap](#-future-roadmap)
14. [Limitations & Clinical Disclaimer](#-limitations--clinical-disclaimer)
15. [Author & Institutional Credits](#-author--institutional-credits)

---

## 🔬 Executive Summary

**DermaLens** is an open-source, domain-aware image enhancement and quality-validation framework engineered specifically for dermatological AI systems. 

In real-world clinical and teledermatology settings, consumer-grade smartphone sensors produce images with significant quality variances—ranging from extreme underexposure and motion blur to compression artifacts and sensor noise. Standard computer vision preprocessing pipelines typically apply fixed, indiscriminate filters (such as global histogram equalization or universal bilateral blurring) across all inputs. When applied to clinical imagery, these fixed transformations frequently introduce false diagnostic artifacts, alter subtle lesion pigmentation, and distort boundary margins necessary for differential diagnosis.

DermaLens re-frames medical image enhancement as a **decision-making and validation-first workflow**. Instead of immediately modifying an input image, DermaLens:
1. Profiles the physical image quality across brightness, contrast, blur, and noise dimensions.
2. Dynamically generates targeted enhancement candidates using localized signal-processing algorithms and deep super-resolution neural networks.
3. Evaluates every candidate using strict structural preservation metrics and color variance boundaries.
4. Accepts only the best safe candidate, or **reverts directly to the original image** if all candidates introduce unacceptable distortion.
5. Measures downstream AI classification performance across both original and enhanced images to verify prediction stability.

---

## 🚨 Problem Statement & Domain Challenges

### The Teledermatology Degradation Problem

Remote clinical assessment and consumer mobile health platforms rely heavily on unstructured image acquisition. Unlike dermoscopic imaging—which uses specialized polarized lighting and fixed magnification lenses—smartphone imagery captured by patients or non-specialist clinicians encounters severe environmental variables:

* **Low Illumination:** Ambient light deficits obscure subtle erythematous or hyperpigmented boundaries.
* **Handshake Motion Blur:** Low shutter speeds under indoor lighting introduce high spatial frequency loss.
* **Sensor Thermal Noise:** High ISO settings on smaller mobile camera sensors corrupt uniform skin textures with granular noise patterns.
* **JPEG & Compression Artifacts:** Bandwidth-saving compression algorithms erase fine structural details, creating blocky false edges.
* **Inconsistent Color Temperature:** Uncalibrated white balances shift skin tones, complicating color-based diagnostic scoring (such as the ABCD rule for melanoma).

| Artifact Type | Physical Cause | Impact on Downstream Models |
| :--- | :--- | :--- |
| **Underexposure** | Low ambient lighting, backlighting | Shifts activation maps away from lesion center; reduces confidence |
| **Motion Blur** | Low shutter speed, unstable camera | Erases fine peripheral boundary contours; causes false lesion sizing |
| **Sensor Noise** | High ISO settings | Induces spurious gradient spikes in convolutional filters |
| **Over-Saturation** | Direct flash reflection | Saturates pixel intensities, destroying color gradient information |

### Why Traditional Preprocessing Fails

Generic image preprocessing pipelines operate under the assumption that increasing global contrast or high-frequency sharpness yields a "better" image. In medical computer vision, this assumption breaks down completely:

1. **Over-Sharpening Creates Hallucinated Borders:** Unsharp masking or aggressive Laplace filters amplify skin texture lines into thick, dark borders that downstream neural networks misinterpret as irregular lesion margins.
2. **Global Histogram Equalization Distorts Pigmentation:** Stretching contrast across the entire image channel alters true color representation, making benign lesions appear necrotic or inflamed.
3. **Deep Restoration Hallucinations:** Generative super-resolution models (e.g., GAN-based architectures) can introduce hallucinated visual patterns that mimic clinical structures not present in the patient's actual skin.

> **Crucial Diagnostic Axiom:** *Higher visual sharpness or artificially boosted model confidence does not imply clinical correctness.*

---

## 🛡️ The DermaLens Paradigm

### Validation-First Enhancement

DermaLens operates under a strict **Validation-First** paradigm. Enhancement is not treated as a deterministic image transformation function $f(x) = y$, but as an optimization and candidate filtering problem:

$$\text{Select } C^* \in \{C_0, C_1, \dots, C_n\} \quad \text{subject to} \quad \mathcal{V}(C_i, C_0) \le \tau$$

Where $C_0$ represents the unassisted original image, $C_1 \dots C_n$ represent generated enhancement candidates, $\mathcal{V}$ represents a multi-metric distortion validator function, and $\tau$ denotes strict safety thresholds.
+-----------------------------------------------------------------------+
|                         DERMALENS PARADIGM                            |
+-----------------------------------------------------------------------+
|                                                                       |
|   [Raw Input Image]                                                   |
|           |                                                           |
|           v                                                           |
|   +-------------------+                                               |
|   | Quality Profiling | --> Measures Brightness, Blur, Noise, Contrast|
|   +-------------------+                                               |
|           |                                                           |
|           v                                                           |
|   +-------------------+                                               |
|   | Targeted Candidate| --> Generates CLAHE, Gamma, Real-ESRGAN,     |
|   |    Generation     |     and EDSR Candidate Variations             |
|   +-------------------+                                               |
|           |                                                           |
|           v                                                           |
|   +-------------------+                                               |
|   |  Multi-Metric     | --> Evaluates SSIM, Delta E, Noise Spike,     |
|   | Validation Layer  |     and Blur Differential Bounds              |
|   +-------------------+                                               |
|           |                                                           |
|     +-----+-----+                                                     |
|     |           |                                                     |
|  [PASSED]   [REJECTED ALL]                                            |
|     |           |                                                     |
|     v           v                                                     |
| [Best Valid] [Original Image Preserved]                               |
|     \           /                                                     |
|      v         v                                                      |
|   +-------------------+                                               |
|   | Downstream ISIC   | --> Measures Confidence & Prediction Shifts   |
|   | Classifier Model  |                                               |
|   +-------------------+                                               |
+-----------------------------------------------------------------------+


### The Safety Fallback Guarantee

If an input image is already optimal, or if every generated candidate exceeds the distortion boundaries (e.g., introducing a color shift $\Delta E > 12.0$ or dropping Structural Similarity $\text{SSIM} < 0.85$), DermaLens activates its **Safety Fallback Mechanism**.

The system completely rejects all modified images and passes the unassisted original image $C_0$ directly to the downstream classification tier, guaranteeing that image quality is never degraded below its input state.

---

## ⚙️ System Architecture

### High-Level Data Flow

DermaLens is divided into a lightweight client interface (**React 18 + Tailwind CSS + Framer Motion**) and an asynchronous processing engine (**FastAPI + PyTorch + OpenCV**).

```text
+-----------------------------------------------------------------------------------+
|                                 CLIENT SIDE                                       |
|                                                                                   |
|  +--------------------+      HTTP POST (Multipart)     +-----------------------+  |
|  | User Image Upload  | -----------------------------> | React UI State Engine |  |
|  +--------------------+                                +-----------------------+  |
|                                                                    |              |
+--------------------------------------------------------------------|--------------+
                                                                     |
                                                           JSON / Base64 Payload
                                                                     |
+--------------------------------------------------------------------|--------------+
|                                 SERVER SIDE                        v              |
|                                                                                   |
|  +-----------------------------------------------------------------------------+  |
|  |                             FastAPI Route Handler                           |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                        Quality Analyzer Module                              |  |
|  |  - LAB Illumination    - Laplacian Blur      - Michelson Contrast           |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                     Adaptive Enhancement Pipeline                           |  |
|  |  - Gamma Adjust        - CLAHE Contrast      - Fast Denoising               |  |
|  |  - Real-ESRGAN Model   - EDSR $2\times$ Engine                              |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                       Validation & Metrics Layer                            |  |
|  |  - SSIM Check          - Color Shift ($\Delta E$) - Sharpness Delta           |  |
|  +-----------------------------------------------------------------------------+  |
|                                        |                                          |
|                                        v                                          |
|  +-----------------------------------------------------------------------------+  |
|  |                  Downstream Classifier (ResNet18 ISIC)                      |  |
|  |  - Original Prediction - Enhanced Prediction - Confidence Delta ($\Delta C$)  |  |
|  +-----------------------------------------------------------------------------+  |
+-----------------------------------------------------------------------------------+
Component Architecture Breakdown
The framework comprises five isolated core processing modules:

Plaintext
backend/app/ml/
├── quality_analyzer.py   <-- Computes raw image quality indicators
├── enhancement.py        <-- Generates deterministic candidate transformations
├── superres_enhancer.py  <-- Handles deep learning super-resolution models
├── metrics.py            <-- Calculates structural & perceptual distortion
├── derm_model.py         <-- Manages PyTorch ISIC classification models
└── pipeline.py           <-- Orchestrates multi-stage processing & selection
🧪 Deep Dive: Pipeline Execution Stages
Stage 1: Image Acquisition & Pre-Check
The uploaded image file is parsed in memory via OpenCV and PIL. The pre-check layer inspects dimension constraints, spatial orientation, color space definitions, and file integrity before initiating processing.

Stage 2: Multi-Dimensional Quality Profiling
The image is passed through quality_analyzer.py to extract quantitative metrics:

Illumination Index (
L 
∗
 ): Extracted from CIELAB color space. Images with 
L 
∗
 <40.0 are flagged as underexposed; images with 
L 
∗
 >85.0 are flagged as overexposed.

Sharpness Variance (
σ 
Lap
2
​
 ): Evaluates high-frequency spatial derivatives. Values below 
100.0 indicate significant blur.

Contrast Score (
C 
Michelson
​
 ): Evaluates luminosity spreads across local pixel neighborhoods.

High-Frequency Noise (
σ 
N
​
 ): Estimated using fast median absolute deviation over high-frequency wavelet sub-bands.

Stage 3: Adaptive Candidate Generation
Depending on the quality profile, enhancement.py and superres_enhancer.py conditionally instantiate targeted processing operations:

Plaintext
Input Profile Flags
       |
       +---> [Underexposed] ------> Candidates: Adaptive Gamma (\gamma=1.5, 2.0), LAB CLAHE
       |
       +---> [Low Contrast] ------> Candidates: CLAHE (Clip Limit=2.0, Tile Grid=8x8)
       |
       +---> [Blur / Low-Res] -----> Candidates: Real-ESRGAN x4, EDSR x2 Super Resolution
       |
       +---> [Noisy Texture] ------> Candidates: Fast Non-Local Means Denoising (h=3, templateWindowSize=7)
Stage 4: Validation & Boundary Filtering
Every generated candidate image 
C 
i
​
  is paired with the original input 
C 
0
​
  inside metrics.py to verify distortion bounds:

Structural Similarity Index (SSIM): Rejects candidates where 
SSIM(C 
0
​
 ,C 
i
​
 )<0.82.

Color Shift (
ΔE 
ab
∗
​
 ): Rejects candidates where mean color deviation in CIELAB space exceeds 
12.0.

Noise Growth Differential (
Δσ 
N
​
 ): Rejects candidates where noise estimation increases by over 
35% relative to original sharpness gain.

Plaintext
Candidate Validation Result Matrix Example:

Candidate ID | Transform Type  | SSIM  | Delta E | Sharpness Gain | Decision
-------------|-----------------|-------|---------|----------------|----------
Candidate 1  | CLAHE (Clip 2.0)| 0.942 |  3.12   |    +24.5%      | ACCEPTED
Candidate 2  | Gamma Correction| 0.910 |  5.80   |    +12.1%      | ACCEPTED
Candidate 3  | Aggressive Sharpen|0.745| 14.20   |   +140.2%      | REJECTED (High Distortion)
Candidate 4  | Real-ESRGAN x4  | 0.865 |  8.40   |    +88.3%      | ACCEPTED
Stage 5: Candidate Selection & Safety Fallback
Accepted candidates are ranked using a multi-attribute utility function:

U(C 
i
​
 )=w 
1
​
 ⋅SSIM(C 
0
​
 ,C 
i
​
 )+w 
2
​
 ⋅ 
S
^
 (C 
i
​
 )−w 
3
​
 ⋅ΔE(C 
0
​
 ,C 
i
​
 )
Where 
S
^
 (C 
i
​
 ) represents normalized sharpness gain. If no candidates pass validation, 
U(C 
0
​
 ) is selected as the default output.

Stage 6: Downstream Classifier Evaluation
The selected candidate (or preserved original) is evaluated by derm_model.py—a PyTorch implementation of a ResNet18 model trained on the ISIC dermatological dataset.

The system evaluates:

Original Prediction Vector (
P 
0
​
 ): Softmax probabilities across target abnormality classes.

Enhanced Prediction Vector (
P 
e
​
 ): Softmax probabilities using the selected candidate image.

Confidence Delta (
ΔC): Difference in top class prediction confidence: 
ΔC=max(P 
e
​
 )−max(P 
0
​
 ).

🧮 Mathematical & Algorithmic Foundations
Variance of Laplacian (Blur Metrics)
Spatial blur is quantified by calculating the variance of the image smoothed with the isotropic 2D Laplacian operator 
L:

L(x,y)= 
∂x 
2
 
∂ 
2
 I
​
 + 
∂y 
2
 
∂ 
2
 I
​
 
Discrete convolution is performed using the standard 
3×3 kernel:

K 
Laplacian
​
 = 

​
  
0
1
0
​
  
1
−4
1
​
  
0
1
0
​
  

​
 
The blur metric score 
S 
blur
​
  is defined as the variance of the resulting response matrix:

μ= 
M⋅N
1
​
  
x=1
∑
M
​
  
y=1
∑
N
​
 L(x,y)
S 
blur
​
 =σ 
Lap
2
​
 = 
M⋅N
1
​
  
x=1
∑
M
​
  
y=1
∑
N
​
 (L(x,y)−μ) 
2
 
A low variance value (
σ 
Lap
2
​
 <100.0) corresponds to a lack of high-frequency edge transitions, indicating a blurry image.

LAB Color Space Illumination Analysis
To isolate illumination intensity from chrominance details, RGB images are transformed into the CIE 
L 
∗
 a 
∗
 b 
∗
  color space.

Conversion from linear sRGB to XYZ coordinates:


​
  
X
Y
Z
​
  

​
 = 

​
  
0.412456
0.212672
0.019334
​
  
0.357576
0.715152
0.119192
​
  
0.180437
0.072175
0.950301
​
  

​
  

​
  
R 
linear
​
 
G 
linear
​
 
B 
linear
​
 
​
  

​
 
Transformation from XYZ to CIELAB 
L 
∗
  (Luminance Channel):

L 
∗
 =116⋅f( 
Y 
n
​
 
Y
​
 )−16
Where 
f(t) is defined as:

f(t)={ 
t 
1/3
 
3
1
​
 ( 
6
29
​
 ) 
2
 t+ 
116
16
​
 
​
  
if t>( 
29
6
​
 ) 
3
 
otherwise
​
 
The mean luminance 
L
ˉ
  
∗
 = 
N
1
​
 ∑L 
∗
  provides a stable measure of image exposure that is uncorrupted by skin redness or pigmentation.

Contrast Limited Adaptive Histogram Equalization (CLAHE)
Standard histogram equalization calculates a global transformation to redistribute pixel intensities across the full range 
[0,255]. In medical images, this stretches background noise in uniform skin regions.

CLAHE solves this by dividing the 
L 
∗
  channel into contextual regions (tiles of size 
8×8). Within each tile, the local histogram 
h(k) is calculated and clipped at a pre-set threshold 
β:

h 
clipped
​
 (k)=min(h(k),β)
The excess clipped pixels 
N 
clip
​
  are uniformly redistributed across all histogram bins:

L 
redistribute
​
 = 
256
N 
clip
​
 
​
 
Adjacent tiles are joined using bilinear interpolation to eliminate artificial tile boundary seams.

Structural Similarity Index Measure (SSIM)
The structural similarity between the original image 
x and candidate image 
y is computed across localized windows of size 
N×N:

SSIM(x,y)= 
(μ 
x
2
​
 +μ 
y
2
​
 +C 
1
​
 )(σ 
x
2
​
 +σ 
y
2
​
 +C 
2
​
 )
(2μ 
x
​
 μ 
y
​
 +C 
1
​
 )(2σ 
xy
​
 +C 
2
​
 )
​
 
Where:

μ 
x
​
 ,μ 
y
​
  are local pixel sample means.

σ 
x
2
​
 ,σ 
y
2
​
  are local pixel sample variances.

σ 
xy
​
  is the local cross-covariance between 
x and 
y.

C 
1
​
 =(k 
1
​
 L) 
2
 ,C 
2
​
 =(k 
2
​
 L) 
2
  are stabilization constants (
k 
1
​
 =0.01,k 
2
​
 =0.03,L=255).

Color Delta Metric (
ΔE 
ab
∗
​
 )
Perceptual color alteration between original image pixel 
p 
1
​
  and candidate pixel 
p 
2
​
  is computed using Euclidean distance in CIELAB color space:

ΔE 
ab
∗
​
 = 
(L 
2
∗
​
 −L 
1
∗
​
 ) 
2
 +(a 
2
∗
​
 −a 
1
∗
​
 ) 
2
 +(b 
2
∗
​
 −b 
1
∗
​
 ) 
2
 

​
 
DermaLens computes the mean frame color shift 
ΔE
 :

ΔE
 = 
M⋅N
1
​
  
i=1
∑
M
​
  
j=1
∑
N
​
 ΔE 
ab
∗
​
 (i,j)
Candidate enhancements yielding 
ΔE
 >12.0 are automatically rejected to maintain diagnostic integrity.

🤖 Deep Learning Restoration Models
Real-ESRGAN (General 
4× Generative Restoration)
DermaLens uses Real-ESRGAN (realesr-general-x4v3.pth) to restore severely degraded low-resolution skin images.

Unlike standard ESRGAN trained on synthetic bicubic downsampling, Real-ESRGAN uses a higher-order degradation modeling process that simulates complex real-world artifacts:

D=(D 
2
 ∘D 
1
 )(I 
clean
​
 )
Where each degradation step 
D 
k
  includes sequential blur filters, generalized Gaussian noise, 3D ring artifacts, and lossy JPEG compression. The generator network uses a Residual-in-Residual Dense Block (RRDB) backbone to reconstruct high-resolution features without introducing over-smoothed surfaces.

EDSR (Enhanced Deep Residual Networks 
2×)
For scenes where generative models risk introducing synthetic artifacts, DermaLens uses EDSR (EDSR_x2.pb).

EDSR optimizes super-resolution performance by removing unnecessary Batch Normalization layers from traditional ResNet architectures:

Plaintext
Standard ResNet Block:
[ Conv ] ---> [ Batch Norm ] ---> [ ReLU ] ---> [ Conv ] ---> [ Batch Norm ] ---> (+)

EDSR Residual Block:
[ Conv ] ---> [ ReLU ] ---> [ Conv ] ---> [ Constant Scaling (0.1) ] -----------> (+)
By removing Batch Normalization:

GPU memory utilization is reduced by approximately 
40%, enabling processing of higher spatial dimensions.

Range flexibility is maintained across diverse lighting conditions without normalizing away subtle clinical details.

ISIC ResNet18 Abnormality Classification
To evaluate downstream model behavior, DermaLens includes a fine-tuned ResNet18 model trained on the International Skin Imaging Collaboration (ISIC) dataset.

Plaintext
Input Image (224 x 224 x 3)
         │
         ▼
[ Conv2D (7x7, stride 2) ] ──> [ BatchNorm ] ──> [ ReLU ] ──> [ MaxPool (3x3) ]
         │
         ▼
[ BasicBlock Layer 1 ] (2 Blocks, 64 Filters)
         │
         ▼
[ BasicBlock Layer 2 ] (2 Blocks, 128 Filters, Stride 2)
         │
         ▼
[ BasicBlock Layer 3 ] (2 Blocks, 256 Filters, Stride 2)
         │
         ▼
[ BasicBlock Layer 4 ] (2 Blocks, 512 Filters, Stride 2)
         │
         ▼
[ Adaptive Avg Pooling (1x1) ]
         │
         ▼
[ Linear Classifier Layer ] ──> Softmax Output Probability Distribution
The classifier provides secondary operational metrics:

ΔC=C(Enhanced)−C(Original)
Where 
C(I)=max 
k
​
 P(Y=k∣I) represents top-class prediction confidence.

💻 Comprehensive Tech Stack
Backend System
Core Language: Python 3.10+

API Framework: FastAPI (0.104.0+)

ASGI Web Server: Uvicorn (0.24.0+)

Computer Vision Operations: OpenCV (4.8.1+), Pillow (10.1.0+), Scikit-Image (0.22.0+)

Deep Learning Runtime: PyTorch (2.1.0+), Torchvision (0.16.0+)

Numerical Utilities: NumPy (1.26.0+), SciPy (1.11.0+)

Frontend Interface
UI Framework: React 18 (18.2.0)

Build System: Vite (5.0.0)

Styling System: Tailwind CSS (3.3.0)

Animation Engine: Framer Motion (10.16.0)

Icons: Lucide React (0.292.0)

HTTP Transport: Axios (1.6.0)

📂 Directory Structure & Module Reference
Plaintext
dermalens/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI application initialization & middleware
│   │   │
│   │   ├── ml/                        # Machine learning & computer vision module
│   │   │   ├── __init__.py
│   │   │   ├── quality_analyzer.py    # Multi-dimensional image quality profiling
│   │   │   ├── enhancement.py         # Deterministic candidate generation methods
│   │   │   ├── superres_enhancer.py   # Real-ESRGAN and EDSR model runners
│   │   │   ├── metrics.py             # Validation routines (SSIM, Delta E, Blur)
│   │   │   ├── derm_model.py          # PyTorch ResNet18 ISIC model runner
│   │   │   ├── pipeline.py            # Master orchestration & fallback router
│   │   │   │
│   │   │   ├── models/                # Trained model weights directory
│   │   │   │   └── isic_resnet18.pth  # ISIC classifier model parameters
│   │   │   │
│   │   │   ├── realesrgan_weights/    # Super-resolution neural network weights
│   │   │   │   └── realesr-general-x4v3.pth
│   │   │   │
│   │   │   └── weights/               # OpenCV DNN super-resolution models
│   │   │       └── EDSR_x2.pb
│   │   │
│   │   └── routes/                    # API route definitions
│   │       ├── __init__.py
│   │       └── upload.py              # Endpoint for image processing & inference
│   │
│   ├── requirements.txt               # Backend Python dependency specifications
│   └── test_realesrgan.py             # Standalone test runner for Real-ESRGAN
│
└── frontend/
    └── dermalens/
        ├── public/                    # Static public assets
        ├── src/
        │   ├── api/                   # API interaction layer
        │   │   └── api.js             # Base Axios configuration & endpoint calls
        │   │
        │   ├── assets/                # Structural icons and brand vectors
        │   │   ├── hero.png
        │   │   ├── react.svg
        │   │   └── vite.svg
        │   │
        │   ├── components/            # React UI components
        │   │   ├── AIOrb.jsx          # Animated processing indicator
        │   │   ├── AIProcessing.jsx   # Status display during analysis
        │   │   ├── BackgroundGlow.jsx # Background visual layout effects
        │   │   ├── HeroSection.jsx    # Hero landing display section
        │   │   ├── MetricsCard.jsx    # Structural metric readout panel
        │   │   ├── Navbar.jsx         # System navigation bar
        │   │   ├── ResultSection.jsx  # Interactive before/after split view
        │   │   ├── UploadBox.jsx      # Drag-and-drop file uploader
        │   │   └── UploadSection.jsx  # Primary operational upload panel
        │   │
        │   ├── pages/
        │   │   └── Home.jsx           # Main application view container
        │   │
        │   ├── App.css                # Global CSS overrides
        │   ├── App.jsx                # Application root component
        │   ├── index.css              # Tailwind directive imports
        │   └── main.jsx               # React DOM entry point
        │
        ├── package.json               # Node modules and build script definitions
        ├── tailwind.config.js         # Tailwind system customization options
        └── vite.config.js             # Vite development server configuration
🔌 API Reference & Schema Documentation
POST /api/upload
Primary operational endpoint for image quality analysis, candidate generation, validation, and downstream classifier benchmarking.

Request Specification
Content-Type: multipart/form-data

Body Parameters:

file (Binary, Required): Image file payload (image/jpeg, image/png, or image/webp).

Response Schema (200 OK)
JSON
{
  "status": "success",
  "processing_time_seconds": 1.428,
  "original_metrics": {
    "brightness": 58.32,
    "sharpness": 4.74,
    "contrast": 6.89,
    "noise": 2.14,
    "is_underexposed": true,
    "is_blurry": true
  },
  "selected_candidate": {
    "candidate_id": "candidate_clahe_lab",
    "enhancement_type": "CLAHE (LAB Space)",
    "was_fallback_used": false,
    "metrics": {
      "brightness": 87.64,
      "sharpness": 8.54,
      "contrast": 8.28,
      "noise": 2.21
    },
    "validation_checks": {
      "ssim": 0.945,
      "color_shift_delta_e": 3.42,
      "sharpness_gain_percent": 80.17,
      "passed_validation": true
    }
  },
  "downstream_eval": {
    "model_name": "ISIC_ResNet18",
    "original_prediction": {
      "label": "Melanocytic Nevus",
      "confidence": 0.768
    },
    "enhanced_prediction": {
      "label": "Melanocytic Nevus",
      "confidence": 0.794
    },
    "confidence_delta": 0.026
  },
  "images": {
    "original": "data:image/jpeg;base64,/9j/4AAQSkZJRg...",
    "enhanced": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
}
GET /api/health
Health check and status route for system monitors and load balancers.

Response Schema (200 OK)
JSON
{
  "status": "healthy",
  "cuda_available": false,
  "loaded_models": {
    "isic_resnet18": true,
    "realesrgan": true,
    "edsr": true
  }
}
⚡ Installation & Setup Guide
Prerequisites
Ensure your local environment meets the following baseline requirements:

Python: Version 3.10 or higher

Node.js: Version 18.0.0 or higher

Package Managers: pip and npm

Hardware Acceleration (Optional): NVIDIA GPU with CUDA support for accelerated super-resolution model inference.

Backend Setup
Navigate to the backend directory:

Bash
cd backend
Initialize a clean Python virtual environment:

Bash
python -m venv venv
Activate the virtual environment:

Windows (PowerShell):

PowerShell
.\venv\Scripts\Activate.ps1
Linux / macOS:

Bash
source venv/bin/activate
Upgrade baseline deployment tools:

Bash
pip install --upgrade pip setuptools wheel
Install backend dependencies:

Bash
pip install -r requirements.txt
Run the FastAPI application server:

Bash
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
Verify operational API deployment:
Open your browser and navigate to http://127.0.0.1:8000/docs to access the interactive OpenAPI / Swagger UI documentation.

Frontend Setup
Open a new terminal session and navigate to the frontend directory:

Bash
cd frontend/dermalens
Install frontend dependencies:

Bash
npm install
Configure system API environment variables (Optional):
Create a .env.local file inside frontend/dermalens/ if your backend is running on a non-default host or port:

Code snippet
VITE_API_BASE_URL=[http://127.0.0.1:8000](http://127.0.0.1:8000)
Launch the Vite local development server:

Bash
npm run dev
Access the user interface:
Open your browser and visit http://localhost:5173.

📊 Evaluation Metrics & Benchmark Protocols
DermaLens uses a multi-faceted evaluation strategy to measure restoration performance:

Plaintext
+-----------------------------------------------------------------------------------------+
|                                EVALUATION MATRIX METRICS                                |
+-------------------------------+---------------------------------------------------------+
| Metric Category               | Evaluated Attributes                                    |
+-------------------------------+---------------------------------------------------------+
| Image Quality Metrics         | • Luminance ($L^*$), Laplacian Variance ($\sigma^2$),   |
|                               |   Michelson Contrast, Noise Power Spectrum              |
|                               |                                                         |
| Structural Fidelity Metrics   | • SSIM, Feature Similarity (FSIM), Peak Signal-to-Noise |
|                               |   Ratio (PSNR)                                          |
|                               |                                                         |
| Color Integrity Metrics       | • CIELAB $\Delta E^*_{\text{ab}}$, Perceptual Color     |
|                               |   Angle Variation                                       |
|                               |                                                         |
| Downstream Diagnostic Metrics | • Softmax Classification Cross-Entropy, Prediction     |
|                               |   Stability, Confidence Differential ($\Delta C$)       |
+-------------------------------+---------------------------------------------------------+
Benchmark Protocol Rules
Paired Synthetic Evaluation: Synthetically degrade clean ISIC images with known noise variances and measure post-restoration SSIM recovery.

Unpaired Real-World Evaluation: Process real-world mobile camera uploads and verify that candidate rejection rates accurately track extreme color shifts and structural distortions.

🔮 Future Roadmap
[ ] LPIPS Perceptual Loss Integration: Implement Learned Perceptual Image Patch Similarity (LPIPS) to complement SSIM in candidate validation.

[ ] Grad-CAM Explainability Overlay: Integrate Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize neural network attention regions before and after image enhancement.

[ ] Expanded Disease Diagnostic Classifiers: Extend the downstream analysis layer to evaluate additional skin pathologies, including burn severity grading and chronic wound boundaries.

[ ] Mobile & Edge Runtime Optimization: Quantize PyTorch and super-resolution models to ONNX and TensorFlow Lite formats for on-device mobile execution.

[ ] Cross-Camera Color Calibration: Implement automatic color-constancy algorithms using skin-tone reference distributions.

⚠️ Limitations & Clinical Disclaimer
Research Prototype Status: DermaLens is an experimental computer vision framework designed solely for research, prototyping, and evaluation. It is not a certified diagnostic medical device and is not intended to replace professional clinical judgment.

Generative Artifact Risks: While the validation layer rejects high-distortion outputs, generative super-resolution algorithms (such as Real-ESRGAN) can occasionally synthesize fine, hyper-realistic textures that do not reflect true patient anatomy.

Dataset Bias Constraints: The downstream classifier is trained on curated datasets (such as ISIC) and may display reduced classification stability on underrepresented skin phototypes or rare clinical conditions.

👩‍💻 Author & Institutional Credits
Developed By:
Stuti Katiyar

Undergraduate Student, B.E. Artificial Intelligence & Machine Learning

CMR Institute of Technology, Bengaluru, India

GitHub: @stutikatiyar

Role: Lead System Architect & Algorithm Developer
