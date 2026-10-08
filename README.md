# DWT-SVD Fusion Lab

## Overview

DWT-SVD Fusion Lab is a Streamlit application for classical grayscale image fusion. It combines a multilevel Discrete Wavelet Transform (DWT) with Singular Value Decomposition (SVD), then reconstructs and evaluates the result. The project uses traditional Digital Image Processing only; it does not use deep-learning or machine-learning models.

## Features

- Seven workspaces: Dashboard, Image Fusion, Analysis, DWT Decomposition, SVD Analysis, Quality Metrics, and About Project.
- Upload two PNG, JPG/JPEG, or BMP images and inspect original dimensions, channels, file size, and previews.
- Normalize, convert to grayscale, and resize the image pair to common processing dimensions.
- Visualize LL, LH, HL, and HH coefficient bands for each input.
- Inspect compact U and Vᵀ factor heatmaps, singular values, retained ranks, and energy weights without dumping raw matrices.
- View histograms and summary statistics for the sources and fused image.
- Calculate MSE, PSNR, SSIM, and entropy from the actual processed images.
- Download the fused result as an 8-bit grayscale PNG.
- In-memory processing with upload size and decoded-pixel safety limits.

## Technologies

- Python 3.10+
- Streamlit
- OpenCV (headless)
- NumPy
- PyWavelets
- Matplotlib
- scikit-image
- Pillow

## Algorithm

1. OpenCV decodes the two uploaded images and validates their channel count, byte size, and decoded dimensions.
2. Images are converted to grayscale and normalized. They are resized to the selected common dimensions and capped at 1024 pixels on the longest processing side.
3. PyWavelets `wavedec2` computes the multilevel 2D wavelet coefficient pyramid for both inputs.
4. For each LL matrix, NumPy computes compact SVD. The default retained rank is the ceiling of 10% of the singular values, with at least one retained value. The two low-rank reconstructions are blended according to their retained squared-singular-value energies.
5. The default detail rule selects, at each coefficient position, the LH/HL/HH coefficient with larger absolute magnitude. The optional all-bands mode applies the SVD energy fusion rule to detail matrices too.
6. PyWavelets `waverec2` reconstructs the output. It is clipped and converted to an 8-bit grayscale image.
7. MSE, PSNR, and SSIM are calculated separately against each aligned input. Entropy is reported for the fused result.

For 8-bit inputs, $\mathrm{MSE}=\frac{1}{N}\sum(A-F)^2$ and $\mathrm{PSNR}=10\log_{10}(255^2/\mathrm{MSE})$. Identical images produce infinite PSNR. SSIM is reported as unavailable for images smaller than 3×3. There is no ground-truth fused reference, so source-relative metrics must not be described as absolute fusion-quality scores.

## Project Architecture

```text
app.py                         Streamlit navigation and pages
fusion/
  preprocessing.py             Decode, validate, grayscale, resize
  dwt_fusion.py                DWT and coefficient fusion
  svd_fusion.py                Truncated SVD and energy weighting
  reconstruction.py            Inverse DWT and uint8 output
  metrics.py                   MSE, PSNR, SSIM, entropy
utils/
  visualization.py             DWT, SVD, histogram, and comparison plots
tests/
  test_scenarios.py            Input and Streamlit workflow tests
.streamlit/
  config.toml                  Streamlit theme and server limits
requirements.txt               Deployment dependencies
```

## Installation

Create and activate a virtual environment, then install dependencies.

```powershell
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\activate
```

Install packages:

```powershell
pip install -r requirements.txt
```

## Running Locally

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. In the sidebar, choose wavelet, decomposition level, fusion rule, and dimension matching. Open Image Fusion, upload both files, review preprocessing, and press **Fuse Images**. Results and download are shown below the controls; the other navigation pages provide DWT, SVD, histogram, and metric views.

Run regression tests with:

```powershell
python -m unittest discover -s tests -v
```

## Deployment

The repository is prepared for Streamlit Community Cloud. It uses a root-level `app.py`, a root-level `requirements.txt`, package-relative imports, and no local databases, absolute machine paths, API keys, or temporary upload files. Uploaded data stays in memory. The app enforces 20 MB per upload, a 20-megapixel decoded-image limit, and 1024 pixels as the maximum processed side to keep resource use bounded.

1. Push this project to a GitHub repository.
2. Sign in to Streamlit Community Cloud with GitHub and grant access to the repository.
3. Create a new app and select the repository, branch, and `app.py` as the main file.
4. Deploy. The platform installs `requirements.txt` and reads `.streamlit/config.toml` automatically.
5. Confirm the app opens, upload two test images, run fusion, and verify the PNG download.

No `runtime.txt` is included because Streamlit Community Cloud can use its supported default Python runtime; the code and dependencies target Python 3.10 or newer.

## Screenshots

Add screenshots from your own running deployment to a repository folder such as `docs/screenshots/`, then embed them here. Suggested captures:

- Dashboard hero and feature cards
- Image upload, previews, and preprocessing metadata
- DWT coefficient page
- SVD spectrum and factor visualizations
- Fused output, comparison, metrics, and download control

No sample result screenshot is included because output depends on the images selected by the user.

## Results

Numerical output depends on the selected image pair and settings. Record values from the deployed or local app; do not reuse values from another run.

| Measurement | Value from your run |
|---|---|
| Image A / B filenames and original dimensions | [Fill from your run] |
| Common processed dimensions | [Fill from your run] |
| Wavelet, actual level, and fusion rule | [Fill from your run] |
| MSE / PSNR / SSIM vs Image A | [Fill from app] |
| MSE / PSNR / SSIM vs Image B | [Fill from app] |
| Fused-image entropy | [Fill from app] |
| Downloaded output dimensions | [Verify and fill] |

Metrics compare the fused image separately with each aligned source image. They are not ground-truth scores. Entropy measures intensity-distribution complexity, not perceptual quality by itself.

## Future Scope

Possible extensions include color-preserving fusion, image registration, configurable SVD rank, additional classical detail-fusion rules, run-time profiling, and evaluation using registered reference datasets. These are future ideas, not current application features.

## Authors

- [Student name, roll number]
- [Project guide]
- [Department and institution]
