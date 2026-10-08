# Project Report: Image Fusion Using DWT and SVD

## Title Page

**Project title:** Image Fusion Using DWT and SVD  
**Subtitle:** Digital Image Processing Based Image Fusion System  
**Submitted by:** [Student name, roll number, and class]  
**Department:** [Department name]  
**Institution:** [College name]  
**Project guide:** [Guide name]  
**Academic year:** [Academic year]

## Abstract

Image fusion combines information from two images into one output image. This project implements a classical transform-domain fusion method using the Discrete Wavelet Transform (DWT) and Singular Value Decomposition (SVD). Two uploaded images are decoded, converted to grayscale, normalized, and resized to the same analysis dimensions. PyWavelets decomposes each image into approximation and directional detail coefficients. The default method fuses the LL approximation bands by blending truncated SVD reconstructions according to retained singular-value energy. It combines LH, HL, and HH detail coefficients by selecting the coefficient with the larger absolute magnitude. An optional mode applies the SVD energy rule to all bands. Inverse DWT reconstructs an 8-bit grayscale output. The application reports MSE, PSNR, SSIM, and entropy and provides coefficient visualizations and PNG download through Streamlit. The project uses traditional digital image processing only; it does not use a learned model. Quantitative metric values depend on the input pair and should be recorded from an actual application run.

## 1. Introduction

Images can contain complementary information. One image may show a structure or edge clearly while another may contain different intensity or texture information. Image fusion attempts to combine useful information into one representation.

A direct pixelwise combination treats all image content similarly. A transform-domain approach first separates content according to spatial scale and direction. DWT provides approximation and detail coefficient bands. SVD represents a coefficient matrix using ordered singular values and corresponding spatial patterns. This project uses these established mathematical operations to create an interpretable image-fusion pipeline.

The application is implemented in Python. OpenCV handles image input and preprocessing; NumPy provides array operations; PyWavelets performs the wavelet transform; scikit-image calculates SSIM; Matplotlib presents coefficient and comparison visualizations; and Streamlit provides the interactive interface.

## 2. Problem Statement

Given two image files that may be grayscale or RGB and may have different dimensions, develop a system that preprocesses the images, extracts transform coefficients, combines their information using defined DWT/SVD rules, reconstructs one fused image, evaluates it with standard metrics, and allows the result to be downloaded.

The fusion system must handle invalid or empty uploads, small images, resizing, output dimensions, and metric edge cases. Since no ground-truth fused image is supplied, quality metrics are calculated separately against each aligned source image and must be interpreted as comparative measurements.

## 3. Objectives

1. Accept two supported image uploads through a graphical interface.
2. Decode grayscale or color image data and convert it to grayscale.
3. Normalize and resize inputs to the same analysis shape.
4. Calculate multilevel 2D DWT coefficients for both images.
5. Apply truncated SVD and energy-weighted fusion to approximation coefficients.
6. Fuse detail coefficients using maximum absolute magnitude by default, with an optional all-band SVD mode.
7. Reconstruct an 8-bit grayscale image using inverse DWT.
8. Display coefficient information, the fused image, a comparison, and image metrics.
9. Provide the reconstructed image as a downloadable PNG.

## 4. Scope

The system is a demonstrator of classical grayscale image fusion. Inputs are converted to grayscale before fusion; RGB color information is not preserved. The user can select from `db2`, `haar`, `sym2`, and `coif1`, request decomposition levels 1 through 3, choose the default hybrid fusion method or the all-bands SVD method, and choose a common-size policy. The output is a grayscale PNG.

The project does not perform image registration. The resize policy aligns array dimensions, not necessarily the scenes or objects inside them. It is not a clinically validated medical system or an industrial decision system.

## 5. Existing and Proposed Systems

### 5.1 Existing/basic approach

A basic fusion method may average corresponding input pixels or use a fixed weighted sum. This is straightforward but does not separately model coarse structure and directional details. When source pixels differ, averaging can reduce local contrast or soften edges. A learned fusion system is not considered here because the project requirement is to remain within traditional Digital Image Processing.

### 5.2 Proposed approach

The proposed system first converts both inputs to grayscale and aligns their dimensions. It applies a multilevel 2D DWT to obtain the approximation and detail coefficients. For LL, each input matrix is represented by a truncated SVD reconstruction, and the reconstructions are blended according to retained singular-value energy. For detail bands, the default method selects the coefficient with larger absolute magnitude. An optional mode uses energy-weighted truncated SVD for every coefficient band. The fused coefficient pyramid is reconstructed with inverse DWT and evaluated against both preprocessed inputs.

## 6. Technologies and Tools

| Technology | Role in the project |
|---|---|
| Python | Application and processing language |
| OpenCV | Decode uploads, convert color to grayscale, resize image arrays |
| NumPy | Numerical arrays, normalization support, SVD and coefficient operations |
| PyWavelets | Multilevel 2D DWT and inverse DWT |
| scikit-image | Structural Similarity Index (SSIM) calculation |
| Matplotlib | DWT coefficient, singular spectrum, and comparison plots |
| Streamlit | Upload controls, settings, results, metrics, and download interface |
| Pillow | Encode the final grayscale array as PNG for download |

No CNN, GAN, YOLO, transformer, or machine-learning model is used.

## 7. System Requirements

### 7.1 Software

- Python 3.10 or newer (the project was tested in Python 3.11.9 in the current workspace).
- Dependencies listed in `requirements.txt`.
- A modern web browser to use the Streamlit application.

### 7.2 Hardware

The project runs on a standard computer with sufficient memory for the selected image dimensions. No GPU is required. Runtime and memory requirements increase with image size and with the all-bands SVD option. Record the computer specifications if timing results are included in the report.

## 8. System Architecture

The main data flow is:

```text
Image 1 ─┐
         ├─> Decode and validate ─> Grayscale / normalize / resize ─┐
Image 2 ─┘                                                          │
                                                                    v
                                                2D DWT for each image
                                                                    │
                                  ┌─────────────────────────────────┴────────────┐
                                  v                                              v
                          LL approximation                              LH, HL, HH details
                        Truncated SVD per input                    Max-absolute by default
                        Retained energy weights                    or all-band SVD mode
                                  └─────────────────────────────────┬────────────┘
                                                                    v
                                                     Fused coefficient pyramid
                                                                    │
                                                                    v
                                                          Inverse 2D DWT
                                                                    │
                                                                    v
                                      8-bit image ─> metrics / comparison / PNG download
```

### 8.1 Main software modules

- `app.py`: Defines the Streamlit page, sidebar settings, upload workflow, result views, metrics, and download control.
- `fusion/preprocessing.py`: Decodes image bytes and prepares grayscale, normalized, equal-sized inputs.
- `fusion/dwt_fusion.py`: Performs DWT decomposition, checks coefficient structures, fuses coefficient bands, and summarizes dimensions.
- `fusion/svd_fusion.py`: Performs SVD truncation, reconstruction, retained-energy calculation, and energy-weighted blending.
- `fusion/reconstruction.py`: Applies inverse DWT, restores requested dimensions, clips values, and returns `uint8` output.
- `fusion/metrics.py`: Calculates MSE, PSNR, SSIM, and entropy with small-image and zero-error handling.
- `utils/visualization.py`: Creates DWT band, singular-spectrum, and image-comparison figures.

## 9. Methodology

### 9.1 Input and preprocessing

The interface accepts PNG, JPEG, BMP, TIFF, and WebP uploads. OpenCV decodes the uploaded bytes. Grayscale data is kept as grayscale; BGR/BGRA color data is converted to grayscale. The pair-preparation function converts inputs to floating-point arrays in $[0,1]$ and resizes them to either Image 1's dimensions or the selected smaller dimensions. Downscaling uses area interpolation; upscaling uses cubic interpolation. The current policy resizes width and height independently and may therefore alter aspect ratio.

### 9.2 DWT decomposition

For each image, PyWavelets' `wavedec2` computes the multilevel two-dimensional wavelet coefficients. In the standard PyWavelets representation, the first element is the coarsest approximation array and each following tuple contains three detail arrays for a level. The app visualizes the coarsest LL approximation and the three detail arrays at the final tuple. The actual decomposition level is limited by the image size and selected wavelet. For very small images where no level is initially supported, the implementation symmetrically pads the array before decomposition; inverse reconstruction is returned at the requested original dimensions.

### 9.3 SVD-based approximation fusion

For each LL coefficient matrix, NumPy computes the compact SVD:

$$
A = U_A\Sigma_A V_A^T, \qquad B = U_B\Sigma_B V_B^T.
$$

Let $r$ be the retained rank. By default, $r$ is the ceiling of 10% of the number of singular values, with a minimum of one. The low-rank reconstructions are:

$$
\hat{A}=U_{A,r}\Sigma_{A,r}V_{A,r}^T, \qquad
\hat{B}=U_{B,r}\Sigma_{B,r}V_{B,r}^T.
$$

Their retained energies are calculated as:

$$
E_A=\sum_{i=1}^{r}\sigma_{A,i}^2, \qquad
E_B=\sum_{i=1}^{r}\sigma_{B,i}^2.
$$

The fused approximation uses normalized energy weights:

$$
F_{LL}=\frac{E_A}{E_A+E_B}\hat{A}+\frac{E_B}{E_A+E_B}\hat{B}.
$$

If both retained energies are effectively zero, the SVD fusion helper uses an arithmetic mean fallback to avoid division by zero.

### 9.4 Detail fusion

In the default `SVD energy + max-absolute details` mode, each pair of corresponding detail coefficients $d_A$ and $d_B$ is fused as:

$$
F_d = \begin{cases}
d_A, & |d_A| \ge |d_B|,\\
d_B, & |d_B| > |d_A|.
\end{cases}
$$

Thus a tie selects the coefficient from Image 1. The alternative `SVD energy all bands` mode uses truncated SVD energy fusion for detail matrices as well as for LL.

## 10. Reconstruction and Output

PyWavelets' `waverec2` reconstructs the spatial image from the fused coefficient pyramid. The reconstruction is cropped to the requested height and width; if it is shorter than requested, edge padding restores the dimensions. Non-finite reconstruction values are sanitized, values are clipped to $[0,1]$, multiplied by 255, rounded, and converted to `uint8`. The Streamlit app displays this grayscale image and encodes it in PNG format for download.

## 11. Quality Metrics

Metrics are calculated once against preprocessed Image 1 and once against preprocessed Image 2. This is not a comparison against a ground-truth fused image.

### 11.1 Mean Squared Error

For 8-bit intensity values $A_i$ and $F_i$, MSE is:

$$
\mathrm{MSE}(A,F)=\frac{1}{N}\sum_{i=1}^{N}(A_i-F_i)^2.
$$

Lower MSE indicates lower average squared intensity difference. Identical images have MSE equal to zero.

### 11.2 Peak Signal-to-Noise Ratio

For an 8-bit maximum intensity of 255:

$$
\mathrm{PSNR}=10\log_{10}\left(\frac{255^2}{\mathrm{MSE}}\right).
$$

Higher PSNR indicates lower pixelwise error. The implementation reports positive infinity when MSE is zero.

### 11.3 Structural Similarity Index

SSIM compares local luminance, contrast, and structure, using `data_range=255` and a valid odd window size no larger than 7. It is computed with scikit-image. For any image with a minimum dimension below 3, SSIM is returned as unavailable (`None`) rather than raising a window-size error.

### 11.4 Entropy

Entropy is calculated from the fused image's 256-bin grayscale intensity histogram:

$$
H(F)=-\sum_{k:p_k>0} p_k\log_2(p_k),
$$

where $p_k$ is the fraction of pixels with intensity $k$. Entropy describes intensity variation and should not be interpreted by itself as a perceptual quality score.

## 12. User Interface and Operation

The Streamlit page contains upload controls, project settings for wavelet, level, fusion mode, and output dimensions, followed by preprocessing views, DWT band visualizations, the LL singular-value spectrum, fused output, metrics, three-image comparison, and PNG download. A result is shown only when both uploads are present and the stored result corresponds to the current input and settings.

### 12.1 Run instructions

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

### 12.2 Test instructions

```powershell
python -m unittest discover -s tests -v
```

The current project test suite contains 12 tests and passed in the prepared workspace environment. It covers same-size grayscale inputs, mismatched dimensions, RGB pairs, mixed grayscale/RGB, 2×2 small inputs, 1024×1024 inputs, invalid/empty image bytes, RGB decoding, the no-upload interface state, the uploaded-image fusion/download-widget workflow, the optional all-bands fusion mode, and friendly UI handling of invalid uploads. These tests verify program behavior; they do not establish that any particular metric value is optimal.

## 13. Results and Discussion

### 13.1 Results to obtain from the demonstration run

Run the application with the pair selected for the report and fill in this table from the visible output. Do not substitute values from a different pair.

| Item | Run-specific value |
|---|---|
| Image 1 filename and original dimensions | [Record from selected file] |
| Image 2 filename and original dimensions | [Record from selected file] |
| Common analysis dimensions | [Record from app] |
| Wavelet and actual decomposition level | [Record from app] |
| Fusion method | [Record from sidebar] |
| Fused image dimensions | [Record from app/download] |
| MSE against Image 1 | [Record from app] |
| PSNR against Image 1 (dB) | [Record from app] |
| SSIM against Image 1 | [Record from app or N/A for dimensions under 3] |
| MSE against Image 2 | [Record from app] |
| PSNR against Image 2 (dB) | [Record from app] |
| SSIM against Image 2 | [Record from app or N/A for dimensions under 3] |
| Fused-image entropy (bits) | [Record from app] |

### 13.2 Discussion guidance

Discuss which structures from each input appear in the output and how the LL/detail fusion rules contributed. Explain that metric values differ depending on which input is used as the reference. A fused image need not have the lowest MSE or highest SSIM against both sources because its purpose is to combine information, not reproduce either source exactly. Entropy indicates intensity diversity, not necessarily better visual quality. Include actual screenshots of the input pair, coefficient panels, singular-value spectrum, fused result, and metrics.

## 14. Testing and Error Handling

The preprocessing code rejects empty byte streams, undecodable files, unsupported channel layouts, invalid image arrays, and unsupported size policies with clear errors. The fusion code validates coefficient counts and shapes. The reconstruction produces the selected target dimensions and `uint8` output. Metric calculations validate grayscale shape and matching dimensions, handle zero MSE by returning infinite PSNR, sanitize non-finite values during metric conversion, and return unavailable SSIM for images too small for a valid window.

Functional test execution in the prepared project environment:

```text
12 tests run: 12 passed, 0 failed
```

This pass count is a real software test result. Per-image numerical metrics and fusion-quality conclusions must still be captured from the images used in the student's demonstration.

## 15. Advantages

- Uses mathematically interpretable, non-learned operations.
- Separates coarse approximation and directional detail information.
- SVD gives a low-rank approximation and energy-based blending weights.
- Supports multiple wavelets and decomposition levels.
- Handles grayscale, RGB, different-size, tiny, and large test inputs.
- Provides visual coefficient inspection, metrics, and a downloadable output.

## 16. Limitations

- Grayscale conversion discards color information.
- The independent width/height resize policy can alter aspect ratio and scene geometry.
- Dimension matching does not register or align objects between images.
- Maximum-absolute detail selection may preserve noise or isolated coefficients.
- Full SVD can become costly for large matrices, especially when applied to every band.
- Metric values are source-relative because no reference fused image is available.
- Entropy alone is not a measure of perceptual fusion quality.

## 17. Applications

Image fusion is a general technique for combining complementary image information. Common research examples include multi-sensor imaging, visible and infrared imagery, multi-focus photography, multi-exposure imaging, and combining scientific or medical image views. The current project demonstrates the method on uploaded grayscale inputs; it does not claim domain-specific validation or suitability for clinical decisions.

## 18. Future Scope

Potential extensions include color-preserving fusion, image registration before coefficient fusion, edge-aware or region-based detail rules, configurable SVD rank, runtime and memory profiling, and a systematic comparison with other traditional fusion methods. Evaluation could also include registered reference datasets and task-specific expert measures. These are proposed future improvements and are not part of the current implementation.

## 19. Conclusion

The project implements a traditional DWT-SVD fusion workflow with a working Streamlit interface. DWT separates the images into approximation and detail coefficients. Truncated SVD and retained singular-value energy guide the default LL fusion; detail coefficients are selected by maximum absolute magnitude. Inverse DWT reconstructs an 8-bit grayscale output, and the app reports MSE, PSNR, SSIM, and entropy against the source images. The implementation is suitable as an educational demonstration of transform-domain image processing, with metric values to be collected from the student's own selected images.

## References

- Rafael C. Gonzalez and Richard E. Woods, *Digital Image Processing*, Pearson.
- Stéphane Mallat, *A Wavelet Tour of Signal Processing*, Academic Press.
- Gene H. Golub and Charles F. Van Loan, *Matrix Computations*, Johns Hopkins University Press.
- PyWavelets documentation for `wavedec2` and `waverec2`.
- scikit-image documentation for `structural_similarity`.
