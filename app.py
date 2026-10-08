"""DWT-SVD Fusion Lab: an interactive classical digital image processing app."""

from __future__ import annotations

import hashlib
import html
import logging
from io import BytesIO

import cv2
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from PIL import Image

from fusion import calculate_metrics, decompose_image, fuse_coefficients, prepare_image_pair, reconstruct_image
from fusion.dwt_fusion import coefficient_summary
from fusion.preprocessing import MAX_UPLOAD_BYTES, decode_uploaded_image_details
from utils.visualization import (
    comparison_figure,
    decomposition_figure,
    intensity_histogram_figure,
    singular_spectrum_figure,
    svd_factor_figure,
)

LOGGER = logging.getLogger(__name__)
APP_NAME = "DWT-SVD Fusion Lab"
MAX_PROCESSING_SIDE = 1024
IMAGE_TYPES = ["png", "jpg", "jpeg", "bmp"]
PAGES = [
    "🏠 Dashboard",
    "🖼 Image Fusion",
    "📊 Analysis",
    "📐 DWT Decomposition",
    "🔬 SVD Analysis",
    "📈 Quality Metrics",
    "ℹ About Project",
]


@st.cache_data(show_spinner=False, max_entries=4)
def _decode_cached(raw_bytes: bytes) -> dict[str, object]:
    return decode_uploaded_image_details(raw_bytes)


@st.cache_data(show_spinner=False, max_entries=4)
def _prepare_pair_cached(
    grayscale_a: np.ndarray,
    grayscale_b: np.ndarray,
    size_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    return prepare_image_pair(
        grayscale_a,
        grayscale_b,
        size_mode=size_mode,
        max_side=MAX_PROCESSING_SIDE,
    )


@st.cache_data(show_spinner=False, max_entries=3)
def _fusion_cached(
    image_a: np.ndarray,
    image_b: np.ndarray,
    wavelet: str,
    level: int,
    method: str,
) -> dict[str, object]:
    coefficients_a = decompose_image(image_a, wavelet=wavelet, level=level)
    coefficients_b = decompose_image(image_b, wavelet=wavelet, level=level)
    svd_analysis: dict = {}
    fused_coefficients = fuse_coefficients(
        coefficients_a,
        coefficients_b,
        method=method,
        svd_analysis=svd_analysis,
    )
    fused = reconstruct_image(fused_coefficients, wavelet=wavelet, output_shape=image_a.shape)
    source_a = np.rint(image_a * 255).astype(np.uint8)
    source_b = np.rint(image_b * 255).astype(np.uint8)
    return {
        "image_a": image_a,
        "image_b": image_b,
        "coefficients_a": coefficients_a,
        "coefficients_b": coefficients_b,
        "coefficients_fused": fused_coefficients,
        "svd_analysis": svd_analysis,
        "fused": fused,
        "metrics_a": calculate_metrics(source_a, fused),
        "metrics_b": calculate_metrics(source_b, fused),
        "wavelet": wavelet,
        "level": len(coefficients_a) - 1,
        "method": method,
    }


def _signature(raw_a: bytes, raw_b: bytes, wavelet: str, level: int, method: str, size_mode: str) -> str:
    digest = hashlib.sha256()
    digest.update(len(raw_a).to_bytes(8, "big"))
    digest.update(raw_a)
    digest.update(len(raw_b).to_bytes(8, "big"))
    digest.update(raw_b)
    digest.update(f"{wavelet}|{level}|{method}|{size_mode}|{MAX_PROCESSING_SIDE}".encode())
    return digest.hexdigest()


def _format_bytes(size: int) -> str:
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _entropy_u8(image: np.ndarray) -> float:
    histogram = np.bincount(np.asarray(image, dtype=np.uint8).ravel(), minlength=256).astype(np.float64)
    probabilities = histogram[histogram > 0] / histogram.sum()
    return float(-np.sum(probabilities * np.log2(probabilities)))


def _clear_uploads() -> None:
    for key in ("image_a_bytes", "image_a_name", "image_b_bytes", "image_b_name", "fusion_result"):
        st.session_state.pop(key, None)
    st.session_state["upload_nonce"] = st.session_state.get("upload_nonce", 0) + 1


def _go_to_fusion() -> None:
    st.session_state["navigation"] = "🖼 Image Fusion"


def _render_section_intro(kicker: str, title: str, description: str) -> None:
    st.markdown(f'<div class="section-kicker">{kicker}</div>', unsafe_allow_html=True)
    st.title(title)
    st.markdown(f'<p class="section-copy">{description}</p>', unsafe_allow_html=True)


def _render_feature_card(icon: str, title: str, detail: str) -> None:
    st.markdown(
        f'<div class="feature-card"><div class="feature-icon">{icon}</div>'
        f'<div class="feature-title">{title}</div><div class="feature-copy">{detail}</div></div>',
        unsafe_allow_html=True,
    )


def _current_result(signature: str | None) -> dict | None:
    result = st.session_state.get("fusion_result")
    if result is not None and signature is not None and result.get("signature") == signature:
        return result
    return None


def _render_comparison(result: dict) -> None:
    figure = comparison_figure(
        np.rint(result["image_a"] * 255).astype(np.uint8),
        np.rint(result["image_b"] * 255).astype(np.uint8),
        result["fused"],
    )
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def _render_metric_cards(result: dict) -> None:
    metrics = result["metrics_a"]
    columns = st.columns(4, gap="medium")
    columns[0].metric("MSE · vs Image A", f"{metrics['MSE']:,.2f}")
    psnr = "∞" if np.isinf(metrics["PSNR (dB)"]) else f"{metrics['PSNR (dB)']:.2f} dB"
    columns[1].metric("PSNR · vs Image A", psnr)
    ssim = "N/A" if metrics["SSIM"] is None else f"{metrics['SSIM']:.4f}"
    columns[2].metric("SSIM · vs Image A", ssim)
    columns[3].metric("Fused entropy", f"{metrics['Entropy (bits)']:.3f} bits")


def _render_download(result: dict) -> None:
    stream = BytesIO()
    Image.fromarray(result["fused"]).save(stream, format="PNG")
    st.download_button(
        "⬇  Download Fused Image",
        data=stream.getvalue(),
        file_name="dwt_svd_fused.png",
        mime="image/png",
        type="primary",
        width="stretch",
        key="download_fused_image",
    )


st.set_page_config(page_title=APP_NAME, page_icon="◈", layout="wide", initial_sidebar_state="expanded")
st.markdown(
    """
    <style>
    :root { color-scheme:dark; --canvas:#07110f; --panel:rgba(18,34,31,.82); --stroke:rgba(173,211,196,.13); --text:#e8f3ef; --muted:#9bb0a8; --mint:#59ddb0; }
    [data-testid="stAppViewContainer"] { background:radial-gradient(ellipse at 78% -10%,#15352c 0,transparent 40%),var(--canvas); color:var(--text); }
    [data-testid="stHeader"] { background:transparent; }
    [data-testid="stSidebar"] { background:#0b1714; border-right:1px solid var(--stroke); }
    .block-container { max-width:1440px; padding:2.1rem clamp(1rem,3vw,3rem) 3.2rem; }
    h1,h2,h3,p,label { color:var(--text); } h1 { font-size:clamp(1.8rem,3vw,2.55rem); font-weight:700; letter-spacing:0; }
    h2 { font-size:1.35rem; font-weight:650; letter-spacing:0; } h3 { font-size:1.05rem; font-weight:620; letter-spacing:0; }
    [data-testid="stCaptionContainer"] p,.section-copy { color:var(--muted); }
    .brand-lockup { display:flex; align-items:center; gap:.75rem; padding:.55rem .3rem 1.4rem; }
    .brand-mark { display:grid; place-items:center; width:2.5rem; height:2.5rem; border-radius:9px; color:#07110f; background:linear-gradient(140deg,var(--mint),#8dc8ed); font-weight:800; }
    .brand-name { color:var(--text); font-size:1.02rem; font-weight:700; } .brand-sub { color:var(--muted); font-size:.72rem; margin-top:.13rem; }
    .hero { position:relative; overflow:hidden; padding:clamp(2rem,5vw,4.4rem); border:1px solid rgba(147,219,190,.22); border-radius:18px; background:linear-gradient(115deg,rgba(27,74,59,.94),rgba(12,35,36,.92) 58%,rgba(18,43,58,.94)); box-shadow:0 24px 64px rgba(0,0,0,.25); }
    .hero:after { content:""; position:absolute; inset:auto -8% -90% 48%; height:125%; border:1px solid rgba(112,226,185,.18); border-radius:50%; box-shadow:0 0 0 36px rgba(112,226,185,.035),0 0 0 72px rgba(112,226,185,.025); pointer-events:none; }
    .hero-kicker,.section-kicker { color:var(--mint); text-transform:uppercase; font-size:.72rem; font-weight:750; letter-spacing:.12em; }
    .hero h1 { max-width:780px; margin:.8rem 0 .7rem; font-size:clamp(2.2rem,5vw,4rem); line-height:1.08; }
    .hero-copy { max-width:650px; color:#c3d7ce; font-size:1.08rem; line-height:1.65; }
    .feature-card,.upload-card,.analysis-card { height:100%; padding:1.15rem; border:1px solid var(--stroke); border-radius:12px; background:linear-gradient(145deg,rgba(23,43,39,.9),rgba(14,28,26,.83)); }
    .feature-icon { color:var(--mint); font-size:1.3rem; margin-bottom:.55rem; } .feature-title { color:var(--text); font-weight:700; margin-bottom:.3rem; }
    .feature-copy,.image-meta { color:var(--muted); font-size:.88rem; line-height:1.5; } .image-meta { overflow-wrap:anywhere; }
    .upload-label { color:var(--mint); text-transform:uppercase; font-size:.72rem; font-weight:750; letter-spacing:.11em; }
    .upload-title { font-weight:700; font-size:1.08rem; margin:.3rem 0 .8rem; }
    .step-row { display:grid; grid-template-columns:repeat(5,1fr); gap:.65rem; }
    .step { padding:.9rem; border-top:2px solid rgba(89,221,176,.65); background:rgba(17,36,32,.68); border-radius:0 0 8px 8px; }
    .step-num { color:var(--mint); font-size:.72rem; font-weight:750; } .step-name { margin-top:.35rem; font-weight:650; }
    [data-testid="stMetric"] { background:var(--panel); border:1px solid var(--stroke); padding:1rem 1.05rem; border-radius:11px; min-height:108px; }
    [data-testid="stMetricLabel"] { color:var(--muted); } [data-testid="stMetricValue"] { color:var(--text); font-weight:700; }
    div[data-testid="stButton"]>button,div[data-testid="stDownloadButton"]>button { border-radius:9px; min-height:2.9rem; font-weight:700; transition:filter .15s ease,transform .15s ease; }
    div[data-testid="stButton"]>button[kind="primary"],div[data-testid="stDownloadButton"]>button[kind="primary"] { background:linear-gradient(105deg,#50d5a4,#79c6da); border:0; color:#07110f; }
    div[data-testid="stButton"]>button:hover,div[data-testid="stDownloadButton"]>button:hover { filter:brightness(1.08); transform:translateY(-1px); }
    [data-testid="stFileUploader"] section { border:1px dashed rgba(124,202,170,.34); border-radius:10px; background:rgba(8,21,18,.42); }
    [data-testid="stDataFrame"] { border:1px solid var(--stroke); border-radius:10px; overflow:hidden; } hr { border-color:var(--stroke); }
    @media(max-width:800px) { .step-row { grid-template-columns:repeat(2,1fr); } .hero { padding:1.7rem; } }
    @media(max-width:520px) { .step-row { grid-template-columns:1fr; } .block-container { padding:1.2rem .8rem 2rem; } }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand-lockup"><div class="brand-mark">D/S</div><div><div class="brand-name">DWT-SVD Fusion Lab</div><div class="brand-sub">CLASSICAL DIP WORKBENCH</div></div></div>',
        unsafe_allow_html=True,
    )
    page = st.radio("Navigation", PAGES, key="navigation", label_visibility="collapsed")
    st.divider()
    st.markdown("**Processing setup**")
    wavelet = st.selectbox("Wavelet", ["db2", "haar", "sym2", "coif1"], index=0)
    requested_level = st.select_slider("DWT level", options=[1, 2, 3], value=2)
    fusion_method = st.selectbox("Fusion rule", ["SVD energy + max-absolute details", "SVD energy all bands"])
    size_mode = st.selectbox(
        "Dimension matching",
        ["Image A dimensions", "Smaller dimensions"],
        help="Images are resized to a common shape and limited to 1024 px on the longest processing side.",
    )
    st.caption("No deep-learning or machine-learning models are used.")

st.markdown(f'<div class="section-kicker">{APP_NAME} · DIGITAL IMAGE PROCESSING</div>', unsafe_allow_html=True)
raw_a = st.session_state.get("image_a_bytes", b"")
raw_b = st.session_state.get("image_b_bytes", b"")
signature = _signature(raw_a, raw_b, wavelet, requested_level, fusion_method, size_mode) if raw_a and raw_b else None
result = _current_result(signature)

if page == "🏠 Dashboard":
    st.markdown(
        '<section class="hero"><div class="hero-kicker">Transform-domain research interface</div><h1>DWT-SVD Image Fusion</h1><div class="hero-copy">Combine complementary information from two images using classical Digital Image Processing techniques.</div></section>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.button("Start Fusion  →", type="primary", width="content", key="start_fusion", on_click=_go_to_fusion)
    st.write("")
    cards = st.columns(4, gap="medium")
    for column, args in zip(cards, (("◫", "DWT", "Multi-resolution image decomposition"), ("Σ", "SVD", "Matrix-based structural factorization"), ("⟡", "Fusion", "Combine complementary coefficients"), ("▥", "Metrics", "PSNR · SSIM · MSE · Entropy"))):
        with column:
            _render_feature_card(*args)
    st.markdown("### How it works")
    st.markdown('<div class="step-row"><div class="step"><div class="step-num">01</div><div class="step-name">Upload</div></div><div class="step"><div class="step-num">02</div><div class="step-name">Decompose</div></div><div class="step"><div class="step-num">03</div><div class="step-name">Fuse</div></div><div class="step"><div class="step-num">04</div><div class="step-name">Reconstruct</div></div><div class="step"><div class="step-num">05</div><div class="step-name">Evaluate</div></div></div>', unsafe_allow_html=True)
    if result:
        st.markdown("### Latest result")
        st.image(result["fused"], caption="Latest reconstructed grayscale output", width="stretch")
        _render_metric_cards(result)

elif page == "🖼 Image Fusion":
    _render_section_intro("WORKSPACE 01", "Image Fusion", "Upload a pair, review preprocessing, and run the DWT-SVD coefficient fusion pipeline.")
    st.markdown('<div class="section-kicker">IMAGE INPUTS</div>', unsafe_allow_html=True)
    upload_nonce = st.session_state.setdefault("upload_nonce", 0)
    columns = st.columns(2, gap="large")
    with columns[0]:
        st.markdown('<div class="upload-label">IMAGE A</div><div class="upload-title">Upload First Image</div>', unsafe_allow_html=True)
        upload_a = st.file_uploader("Choose image A", type=IMAGE_TYPES, key=f"upload_a_widget_{upload_nonce}", label_visibility="collapsed")
        st.caption("PNG · JPG · JPEG · BMP · up to 20 MB")
    with columns[1]:
        st.markdown('<div class="upload-label">IMAGE B</div><div class="upload-title">Upload Second Image</div>', unsafe_allow_html=True)
        upload_b = st.file_uploader("Choose image B", type=IMAGE_TYPES, key=f"upload_b_widget_{upload_nonce}", label_visibility="collapsed")
        st.caption("PNG · JPG · JPEG · BMP · up to 20 MB")

    if upload_a is not None:
        st.session_state["image_a_bytes"], st.session_state["image_a_name"] = upload_a.getvalue(), upload_a.name
    if upload_b is not None:
        st.session_state["image_b_bytes"], st.session_state["image_b_name"] = upload_b.getvalue(), upload_b.name
    stored_a, stored_b = st.session_state.get("image_a_bytes", b""), st.session_state.get("image_b_bytes", b"")
    st.button("Clear Images", key="clear_images", on_click=_clear_uploads)
    if not stored_a or not stored_b:
        needed = "Image A" if not stored_a else "Image B"
        st.info(f"Upload both images to continue. {needed} is still needed.")
    else:
        try:
            info_a, info_b = _decode_cached(stored_a), _decode_cached(stored_b)
            image_a, image_b = _prepare_pair_cached(
                info_a["grayscale"], info_b["grayscale"],
                "Image 1 dimensions" if size_mode == "Image A dimensions" else size_mode,
            )
        except (ValueError, cv2.error, MemoryError) as error:
            st.error(f"Image validation failed: {error}")
            info_a = info_b = image_a = image_b = None
        if image_a is not None and image_b is not None:
            st.markdown("### Image previews")
            previews = st.columns(2, gap="large")
            for column, label, preview, info, filename in (
                (previews[0], "IMAGE A", info_a["preview"], info_a, st.session_state.get("image_a_name", "Image A")),
                (previews[1], "IMAGE B", info_b["preview"], info_b, st.session_state.get("image_b_name", "Image B")),
            ):
                column.markdown(
                    f'<div class="upload-card"><div class="upload-label">{label}</div><div class="image-meta">{html.escape(filename)}<br>{info["width"]} × {info["height"]} px · {info["channels"]} channel(s) · {_format_bytes(info["file_size"])}</div></div>',
                    unsafe_allow_html=True,
                )
                column.image(preview, width="stretch")
            st.markdown("### Preprocessing")
            checks = st.columns(4, gap="small")
            for column, label in zip(checks, ("✓ Image validation", "✓ Size normalization", "✓ Grayscale conversion", "✓ Dimension matching")):
                column.success(label)
            dims = st.columns(2)
            dims[0].metric("Original A → processed", f"{info_a['width']} × {info_a['height']} → {image_a.shape[1]} × {image_a.shape[0]} px")
            dims[1].metric("Original B → processed", f"{info_b['width']} × {info_b['height']} → {image_b.shape[1]} × {image_b.shape[0]} px")
            if max(info_a["width"], info_a["height"], info_b["width"], info_b["height"]) > MAX_PROCESSING_SIDE:
                st.warning(f"Large input detected. Fusion is capped at {MAX_PROCESSING_SIDE} px on the longest side.")
            active_signature = _signature(stored_a, stored_b, wavelet, requested_level, fusion_method, size_mode)
            if st.button("⚡ Fuse Images", type="primary", width="stretch", key="fuse_images"):
                try:
                    with st.spinner("Applying DWT, SVD fusion, and inverse reconstruction…"):
                        computed = _fusion_cached(image_a, image_b, wavelet, requested_level, fusion_method)
                    st.session_state["fusion_result"] = {**computed, "signature": active_signature}
                    st.success("Fusion complete. Review the analysis pages or download the result below.")
                except Exception as error:
                    LOGGER.exception("DWT-SVD fusion failed")
                    st.error("Fusion could not be completed. Verify both images and try a lower DWT level or the default fusion rule.")
                    st.caption(f"Processing detail: {error}")
            current = _current_result(active_signature)
            if current:
                st.markdown("## ✨ Fusion Result")
                st.image(current["fused"], caption=f"Reconstructed output · {current['fused'].shape[1]} × {current['fused'].shape[0]} px", width="stretch")
                st.markdown("### Input A → Input B → Fused Result")
                _render_comparison(current)
                _render_download(current)
            elif st.session_state.get("fusion_result"):
                st.caption("Inputs or processing settings changed. Run fusion again to refresh the result.")

elif page == "📊 Analysis":
    _render_section_intro("WORKSPACE 02", "Visual Analysis", "Compare grayscale intensity distributions and summary statistics for both sources and the fused result.")
    if result is None:
        st.info("Run fusion from Image Fusion to unlock histogram and image-statistics analysis.")
    else:
        figure = intensity_histogram_figure(result["image_a"] * 255, result["image_b"] * 255, result["fused"])
        st.pyplot(figure, width="stretch")
        plt.close(figure)
        rows = []
        for name, array in (("Image A", result["image_a"]), ("Image B", result["image_b"]), ("Fused", result["fused"])):
            values = np.asarray(array, dtype=np.float64)
            if values.max(initial=0) <= 1:
                values *= 255
            quantized = np.rint(values).clip(0, 255).astype(np.uint8)
            entropy = _entropy_u8(quantized)
            rows.append({"Image": name, "Width": int(array.shape[1]), "Height": int(array.shape[0]), "Mean intensity": float(np.mean(values)), "Standard deviation": float(np.std(values)), "Entropy (bits)": entropy})
        st.dataframe(rows, hide_index=True, width="stretch")

elif page == "📐 DWT Decomposition":
    _render_section_intro("WORKSPACE 03", "DWT Decomposition", "Inspect approximation and directional detail bands calculated from each preprocessed input.")
    if result is None:
        st.info("Run fusion from Image Fusion to view the DWT coefficient planes.")
    else:
        summary = coefficient_summary(result["coefficients_a"])
        st.caption(f"Wavelet: {result['wavelet']} · Actual level: {result['level']} · Coarsest LL shape: {summary['approximation_shape']}")
        columns = st.columns(2, gap="large")
        for column, coefficients, label in zip(columns, (result["coefficients_a"], result["coefficients_b"]), ("Image A", "Image B")):
            figure = decomposition_figure(coefficients, label)
            column.pyplot(figure, width="stretch")
            plt.close(figure)
        st.markdown("<div class='analysis-card'><b>Reading the coefficient planes</b><br>LL captures coarse approximation. LH, HL, and HH contain directional detail responses. Detail panels use symmetric contrast to show positive and negative coefficients.</div>", unsafe_allow_html=True)

elif page == "🔬 SVD Analysis":
    _render_section_intro("WORKSPACE 04", "SVD Analysis", "Explore the LL factorization used by fusion without displaying unwieldy raw matrices.")
    if result is None:
        st.info("Run fusion from Image Fusion to inspect the SVD factors.")
    else:
        analysis = result["svd_analysis"]
        factors_a = analysis["image_a"]
        factors_b = analysis["image_b"]
        plot = singular_spectrum_figure(factors_a["singular_values"], factors_b["singular_values"])
        st.pyplot(plot, width="stretch")
        plt.close(plot)
        st.caption("A = U Σ Vᵀ. Heatmaps show sampled matrix coordinates and retained modes; full raw matrices are not dumped into the page.")
        columns = st.columns(2, gap="large")
        for column, key, label in zip(columns, ("image_a", "image_b"), ("Image A", "Image B")):
            factors = analysis[key]
            figure = svd_factor_figure(factors, label)
            column.pyplot(figure, width="stretch")
            plt.close(figure)
            column.metric("Retained rank", int(factors["rank"]))
            column.metric("Retained LL energy", f"{float(factors['retained_energy_fraction']):.2%}")
        st.info(f"LL fusion weights: Image A {analysis['weight_a']:.2%} · Image B {analysis['weight_b']:.2%}.")
        st.caption("Default mode applies SVD energy fusion to LL and maximum-absolute selection to detail bands." if result["method"] == "SVD energy + max-absolute details" else "All-bands mode applies SVD energy fusion to detail coefficient matrices as well as LL.")

elif page == "📈 Quality Metrics":
    _render_section_intro("WORKSPACE 05", "Quality Metrics", "Read source-relative metrics and compare both inputs with the reconstruction.")
    if result is None:
        st.info("Run fusion from Image Fusion to calculate quality metrics.")
    else:
        _render_metric_cards(result)
        st.caption("MSE, PSNR, and SSIM compare the fused result independently against each aligned source. No ground-truth fused image is available. Entropy describes the fused image's intensity distribution.")
        st.dataframe([{"Reference": "Image A", **result["metrics_a"]}, {"Reference": "Image B", **result["metrics_b"]}], hide_index=True, width="stretch")
        _render_comparison(result)

elif page == "ℹ About Project":
    _render_section_intro("PROJECT NOTES", "About the Project", "A classical Digital Image Processing implementation built around wavelet analysis and matrix factorization.")
    st.markdown("### Image Fusion Using DWT and SVD\n**Domain:** Digital Image Processing  \n**Techniques:** DWT · SVD · Coefficient Fusion · Inverse DWT · Image Quality Assessment  \n**Technology:** Python · OpenCV · NumPy · PyWavelets · Matplotlib · scikit-image · Streamlit")
    st.markdown("### Processing pipeline\n`Input Images → Preprocessing → DWT Decomposition → SVD Analysis → Fusion Rule → Inverse DWT → Fused Image → Quality Assessment`")
    st.markdown("### About the algorithm\nThe 2D DWT separates each grayscale image into a coarse LL approximation and three directional detail bands at each scale. For LL, compact SVD is computed for both inputs, and the leading 10% of singular values (at least one) define low-rank approximations. Their retained squared-singular-value energies provide normalized blend weights. By default, each detail coefficient is selected from the input with the larger absolute response. The optional all-bands mode uses the SVD energy rule for detail matrices too. Inverse DWT reconstructs and clips the grayscale output to 8-bit range.")
    st.warning("MSE, PSNR, and SSIM compare with each aligned source, not a ground-truth fused image. Entropy describes intensity distribution and is not a standalone quality score.")
    st.markdown(f"### Deployment characteristics\nImages are decoded and processed in memory. Each upload is limited to {_format_bytes(MAX_UPLOAD_BYTES)} and decoded images to 20 megapixels. The longest processing side is capped at {MAX_PROCESSING_SIDE} px. No local database, external API, model weights, or secrets are required.")
