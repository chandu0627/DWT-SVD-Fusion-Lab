"""End-to-end checks for image inputs and the Streamlit empty state."""

from __future__ import annotations

import unittest
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from fusion import (
    calculate_metrics,
    decode_uploaded_image,
    decompose_image,
    fuse_coefficients,
    prepare_image_pair,
    reconstruct_image,
)


class FusionScenarioTests(unittest.TestCase):
    def fuse_pair(self, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, dict]:
        prepared_a, prepared_b = prepare_image_pair(first, second)
        coefficients_a = decompose_image(prepared_a, wavelet="db2", level=2)
        coefficients_b = decompose_image(prepared_b, wavelet="db2", level=2)
        fused_coefficients = fuse_coefficients(coefficients_a, coefficients_b)
        fused = reconstruct_image(
            fused_coefficients,
            wavelet="db2",
            output_shape=prepared_a.shape,
        )
        metrics_a = calculate_metrics(
            np.rint(prepared_a * 255).astype(np.uint8), fused
        )
        metrics_b = calculate_metrics(
            np.rint(prepared_b * 255).astype(np.uint8), fused
        )
        self.assertEqual(fused.shape, prepared_a.shape)
        self.assertEqual(fused.dtype, np.uint8)
        self.assertTrue(np.isfinite(fused).all())
        for metrics in (metrics_a, metrics_b):
            self.assertTrue(np.isfinite(metrics["MSE"]))
            self.assertFalse(np.isnan(metrics["PSNR (dB)"]))
            self.assertTrue(
                metrics["SSIM"] is None or np.isfinite(metrics["SSIM"])
            )
            self.assertTrue(np.isfinite(metrics["Entropy (bits)"]))
        return fused, {"image_1": metrics_a, "image_2": metrics_b}

    def test_01_same_size_grayscale(self) -> None:
        first = np.tile(np.arange(64, dtype=np.uint8), (64, 1))
        second = np.flipud(first).copy()
        self.fuse_pair(first, second)

    def test_02_different_size_grayscale(self) -> None:
        first = np.full((80, 96), 80, dtype=np.uint8)
        second = np.full((45, 72), 180, dtype=np.uint8)
        fused, _ = self.fuse_pair(first, second)
        self.assertEqual(fused.shape, first.shape)

    def test_03_two_rgb_images(self) -> None:
        first = np.zeros((48, 64, 3), dtype=np.uint8)
        first[..., 0] = 255
        second = np.zeros_like(first)
        second[..., 2] = 255
        self.fuse_pair(first, second)

    def test_04_mixed_grayscale_and_rgb(self) -> None:
        first = np.full((48, 64), 110, dtype=np.uint8)
        second = np.zeros((48, 64, 3), dtype=np.uint8)
        second[..., 1] = 200
        self.fuse_pair(first, second)

    def test_05_very_small_images(self) -> None:
        first = np.array([[0, 255], [255, 0]], dtype=np.uint8)
        second = np.array([[255, 0], [0, 255]], dtype=np.uint8)
        fused, metrics = self.fuse_pair(first, second)
        self.assertEqual(fused.shape, (2, 2))
        self.assertIsNone(metrics["image_1"]["SSIM"])

    def test_06_large_images(self) -> None:
        generator = np.random.default_rng(2026)
        first = generator.integers(0, 256, size=(1024, 1024), dtype=np.uint8)
        second = generator.integers(0, 256, size=(1024, 1024), dtype=np.uint8)
        self.fuse_pair(first, second)

    def test_07_invalid_upload(self) -> None:
        with self.assertRaises(ValueError):
            decode_uploaded_image(b"not an image")
        with self.assertRaises(ValueError):
            decode_uploaded_image(b"")

    def test_08_optional_all_bands_svd_mode(self) -> None:
        generator = np.random.default_rng(37)
        first = generator.random((32, 40), dtype=np.float32)
        second = generator.random((32, 40), dtype=np.float32)
        prepared_a, prepared_b = prepare_image_pair(first, second)
        coefficients_a = decompose_image(prepared_a, wavelet="haar", level=2)
        coefficients_b = decompose_image(prepared_b, wavelet="haar", level=2)
        fused_coefficients = fuse_coefficients(
            coefficients_a,
            coefficients_b,
            method="SVD energy all bands",
        )
        output = reconstruct_image(fused_coefficients, wavelet="haar", output_shape=prepared_a.shape)
        self.assertEqual(output.shape, prepared_a.shape)
        self.assertTrue(np.isfinite(output).all())

    def test_09_empty_streamlit_upload_state(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        app.radio[0].set_value("🖼 Image Fusion").run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.file_uploader), 2)
        self.assertNotIn("Run DWT-SVD fusion", [button.label for button in app.button])

    def test_10_streamlit_fusion_and_download(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=60).run()
        app.radio[0].set_value("🖼 Image Fusion").run()
        first = np.tile(np.arange(32, dtype=np.uint8), (32, 1))
        second = np.flipud(first).copy()
        encoded_images = []
        for pixels in (first, second):
            stream = BytesIO()
            Image.fromarray(pixels).save(stream, format="PNG")
            encoded_images.append(stream.getvalue())

        app.file_uploader[0].upload("first.png", encoded_images[0], "image/png")
        app.run()
        app.file_uploader[1].upload("second.png", encoded_images[1], "image/png")
        app.run()
        fusion_button = next(button for button in app.button if button.label == "⚡ Fuse Images")
        fusion_button.click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.success), 5)
        self.assertEqual(len(app.download_button), 1)
        self.assertIn(".png", app.download_button[0].proto.url)
        for page in (
            "📊 Analysis",
            "📐 DWT Decomposition",
            "🔬 SVD Analysis",
            "📈 Quality Metrics",
            "ℹ About Project",
            "🏠 Dashboard",
        ):
            app.radio[0].set_value(page).run()
            self.assertEqual(len(app.exception), 0, page)
            if page == "📈 Quality Metrics":
                self.assertEqual(len(app.dataframe), 1)

    def test_11_invalid_image_has_friendly_ui_error(self) -> None:
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        app = AppTest.from_file(str(app_path), default_timeout=30).run()
        app.radio[0].set_value("🖼 Image Fusion").run()
        app.file_uploader[0].upload("broken.png", b"not image bytes", "image/png")
        app.run()
        app.file_uploader[1].upload("valid-name.png", b"also invalid", "image/png")
        app.run()
        self.assertEqual(len(app.exception), 0)
        self.assertGreaterEqual(len(app.error), 1)

    def test_rgb_upload_decoding(self) -> None:
        rgb = np.zeros((16, 20, 3), dtype=np.uint8)
        rgb[..., 0] = 255
        ok, encoded = cv2.imencode(
            ".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        )
        self.assertTrue(ok)
        decoded = decode_uploaded_image(encoded.tobytes())
        self.assertEqual(decoded.shape, (16, 20))
        self.assertEqual(decoded.dtype, np.uint8)
        self.assertGreater(float(decoded.mean()), 50.0)


if __name__ == "__main__":
    unittest.main()