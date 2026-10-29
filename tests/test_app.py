"""Unit tests for the Streamlit interface inference helpers."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from streamlit.testing.v1 import AppTest

from app.i18n import (
    DEFAULT_LANGUAGE,
    ENGLISH,
    SPANISH,
    TRANSLATIONS,
    translate,
)
from app.segmentation import (
    BACKGROUND_LABEL,
    MODEL_CHECKPOINTS,
    PROJECT_ROOT,
    TUMOR_COLORS,
    apply_confidence_threshold,
    available_models,
    average_tumor_confidence,
    build_confidence_image,
    build_mask_image,
    build_overlay,
    class_pixel_percentages,
    classify_confidence,
    decode_image_bytes,
    discover_sample_images,
    dominant_class_name,
    encode_png,
    load_model_from_checkpoint,
    predict,
    preprocess_image,
    read_sample_image,
    to_display_image,
)
from src.dataset.constants import NUM_CLASSES, TUMOR_CLASS_NAMES
from src.models.unet import UNet

IMAGE_SIDE = 64
BASE_FILTERS = 4
DEPTH = 2
SYNTHETIC_MODEL_KEY = "synthetic"


@pytest.fixture
def checkpoint_path(tmp_path: Path) -> Path:
    network = UNet(
        in_channels=1, num_classes=NUM_CLASSES, base_filters=BASE_FILTERS, depth=DEPTH
    )
    payload = {
        "model_state_dict": network.state_dict(),
        "config": {
            "in_channels": 1,
            "num_classes": NUM_CLASSES,
            "base_filters": BASE_FILTERS,
            "depth": DEPTH,
        },
        "epoch": 7,
        "validation_dice": 0.5,
        "arguments": {"image_size": IMAGE_SIDE, "learning_rate": 1e-3},
    }
    path = tmp_path / "synthetic_unet.pth"
    torch.save(payload, path)
    return path


@pytest.fixture
def model(checkpoint_path: Path):
    return load_model_from_checkpoint(
        SYNTHETIC_MODEL_KEY, checkpoint_path, device=torch.device("cpu")
    )


@pytest.fixture
def grayscale_slice() -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(96, 96), dtype=np.uint8)


def test_load_model_from_checkpoint_rebuilds_network(model) -> None:
    assert model.network.training is False
    assert model.checkpoint.epoch == 7
    assert model.checkpoint.validation_dice == pytest.approx(0.5)
    assert model.image_size == (IMAGE_SIDE, IMAGE_SIDE)
    assert model.transform.augmentation is None


def test_available_models_reports_missing_checkpoints(tmp_path: Path) -> None:
    existing = tmp_path / "existing.pth"
    existing.write_bytes(b"placeholder")
    availability = available_models(
        {"present": existing, "missing": tmp_path / "missing.pth"}
    )
    assert availability == {"present": True, "missing": False}


def test_preprocess_image_returns_normalized_batch(model, grayscale_slice) -> None:
    preprocessed = preprocess_image(grayscale_slice, model)
    image_tensor = preprocessed.tensor

    assert tuple(image_tensor.shape) == (1, 1, IMAGE_SIDE, IMAGE_SIDE)
    assert float(image_tensor.min()) >= 0.0
    assert float(image_tensor.max()) <= 1.0
    assert preprocessed.display_image.shape == (IMAGE_SIDE, IMAGE_SIDE)
    assert preprocessed.display_image.dtype == np.uint8


def test_preprocess_display_matches_model_input(model, grayscale_slice) -> None:
    preprocessed = preprocess_image(grayscale_slice, model)
    expected_display = np.clip(preprocessed.tensor[0, 0].numpy() * 255.0, 0.0, 255.0)
    assert np.allclose(preprocessed.display_image, expected_display, atol=1.0)


def test_predict_returns_consistent_arrays(model, grayscale_slice) -> None:
    preprocessed = preprocess_image(grayscale_slice, model)
    result = predict(model, preprocessed.tensor)

    assert result.mask.shape == (IMAGE_SIDE, IMAGE_SIDE)
    assert result.mask.dtype == np.uint8
    assert result.mask.max() < NUM_CLASSES
    assert result.probabilities.shape == (NUM_CLASSES, IMAGE_SIDE, IMAGE_SIDE)
    assert result.confidence.shape == (IMAGE_SIDE, IMAGE_SIDE)
    assert float(result.confidence.min()) >= 0.0
    assert float(result.confidence.max()) <= 1.0
    assert sum(result.class_pixel_percentages.values()) == pytest.approx(100.0)


def test_class_pixel_percentages_count_every_label() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:5, :] = 1
    mask[5:8, :] = 2
    percentages = class_pixel_percentages(mask)
    assert percentages["background"] == pytest.approx(20.0)
    assert percentages["meningioma"] == pytest.approx(50.0)
    assert percentages["glioma"] == pytest.approx(30.0)
    assert percentages["pituitary"] == pytest.approx(0.0)


def test_dominant_class_returns_none_for_empty_mask() -> None:
    assert dominant_class_name(np.zeros((8, 8), dtype=np.uint8)) is None


def test_dominant_class_returns_most_frequent_tumor() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[:4, :] = 3
    mask[4:6, :] = 1
    assert dominant_class_name(mask) == "pituitary"


def test_confidence_threshold_removes_uncertain_pixels() -> None:
    mask = np.full((4, 4), 2, dtype=np.uint8)
    confidence = np.full((4, 4), 0.4, dtype=np.float32)
    filtered = apply_confidence_threshold(mask, confidence, threshold=0.5)
    assert np.all(filtered == BACKGROUND_LABEL)

    untouched = apply_confidence_threshold(mask, confidence, threshold=0.0)
    assert np.array_equal(untouched, mask)


def test_build_overlay_only_changes_tumor_pixels() -> None:
    display_image = np.full((8, 8, 3), 120, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:2, :] = 1

    overlay = build_overlay(display_image, mask, alpha=0.5)

    assert overlay.shape == display_image.shape
    assert overlay.dtype == np.uint8
    assert np.array_equal(overlay[2:, :], display_image[2:, :])
    expected_color = 0.5 * 120 + 0.5 * np.array(TUMOR_COLORS[1])
    assert overlay[0, 0].tolist() == pytest.approx(expected_color.tolist(), abs=1.0)


def test_build_mask_and_confidence_images_have_rgb_shape() -> None:
    mask = np.zeros((6, 6), dtype=np.uint8)
    mask[0, 0] = 3
    confidence = np.linspace(0.0, 1.0, 36).reshape(6, 6).astype(np.float32)

    mask_image = build_mask_image(mask)
    confidence_image = build_confidence_image(confidence)

    assert mask_image.shape == (6, 6, 3)
    assert confidence_image.shape == (6, 6, 3)
    assert mask_image[0, 0].tolist() == list(TUMOR_COLORS[3])
    assert mask_image[3, 3].tolist() == [0, 0, 0]


def test_build_overlay_accepts_grayscale_input() -> None:
    grayscale = np.full((8, 8), 90, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[:2, :] = 2

    overlay = build_overlay(grayscale, mask, alpha=0.5)

    assert overlay.shape == (8, 8, 3)
    assert overlay[4, 4].tolist() == [90, 90, 90]


def test_encode_and_decode_png_round_trip() -> None:
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:8, :] = (77, 77, 77)

    encoded = encode_png(image)
    decoded = decode_image_bytes(encoded)

    assert encoded.startswith(b"\x89PNG")
    assert decoded.shape == (16, 16)
    assert int(decoded[2, 2]) == pytest.approx(77, abs=2)


def test_decode_image_bytes_rejects_invalid_payload() -> None:
    with pytest.raises(ValueError, match="not a decodable PNG"):
        decode_image_bytes(b"this is not an image")


def test_bundled_samples_cover_every_tumor_class() -> None:
    samples = discover_sample_images()
    assert {sample.class_name for sample in samples} == set(TUMOR_CLASS_NAMES.values())
    assert len(samples) >= len(TUMOR_CLASS_NAMES)
    for sample in samples:
        image = read_sample_image(sample.path)
        assert image.ndim == 2
        assert image.dtype in (np.uint8, np.uint16)


def test_read_sample_image_preserves_sixteen_bit_depth(tmp_path: Path) -> None:
    values = np.array([[0, 1200], [2400, 4000]], dtype=np.uint16)
    path = tmp_path / "sixteen_bit.png"
    cv2.imwrite(str(path), values)

    loaded = read_sample_image(path)

    assert loaded.dtype == np.uint16
    assert int(loaded.max()) == 4000


def test_bundled_samples_are_not_quantized_to_eight_bits() -> None:
    sample = discover_sample_images()[0]
    image = read_sample_image(sample.path)

    assert image.dtype == np.uint16
    assert int(image.max()) > 255


def test_to_display_image_converts_grayscale_to_rgb(grayscale_slice) -> None:
    display = to_display_image(grayscale_slice)
    assert display.shape == (96, 96, 3)
    assert np.array_equal(display[:, :, 0], grayscale_slice)


@pytest.mark.parametrize("model_key", sorted(MODEL_CHECKPOINTS))
def test_bundled_checkpoints_are_loadable(model_key: str) -> None:
    checkpoint = MODEL_CHECKPOINTS[model_key]
    assert checkpoint.is_file()
    model = load_model_from_checkpoint(model_key, checkpoint, device=torch.device("cpu"))
    image = np.random.default_rng(1).integers(0, 256, size=(64, 64), dtype=np.uint8)
    result = predict(model, preprocess_image(image, model).tensor)
    assert result.mask.shape == (256, 256)


def test_streamlit_app_renders_results_without_exceptions() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=180).run()

    assert not app.exception
    assert app.title[0].value == translate(DEFAULT_LANGUAGE, "app_title")
    assert len(app.dataframe) == 1
    assert len(app.get("download_button")) == 2
    assert len(app.selectbox) == 1
    assert {radio.label for radio in app.radio} == {
        "Language",
        "Image source",
        "Checkpoint",
    }


def test_streamlit_app_switches_to_spanish() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=180).run()

    language_selector = next(radio for radio in app.radio if radio.label == "Language")
    language_selector.set_value(SPANISH).run()

    assert not app.exception
    assert app.title[0].value == translate(SPANISH, "app_title")
    assert translate(SPANISH, "app_title") != translate(ENGLISH, "app_title")
    assert {button.label for button in app.get("download_button")} == {
        translate(SPANISH, "download_mask"),
        translate(SPANISH, "download_overlay"),
    }


def test_streamlit_app_upload_mode_waits_for_a_file() -> None:
    app = AppTest.from_file(str(PROJECT_ROOT / "app" / "main.py"), default_timeout=180).run()

    image_source = next(radio for radio in app.radio if radio.label == "Image source")
    image_source.set_value("upload").run()

    assert not app.exception
    assert app.info
    assert not app.get("download_button")


def test_translations_are_complete_for_every_language() -> None:
    assert set(TRANSLATIONS) == {ENGLISH, SPANISH}
    english_keys = set(TRANSLATIONS[ENGLISH])
    assert set(TRANSLATIONS[SPANISH]) == english_keys
    assert all(TRANSLATIONS[ENGLISH][key] for key in english_keys)
    assert all(TRANSLATIONS[SPANISH][key] for key in english_keys)


def test_translate_falls_back_to_english() -> None:
    assert translate("fr", "app_title") == translate(ENGLISH, "app_title")
    assert translate(ENGLISH, "missing_key") == "missing_key"


def test_average_tumor_confidence_ignores_background() -> None:
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[:2, :] = 2
    confidence = np.full((4, 4), 0.1, dtype=np.float32)
    confidence[:2, :] = 0.8

    assert average_tumor_confidence(mask, confidence) == pytest.approx(0.8)
    assert average_tumor_confidence(np.zeros((4, 4), dtype=np.uint8), confidence) is None


def test_classify_confidence_uses_thresholds() -> None:
    assert classify_confidence(0.95) == "high"
    assert classify_confidence(0.80) == "medium"
    assert classify_confidence(0.50) == "low"
