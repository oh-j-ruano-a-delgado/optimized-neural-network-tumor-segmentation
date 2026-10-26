"""Bilingual Streamlit demo that segments a preprocessed MRI slice with the trained U-Net."""

import math
import sys
from functools import partial
from pathlib import Path
from typing import Callable, Final, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from numpy.typing import NDArray

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]
# Streamlit only adds the entry script directory to sys.path; the repository
# root is required so both `src` and `app` imports resolve locally and on Cloud.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.i18n import (
    DEFAULT_LANGUAGE,
    LANGUAGE_LABELS,
    translate,
)
from app.segmentation import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_OVERLAY_ALPHA,
    OPTIMIZED_MODEL_KEY,
    TUMOR_CLASS_NAMES,
    TUMOR_COLORS,
    SampleImage,
    SegmentationModel,
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
    load_model,
    predict,
    preprocess_image,
    read_sample_image,
    to_display_image,
)

PAGE_TITLE_KEY: Final[str] = "app_title"
LANGUAGE_STATE_KEY: Final[str] = "language"
EXAMPLES_SOURCE: Final[str] = "examples"
UPLOAD_SOURCE: Final[str] = "upload"
SOURCE_OPTIONS: Final[Sequence[str]] = (EXAMPLES_SOURCE, UPLOAD_SOURCE)
OVERLAY_ALPHA_RANGE: Final[Tuple[float, float]] = (0.2, 0.9)
CONFIDENCE_THRESHOLD_RANGE: Final[Tuple[float, float]] = (0.0, 0.95)
SLIDER_STEP: Final[float] = 0.05
Translator = Callable[[str], str]


@st.cache_resource(show_spinner=False)
def get_model(model_key: str) -> SegmentationModel:
    return load_model(model_key)


def format_validation_dice(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.4f}"


def format_sample_label(sample: SampleImage, t: Translator) -> str:
    class_display = t(f"class_{sample.class_name}")
    return f"{class_display} — {sample.path.stem}"


def render_header() -> Translator:
    t: Translator = partial(translate, st.session_state[LANGUAGE_STATE_KEY])

    title_column, language_column = st.columns([4, 1])
    with language_column:
        st.radio(
            t("language_label"),
            tuple(LANGUAGE_LABELS),
            format_func=lambda code: LANGUAGE_LABELS[code],
            horizontal=True,
            label_visibility="collapsed",
            key=LANGUAGE_STATE_KEY,
        )
    with title_column:
        st.title(t(PAGE_TITLE_KEY))
        st.caption(t("app_subtitle"))

    with st.expander(t("intro_title")):
        st.markdown(t("intro_body"))
    return t


def render_image_source(t: Translator) -> Tuple[NDArray[np.uint8], Optional[str]]:
    st.subheader(t("step1_title"))
    source = st.radio(
        t("source_label"),
        SOURCE_OPTIONS,
        format_func=lambda option: t(f"source_{option}"),
        horizontal=True,
    )

    if source == EXAMPLES_SOURCE:
        samples = discover_sample_images()
        if not samples:
            st.warning(t("no_samples_warning"))
            st.stop()
        labels = [format_sample_label(sample, t) for sample in samples]
        selected_index = st.selectbox(
            t("example_select_label"),
            range(len(samples)),
            format_func=lambda index: labels[index],
        )
        selected_sample = samples[selected_index]
        st.caption(t(f"class_{selected_sample.class_name}_desc"))
        st.caption(t("examples_hint"))
        return read_sample_image(selected_sample.path), selected_sample.class_name

    uploaded_file = st.file_uploader(
        t("upload_label"), type=("png",), help=t("upload_hint")
    )
    if uploaded_file is None:
        st.info(t("no_file_info"))
        st.stop()
    return decode_image_bytes(uploaded_file.getvalue()), None


def render_advanced_settings(t: Translator, availability: Mapping[str, bool]) -> Tuple:
    available_keys = [key for key, present in availability.items() if present]
    default_key = (
        OPTIMIZED_MODEL_KEY if availability.get(OPTIMIZED_MODEL_KEY) else available_keys[0]
    )

    with st.expander(t("advanced_title")):
        model_key = st.radio(
            t("model_checkpoint_label"),
            available_keys,
            index=available_keys.index(default_key),
            format_func=lambda key: t(f"model_{key}"),
            horizontal=True,
        )
        if not availability.get(OPTIMIZED_MODEL_KEY):
            st.caption(t("optimized_missing"))
        overlay_alpha = st.slider(
            t("overlay_opacity"),
            *OVERLAY_ALPHA_RANGE,
            DEFAULT_OVERLAY_ALPHA,
            SLIDER_STEP,
            help=t("overlay_opacity_help"),
        )
        confidence_threshold = st.slider(
            t("confidence_threshold"),
            *CONFIDENCE_THRESHOLD_RANGE,
            DEFAULT_CONFIDENCE_THRESHOLD,
            SLIDER_STEP,
            help=t("confidence_threshold_help"),
        )
        model = get_model(model_key)
        checkpoint = model.checkpoint
        st.caption(
            t("checkpoint_summary").format(
                epoch=checkpoint.epoch,
                dice=format_validation_dice(checkpoint.validation_dice),
                device=model.device.type,
            )
        )
    return model_key, overlay_alpha, confidence_threshold


def render_legend(t: Translator) -> None:
    st.caption(t("legend_caption"))
    chips: List[str] = []
    for label, class_name in TUMOR_CLASS_NAMES.items():
        red, green, blue = TUMOR_COLORS[label]
        chips.append(
            f'<span style="background:rgb({red},{green},{blue}); color:#0B0F1A; '
            f"padding:0.15rem 0.6rem; border-radius:0.4rem; font-weight:600; "
            f'margin-right:0.4rem;">{t(f"class_{class_name}")}</span>'
        )
    st.markdown(" ".join(chips), unsafe_allow_html=True)


def render_views(
    t: Translator,
    display_image: NDArray[np.uint8],
    mask: NDArray[np.uint8],
    confidence: NDArray[np.float32],
    overlay_alpha: float,
) -> NDArray[np.uint8]:
    st.subheader(t("step2_title"))
    overlay = build_overlay(display_image, mask, overlay_alpha)
    views = (
        (display_image, t("view_input")),
        (overlay, t("view_overlay")),
        (build_mask_image(mask), t("view_mask")),
        (build_confidence_image(confidence), t("view_confidence")),
    )
    for column, (view_image, caption) in zip(st.columns(len(views)), views):
        column.image(view_image, caption=caption, width="stretch")
    render_legend(t)
    return overlay


def render_interpretation(
    t: Translator,
    mask: NDArray[np.uint8],
    confidence: NDArray[np.float32],
    ground_truth_class: Optional[str],
) -> None:
    st.subheader(t("step3_title"))
    dominant_class = dominant_class_name(mask)
    if dominant_class is None:
        st.info(t("interpretation_no_tumor"))
    else:
        percentages = class_pixel_percentages(mask)
        st.success(
            t("interpretation_tumor").format(
                area=percentages[dominant_class],
                dominant=t(f"class_{dominant_class}"),
            )
        )
        average_confidence = average_tumor_confidence(mask, confidence)
        if average_confidence is not None:
            level = classify_confidence(average_confidence)
            st.markdown(
                t("confidence_summary").format(
                    value=average_confidence, level=t(f"confidence_{level}")
                )
                + " "
                + t(f"confidence_{level}_hint")
            )
    if ground_truth_class is not None:
        st.caption(
            t("ground_truth").format(label=t(f"class_{ground_truth_class}"))
        )


def render_statistics(t: Translator, mask: NDArray[np.uint8]) -> None:
    rows = [
        {
            t("column_region"): t("region_background")
            if region == "background"
            else t(f"class_{region}"),
            t("column_percentage"): round(percentage, 2),
        }
        for region, percentage in class_pixel_percentages(mask).items()
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def render_downloads(
    t: Translator,
    mask_image: NDArray[np.uint8],
    overlay: NDArray[np.uint8],
) -> None:
    for column, (label, download_image, file_name) in zip(
        st.columns(2),
        (
            (t("download_mask"), mask_image, "predicted_mask.png"),
            (t("download_overlay"), overlay, "prediction_overlay.png"),
        ),
    ):
        column.download_button(
            label, encode_png(download_image), file_name=file_name, mime="image/png"
        )


def main() -> None:
    st.session_state.setdefault(LANGUAGE_STATE_KEY, DEFAULT_LANGUAGE)
    st.set_page_config(
        page_title=translate(st.session_state[LANGUAGE_STATE_KEY], PAGE_TITLE_KEY),
        layout="wide",
    )
    t = render_header()

    availability = available_models()
    if not any(availability.values()):
        st.error(
            "No trained checkpoint was found in `app/models`. Copy "
            "`runs/baseline/best_model.pth` to `app/models/baseline_unet.pth` first."
        )
        st.stop()

    image, ground_truth_class = render_image_source(t)
    model_key, overlay_alpha, confidence_threshold = render_advanced_settings(t, availability)

    model = get_model(model_key)
    with st.spinner(t("running_inference")):
        preprocessed = preprocess_image(image, model)
        prediction = predict(model, preprocessed.tensor)
    mask = apply_confidence_threshold(
        prediction.mask, prediction.confidence, confidence_threshold
    )

    overlay = render_views(
        t, preprocessed.display_image, mask, prediction.confidence, overlay_alpha
    )
    render_interpretation(t, mask, prediction.confidence, ground_truth_class)
    st.subheader(t("coverage_title"))
    render_statistics(t, mask)
    render_downloads(t, build_mask_image(mask), overlay)

    with st.expander(t("about_title")):
        st.markdown(t("about_body"))
    st.caption(t("disclaimer"))


if __name__ == "__main__":
    main()
