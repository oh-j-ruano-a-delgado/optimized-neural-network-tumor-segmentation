"""Bilingual copy for the Streamlit interface (English and Spanish)."""

from types import MappingProxyType
from typing import Final, Mapping

ENGLISH: Final[str] = "en"
SPANISH: Final[str] = "es"
DEFAULT_LANGUAGE: Final[str] = ENGLISH
LANGUAGE_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {ENGLISH: "English", SPANISH: "Español"}
)

TRANSLATIONS: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        ENGLISH: MappingProxyType(
            {
                "app_title": "Brain Tumor Segmentation",
                "app_subtitle": (
                    "Pick an MRI scan and see how a neural network finds brain tumors. "
                    "No medical knowledge needed."
                ),
                "language_label": "Language",
                "intro_title": "How does this work?",
                "intro_body": (
                    "1. Choose an MRI slice from the examples (or upload a PNG).\n"
                    "2. A U-Net, a convolutional neural network trained on 2,004 labelled "
                    "slices, paints the tumor regions it recognizes.\n"
                    "3. Colors tell the predicted tumor type and the confidence map shows "
                    "how sure the model is on every pixel.\n\n"
                    "The three tumor types it knows are meningioma, glioma and pituitary "
                    "tumors. This is a research demo, not a diagnostic tool."
                ),
                "step1_title": "1 · Choose an image",
                "source_label": "Image source",
                "source_examples": "Example scans",
                "source_upload": "Upload a PNG",
                "examples_hint": (
                    "These slices belong to the test set, so the model never saw them "
                    "during training."
                ),
                "example_select_label": "Example scan",
                "class_meningioma": "Meningioma",
                "class_glioma": "Glioma",
                "class_pituitary": "Pituitary tumor",
                "class_meningioma_desc": (
                    "Meningioma: grows from the membranes that cover the brain."
                ),
                "class_glioma_desc": (
                    "Glioma: grows from the glial cells that support neurons."
                ),
                "class_pituitary_desc": (
                    "Pituitary tumor: grows in the pituitary gland, at the base of the brain."
                ),
                "upload_label": "MRI slice (PNG)",
                "upload_hint": (
                    "Slices preprocessed like the dataset work best (256×256 grayscale, "
                    "head centred)."
                ),
                "no_file_info": "Pick an example scan above or upload your own PNG to start.",
                "no_samples_warning": "No bundled samples were found in `app/samples`.",
                "running_inference": "Running inference...",
                "step2_title": "2 · What the model sees",
                "view_input": "Input slice (as the model sees it)",
                "view_overlay": "Tumor types on the scan",
                "view_mask": "Predicted regions only",
                "view_confidence": "Confidence (brighter = more sure)",
                "legend_caption": "Colors for each tumor type:",
                "step3_title": "3 · What did the model find?",
                "interpretation_tumor": (
                    "**Tumor region detected:** about **{area:.1f}%** of the slice, mostly "
                    "**{dominant}**."
                ),
                "interpretation_no_tumor": (
                    "**No tumor region detected** in this slice. The model is conservative: "
                    "small tumors can be missed, so this is not proof that the slice is healthy."
                ),
                "confidence_summary": (
                    "**Model confidence:** {value:.0%} on average inside the detected "
                    "region — {level}."
                ),
                "confidence_high": "high",
                "confidence_medium": "medium",
                "confidence_low": "low",
                "confidence_high_hint": (
                    "The model is very sure, but this is still a research prototype."
                ),
                "confidence_medium_hint": (
                    "The model is fairly sure; check the confidence map for weaker spots."
                ),
                "confidence_low_hint": (
                    "The model is unsure here; treat the result with caution."
                ),
                "ground_truth": "Ground truth from the dataset: {label}.",
                "coverage_title": "Share of the slice by region",
                "column_region": "Region",
                "column_percentage": "Share (%)",
                "region_background": "Healthy tissue / background",
                "download_mask": "Download mask (PNG)",
                "download_overlay": "Download overlay (PNG)",
                "advanced_title": "Advanced settings",
                "model_checkpoint_label": "Checkpoint",
                "model_e": "E · U-Net from scratch",
                "model_f": "F · ResNet34 U-Net (transfer)",
                "model_e_desc": (
                    "Trained from random initialization; the genetic algorithm chose its "
                    "learning rate, weight decay, filters and depth."
                ),
                "model_f_desc": (
                    "Starts from a ResNet34 encoder pre-trained on ImageNet and is fine-tuned "
                    "on MRI slices."
                ),
                "model_missing": "Only the checkpoints present in `app/models` can be selected.",
                "model_card_architecture": "Architecture: {architecture}",
                "model_card_parameters": "Trainable parameters: {parameters}",
                "model_card_test_dice": (
                    "Held-out test Dice: {mean} mean · {whole} whole tumor"
                ),
                "model_card_training": "Training budget: {epochs} epochs · {minutes:.1f} min",
                "challenging_suffix": "challenging case",
                "challenging_hint": (
                    "Challenging case: model F returns an empty mask here even though a small "
                    "glioma is present. Switch to model E to see the difference."
                ),
                "inference_time": "Inference time: {seconds:.2f} s",
                "overlay_opacity": "Highlight opacity",
                "overlay_opacity_help": "How strongly the tumor colors cover the scan.",
                "confidence_threshold": "Hide uncertain pixels",
                "confidence_threshold_help": (
                    "Pixels where the model is less confident than this are shown as "
                    "background."
                ),
                "checkpoint_details": "Checkpoint details",
                "checkpoint_summary": (
                    "Epoch {epoch} · best validation Dice {dice} · device {device}"
                ),
                "disclaimer": (
                    "Research demo trained on the Figshare brain tumor dataset. It is not a "
                    "medical device and must not be used for diagnosis."
                ),
                "about_title": "About this demo",
                "about_body": (
                    "- Two checkpoints are bundled. E is a custom U-Net trained from "
                    "scratch; F is a U-Net with a ResNet34 encoder pre-trained on ImageNet "
                    "(transfer learning) and is the default.\n"
                    "- Both segment three tumor types (meningioma, glioma, pituitary) plus "
                    "background on 2D coronal slices of 256×256 pixels.\n"
                    "- Dice and IoU exclude the background because tumor regions cover a "
                    "tiny part of each slice.\n"
                    "- The genetic algorithm tunes the learning rate, weight decay, number "
                    "of filters and depth of the from-scratch models.\n"
                    "- The models are conservative and tend to miss small tumors; an empty "
                    "mask is not proof that the slice is healthy.\n"
                    "- Confidence is the maximum softmax probability on each pixel."
                ),
            }
        ),
        SPANISH: MappingProxyType(
            {
                "app_title": "Segmentación de tumores cerebrales",
                "app_subtitle": (
                    "Elige una resonancia y observa cómo una red neuronal localiza tumores "
                    "cerebrales. No hace falta saber medicina."
                ),
                "language_label": "Idioma",
                "intro_title": "¿Cómo funciona?",
                "intro_body": (
                    "1. Elige un corte de resonancia de los ejemplos (o sube un PNG).\n"
                    "2. Una U-Net, una red neuronal convolucional entrenada con 2.004 cortes "
                    "etiquetados, colorea las regiones tumorales que reconoce.\n"
                    "3. Los colores indican el tipo de tumor predicho y el mapa de confianza "
                    "muestra la seguridad del modelo en cada píxel.\n\n"
                    "Los tres tipos de tumor que conoce son meningioma, glioma y tumor de "
                    "hipófisis. Esta es una demostración de investigación, no una "
                    "herramienta de diagnóstico."
                ),
                "step1_title": "1 · Elige una imagen",
                "source_label": "Origen de la imagen",
                "source_examples": "Ejemplos del conjunto de test",
                "source_upload": "Subir un PNG",
                "examples_hint": (
                    "Estos cortes pertenecen al conjunto de test: el modelo no los vio "
                    "durante el entrenamiento."
                ),
                "example_select_label": "Corte de ejemplo",
                "class_meningioma": "Meningioma",
                "class_glioma": "Glioma",
                "class_pituitary": "Tumor de hipófisis",
                "class_meningioma_desc": (
                    "Meningioma: crece en las membranas que recubren el cerebro."
                ),
                "class_glioma_desc": (
                    "Glioma: crece a partir de las células gliales que sostienen las neuronas."
                ),
                "class_pituitary_desc": (
                    "Tumor de hipófisis: crece en la glándula hipófisis, en la base del "
                    "cerebro."
                ),
                "upload_label": "Corte de resonancia (PNG)",
                "upload_hint": (
                    "Funcionan mejor cortes preprocesados como el dataset (256×256 en "
                    "escala de grises y cabeza centrada)."
                ),
                "no_file_info": "Elige un ejemplo arriba o sube tu propio PNG para empezar.",
                "no_samples_warning": "No se han encontrado muestras en `app/samples`.",
                "running_inference": "Calculando la segmentación...",
                "step2_title": "2 · Lo que ve el modelo",
                "view_input": "Corte de entrada (tal como lo ve el modelo)",
                "view_overlay": "Tipos de tumor sobre la resonancia",
                "view_mask": "Solo las regiones predichas",
                "view_confidence": "Confianza (más brillante = más segura)",
                "legend_caption": "Color de cada tipo de tumor:",
                "step3_title": "3 · ¿Qué ha encontrado el modelo?",
                "interpretation_tumor": (
                    "**Región tumoral detectada:** aproximadamente el **{area:.1f}%** del "
                    "corte, sobre todo **{dominant}**."
                ),
                "interpretation_no_tumor": (
                    "**No se ha detectado ninguna región tumoral** en este corte. El modelo "
                    "es conservador: puede no ver tumores pequeños, así que esto no prueba "
                    "que el corte sea sano."
                ),
                "confidence_summary": (
                    "**Confianza del modelo:** {value:.0%} de media dentro de la región "
                    "detectada — {level}."
                ),
                "confidence_high": "alta",
                "confidence_medium": "media",
                "confidence_low": "baja",
                "confidence_high_hint": (
                    "El modelo está muy seguro, pero sigue siendo un prototipo de "
                    "investigación."
                ),
                "confidence_medium_hint": (
                    "El modelo está bastante seguro; revisa el mapa de confianza para ver "
                    "los puntos más débiles."
                ),
                "confidence_low_hint": (
                    "El modelo no está seguro aquí; interpreta el resultado con cautela."
                ),
                "ground_truth": "Etiqueta real del dataset: {label}.",
                "coverage_title": "Porcentaje del corte por región",
                "column_region": "Región",
                "column_percentage": "Porcentaje (%)",
                "region_background": "Tejido sano / fondo",
                "download_mask": "Descargar máscara (PNG)",
                "download_overlay": "Descargar superposición (PNG)",
                "advanced_title": "Opciones avanzadas",
                "model_checkpoint_label": "Checkpoint",
                "model_e": "E · U-Net desde cero",
                "model_f": "F · U-Net ResNet34 (transferencia)",
                "model_e_desc": (
                    "Entrenada desde inicialización aleatoria; el algoritmo genético eligió "
                    "su tasa de aprendizaje, regularización, filtros y profundidad."
                ),
                "model_f_desc": (
                    "Parte de un codificador ResNet34 preentrenado en ImageNet y se ajusta "
                    "con cortes de resonancia."
                ),
                "model_missing": (
                    "Solo se pueden elegir los checkpoints presentes en `app/models`."
                ),
                "model_card_architecture": "Arquitectura: {architecture}",
                "model_card_parameters": "Parámetros entrenables: {parameters}",
                "model_card_test_dice": (
                    "Dice en test reservado: {mean} medio · {whole} tumor completo"
                ),
                "model_card_training": (
                    "Presupuesto de entrenamiento: {epochs} épocas · {minutes:.1f} min"
                ),
                "challenging_suffix": "caso difícil",
                "challenging_hint": (
                    "Caso difícil: el modelo F devuelve una máscara vacía aunque hay un "
                    "glioma pequeño. Cambia al modelo E para ver la diferencia."
                ),
                "inference_time": "Tiempo de inferencia: {seconds:.2f} s",
                "overlay_opacity": "Opacidad del resaltado",
                "overlay_opacity_help": "Cuánto cubren los colores del tumor a la resonancia.",
                "confidence_threshold": "Ocultar píxeles poco seguros",
                "confidence_threshold_help": (
                    "Los píxeles en los que el modelo tiene menos confianza que este umbral "
                    "se muestran como fondo."
                ),
                "checkpoint_details": "Detalles del checkpoint",
                "checkpoint_summary": (
                    "Época {epoch} · mejor Dice de validación {dice} · dispositivo {device}"
                ),
                "disclaimer": (
                    "Demostración de investigación entrenada con el dataset Figshare de "
                    "tumores cerebrales. No es un dispositivo médico y no debe usarse para "
                    "diagnóstico."
                ),
                "about_title": "Sobre esta demo",
                "about_body": (
                    "- Se incluyen dos checkpoints. E es una U-Net propia entrenada desde "
                    "cero; F es una U-Net con codificador ResNet34 preentrenado en ImageNet "
                    "(transferencia de aprendizaje) y es la opción por defecto.\n"
                    "- Ambas segmentan tres tipos de tumor (meningioma, glioma e hipófisis) "
                    "y el fondo en cortes coronales 2D de 256×256 píxeles.\n"
                    "- Dice e IoU excluyen el fondo porque las regiones tumorales ocupan una "
                    "parte mínima de cada corte.\n"
                    "- El algoritmo genético ajusta la tasa de aprendizaje, la regularización, "
                    "el número de filtros y la profundidad de los modelos desde cero.\n"
                    "- Los modelos son conservadores y tienden a no ver tumores pequeños; una "
                    "máscara vacía no prueba que el corte sea sano.\n"
                    "- La confianza es la probabilidad softmax máxima en cada píxel."
                ),
            }
        ),
    }
)


def translate(language: str, key: str) -> str:
    return TRANSLATIONS.get(language, TRANSLATIONS[DEFAULT_LANGUAGE]).get(key, key)
