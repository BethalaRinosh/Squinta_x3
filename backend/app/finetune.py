"""
LoRA fine-tuning system for personalising TrOCR to individual handwriting.

Uses PEFT (Parameter-Efficient Fine-Tuning) to train lightweight LoRA
adapters on top of the frozen TrOCR-large-handwritten base model, driven
by user-submitted corrections.
"""

from __future__ import annotations

import logging
import math
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from datasets import Dataset
from PIL import Image
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from app.config import settings

logger = logging.getLogger(__name__)

BASE_MODEL_NAME = "microsoft/trocr-large-handwritten"

# ---------------------------------------------------------------------------
# LoRA configuration -- targets the decoder cross- and self-attention
# query / value projections, which capture most of the handwriting style
# information while keeping the number of trainable parameters low.
# ---------------------------------------------------------------------------

DEFAULT_LORA_CONFIG = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
)


# ---------------------------------------------------------------------------
# Custom data collator for TrOCR fine-tuning
# ---------------------------------------------------------------------------


class TrOCRDataCollator:
    """Collate a list of dicts with ``pixel_values`` and ``labels`` into
    batched tensors suitable for ``Seq2SeqTrainer``."""

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pixel_values = torch.stack([f["pixel_values"] for f in features])
        label_ids = [f["labels"] for f in features]

        # Pad labels to the same length within the batch.
        max_len = max(len(ids) for ids in label_ids)
        padded = []
        for ids in label_ids:
            padding_length = max_len - len(ids)
            # -100 is the ignore index used by CrossEntropyLoss.
            padded.append(ids + [-100] * padding_length)

        labels = torch.tensor(padded, dtype=torch.long)
        return {"pixel_values": pixel_values, "labels": labels}


# ---------------------------------------------------------------------------
# FineTuner
# ---------------------------------------------------------------------------


class FineTuner:
    """Manages LoRA fine-tuning of TrOCR for a specific user."""

    def __init__(
        self,
        base_model_name: str = BASE_MODEL_NAME,
        model_dir: str | None = None,
    ) -> None:
        self.base_model_name = base_model_name
        self.model_dir = Path(model_dir or settings.MODEL_DIR)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Lazily loaded -- avoids holding a large model in memory when
        # we are only scoring or exporting.
        self._processor: Optional[TrOCRProcessor] = None
        self._base_model: Optional[VisionEncoderDecoderModel] = None

    # ----- lazy loaders ----------------------------------------------------

    def _get_processor(self) -> TrOCRProcessor:
        if self._processor is None:
            self._processor = TrOCRProcessor.from_pretrained(self.base_model_name)
        return self._processor

    def _get_base_model(self) -> VisionEncoderDecoderModel:
        if self._base_model is None:
            self._base_model = VisionEncoderDecoderModel.from_pretrained(
                self.base_model_name
            )
        return self._base_model

    # ----- path helpers ----------------------------------------------------

    def _user_dir(self, user_id: int) -> Path:
        return self.model_dir / f"user_{user_id}"

    def _latest_version(self, user_id: int) -> int:
        """Return the latest version number for a user, or 0 if none exists."""
        user_dir = self._user_dir(user_id)
        if not user_dir.exists():
            return 0
        versions = [
            int(p.name[1:])
            for p in user_dir.iterdir()
            if p.is_dir() and p.name.startswith("v") and p.name[1:].isdigit()
        ]
        return max(versions) if versions else 0

    def _latest_lora_path(self, user_id: int) -> Optional[Path]:
        """Return the path to the latest LoRA weights, or None.

        Only returns a path if the adapter was fully saved (adapter_config.json exists).
        """
        version = self._latest_version(user_id)
        if version == 0:
            return None
        path = self._user_dir(user_id) / f"v{version}"
        if path.exists() and (path / "adapter_config.json").exists():
            return path
        return None

    # ----- data preparation ------------------------------------------------

    def prepare_training_data(
        self, corrections: list[dict]
    ) -> Dataset:
        """Convert a list of correction dicts into a HuggingFace ``Dataset``.

        Each item in *corrections* must contain:
        - ``page_image_path`` (str): path to the full page image on disk
        - ``bbox`` (dict or tuple): ``{x, y, w, h}`` or ``(x, y, w, h)``
        - ``corrected_text`` (str): the ground-truth text for that crop

        The returned dataset has columns ``pixel_values`` (tensor) and
        ``labels`` (list of token ids).
        """
        processor = self._get_processor()

        pixel_values_list: list[torch.Tensor] = []
        labels_list: list[list[int]] = []

        for correction in corrections:
            # --- crop the image ------------------------------------------------
            page_img = Image.open(correction["page_image_path"]).convert("RGB")

            bbox = correction["bbox"]
            if isinstance(bbox, dict):
                x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
            else:
                x, y, w, h = bbox

            crop = page_img.crop((x, y, x + w, y + h))

            # --- encode image --------------------------------------------------
            pixel_values = processor(crop, return_tensors="pt").pixel_values.squeeze(0)
            pixel_values_list.append(pixel_values)

            # --- encode text labels -------------------------------------------
            text = correction["corrected_text"]
            tokenizer = processor.tokenizer
            encoded = tokenizer(
                text,
                padding=False,
                truncation=True,
                max_length=256,
                return_tensors="pt",
            )
            label_ids = encoded.input_ids.squeeze(0).tolist()
            labels_list.append(label_ids)

        dataset = Dataset.from_dict(
            {
                "pixel_values": pixel_values_list,
                "labels": labels_list,
            }
        )
        # Set format to PyTorch so pixel_values come back as tensors.
        dataset.set_format(type="torch", columns=["pixel_values"], output_all_columns=True)
        return dataset

    # ----- training --------------------------------------------------------

    def train(
        self,
        user_id: int,
        corrections: list[dict],
        epochs: int = 3,
        lr: float = 5e-5,
    ) -> str:
        """Run LoRA fine-tuning for a user and return the path to saved weights.

        Parameters
        ----------
        user_id:
            Identifies the user whose adapter is being trained.
        corrections:
            List of correction dicts (see :meth:`prepare_training_data`).
        epochs:
            Number of training epochs.
        lr:
            Learning rate for the AdamW optimiser.

        Returns
        -------
        str
            Filesystem path where the LoRA adapter weights were saved.
        """
        if not corrections:
            raise ValueError("No corrections provided for training")

        logger.info(
            "Starting LoRA fine-tuning for user %d with %d corrections",
            user_id,
            len(corrections),
        )

        # 1. Load base model (fresh copy so we don't mutate the inference model).
        base_model = VisionEncoderDecoderModel.from_pretrained(self.base_model_name)

        # 2. If the user already has LoRA weights, load them as a starting point.
        existing_lora = self._latest_lora_path(user_id)
        if existing_lora is not None:
            logger.info("Resuming from existing LoRA weights at %s", existing_lora)
            model = PeftModel.from_pretrained(base_model, str(existing_lora))
            # Unfreeze LoRA parameters for continued training.
            for name, param in model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = True
        else:
            # 3. Apply fresh LoRA config.
            model = get_peft_model(base_model, DEFAULT_LORA_CONFIG)

        model.print_trainable_parameters()

        # 4. Prepare the dataset.
        dataset = self.prepare_training_data(corrections)

        # 5. Configure training.
        next_version = self._latest_version(user_id) + 1
        output_dir = self._user_dir(user_id) / f"v{next_version}"
        output_dir.mkdir(parents=True, exist_ok=True)

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=min(4, len(corrections)),
            learning_rate=lr,
            weight_decay=0.01,
            warmup_steps=max(1, len(corrections) // 4),
            logging_steps=1,
            save_strategy="epoch",
            predict_with_generate=False,
            fp16=False,  # safe for CPU
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none",
            average_tokens_across_devices=False,
        )

        collator = TrOCRDataCollator()

        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
            processing_class=self._get_processor().tokenizer,
        )

        # 6. Train.
        trainer.train()

        # 7. Save only the LoRA adapter weights (small).
        model.save_pretrained(str(output_dir))
        logger.info("LoRA weights saved to %s", output_dir)

        return str(output_dir)

    # ----- export ----------------------------------------------------------

    def export_model(self, user_id: int) -> str:
        """Package the user's latest LoRA weights into a zip file.

        Returns the path to the zip archive.
        """
        lora_path = self._latest_lora_path(user_id)
        if lora_path is None:
            raise FileNotFoundError(
                f"No LoRA weights found for user {user_id}"
            )

        zip_path = self._user_dir(user_id) / f"user_{user_id}_lora.zip"

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in lora_path.rglob("*"):
                if file.is_file():
                    arcname = file.relative_to(lora_path)
                    zf.write(file, arcname)

        logger.info("Exported LoRA weights to %s", zip_path)
        return str(zip_path)


# ---------------------------------------------------------------------------
# Active learning / uncertainty scoring
# ---------------------------------------------------------------------------


def score_uncertainty(confidence: float) -> float:
    """Convert a raw confidence value in ``[0, 1]`` to an uncertainty score.

    Higher values mean the model is *less* certain and the item is more
    valuable to correct.  Uses ``1 - confidence`` with a slight non-linear
    boost for very low confidence values (entropy-inspired).
    """
    # Clamp to avoid log(0).
    c = max(min(confidence, 1.0 - 1e-9), 1e-9)
    # Shannon entropy of a binary variable (confident vs. not).
    entropy = -(c * math.log2(c) + (1 - c) * math.log2(1 - c))
    # Normalise: max entropy is 1.0 (at c=0.5).  We also weight by raw
    # uncertainty so that very low confidence items always rank high.
    return 0.5 * (1 - c) + 0.5 * entropy


def get_priority_items(
    ocr_results: list[dict],
    n: int = 10,
) -> list[dict]:
    """Select the *n* OCR results most valuable for a user to correct.

    Each item in *ocr_results* should be a dict with at least:
    - ``confidence`` (float)
    - ``text`` (str)
    - ``id`` (int, optional) -- used to check if already corrected

    Selection criteria (in priority order):
    1. Lowest confidence (model is least certain).
    2. Diversity of text content (avoid showing many similar segments).
    3. Prefer items that have not been corrected yet.

    Returns up to *n* items sorted from highest to lowest priority.
    """
    if not ocr_results:
        return []

    # Score every item.
    scored: list[tuple[float, int, dict]] = []
    for idx, item in enumerate(ocr_results):
        conf = item.get("confidence", 0.5)
        uncertainty = score_uncertainty(conf)

        # Bonus for uncorrected items.
        already_corrected = item.get("corrected", False)
        correction_penalty = 0.3 if already_corrected else 0.0

        score = uncertainty - correction_penalty
        scored.append((score, idx, item))

    # Sort descending by score.
    scored.sort(key=lambda t: t[0], reverse=True)

    # Greedy diversity filter: skip items whose text is very similar to an
    # already-selected item (simple Jaccard on character bigrams).
    selected: list[dict] = []
    selected_bigrams: list[set[str]] = []

    for _score, _idx, item in scored:
        if len(selected) >= n:
            break

        text = item.get("text", "")
        bigrams = _char_bigrams(text)

        # Check similarity against already selected items.
        too_similar = False
        for prev_bigrams in selected_bigrams:
            sim = _jaccard(bigrams, prev_bigrams)
            if sim > 0.7:
                too_similar = True
                break

        if not too_similar:
            selected.append(item)
            selected_bigrams.append(bigrams)

    # If diversity filtering was too aggressive, fill up with remaining items.
    if len(selected) < n:
        selected_ids = {id(item) for item in selected}
        for _score, _idx, item in scored:
            if len(selected) >= n:
                break
            if id(item) not in selected_ids:
                selected.append(item)
                selected_ids.add(id(item))

    return selected


def _char_bigrams(text: str) -> set[str]:
    """Return the set of character bigrams in *text*."""
    text = text.lower().strip()
    if len(text) < 2:
        return {text}
    return {text[i : i + 2] for i in range(len(text) - 1)}


def _jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two sets."""
    if not a and not b:
        return 1.0
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union else 0.0
