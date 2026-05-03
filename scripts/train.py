"""Fine-tune TexTeller on our (image, latex) manifest.

Designed for a single A100/H100 on RunPod:
  - bf16 mixed-precision (set via --bf16)
  - SDPA / FlashAttention auto-enabled by transformers when available
  - HF Trainer with gradient checkpointing for low memory
  - Reads data/rendered/manifest.jsonl by default; can pull from an HF dataset instead

Local (M4) smoke test:
  python scripts/train.py --max-steps 5 --per-device-batch-size 1 --no-bf16

RunPod 30-min fine-tune (A100 80GB):
  python scripts/train.py --max-steps 800 --per-device-batch-size 8 \
      --grad-accum 2 --bf16 --lr 5e-5 --out runs/textteller-ft
"""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import (AutoTokenizer, Trainer, TrainingArguments,
                          VisionEncoderDecoderModel, set_seed)

from latextract.model.inference import _preprocess_vit

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")
ROOT = Path(__file__).resolve().parents[1]


@dataclass
class Sample:
    image_path: Path
    target_text: str


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: Path, image_size: int, num_channels: int,
                 tokenizer, max_target_length: int = 384):
        self.image_size = image_size
        self.num_channels = num_channels
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        self.samples: list[Sample] = []
        for line in manifest_path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            target = d.get("body") or d.get("full_latex")
            self.samples.append(Sample(image_path=ROOT / d["image"], target_text=target))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        pv = _preprocess_vit(img, self.image_size, self.num_channels).squeeze(0)
        tok = self.tokenizer(
            s.target_text, max_length=self.max_target_length,
            truncation=True, padding="max_length", return_tensors="pt",
        )
        labels = tok.input_ids.squeeze(0)
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        return {"pixel_values": pv, "labels": labels}


def main(
    model_id: str = "OleehyO/TexTeller",
    manifest: str = "data/rendered/manifest.jsonl",
    out: str = "runs/ft",
    max_steps: int = 5,
    per_device_batch_size: int = 1,
    grad_accum: int = 1,
    lr: float = 5e-5,
    bf16: bool = False,
    gradient_checkpointing: bool = True,
    seed: int = 42,
    warmup_ratio: float = 0.05,
):
    set_seed(seed)
    out_dir = ROOT / out
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("loading %s", model_id)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = VisionEncoderDecoderModel.from_pretrained(model_id)
    # VisionEncoderDecoderConfig stores these on .generation_config or under .decoder
    gen_cfg = model.generation_config
    if gen_cfg.decoder_start_token_id is None:
        gen_cfg.decoder_start_token_id = tok.bos_token_id or tok.cls_token_id or 0
    if gen_cfg.pad_token_id is None:
        gen_cfg.pad_token_id = tok.pad_token_id or 0
    # Also mirror onto config (Trainer reads this for shifting decoder inputs)
    model.config.decoder_start_token_id = gen_cfg.decoder_start_token_id
    model.config.pad_token_id = gen_cfg.pad_token_id
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    enc = model.config.encoder
    sz = getattr(enc, "image_size", 448)
    image_size = int(sz[0] if isinstance(sz, (list, tuple)) else sz)
    num_channels = int(getattr(enc, "num_channels", 3))

    ds = ManifestDataset(ROOT / manifest, image_size, num_channels, tok)
    log.info("dataset: %d samples", len(ds))
    if len(ds) == 0:
        raise RuntimeError("empty manifest; run scripts/build_dataset.py first")

    args = TrainingArguments(
        output_dir=str(out_dir),
        max_steps=max_steps,
        per_device_train_batch_size=per_device_batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        bf16=bf16,
        logging_steps=10,
        save_steps=max(50, max_steps),
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    def collate(batch):
        return {
            "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      processing_class=tok, data_collator=collate)
    log.info("training: max_steps=%d, batch=%d, grad_accum=%d, bf16=%s",
             max_steps, per_device_batch_size, grad_accum, bf16)
    trainer.train()
    trainer.save_model(str(out_dir / "final"))
    tok.save_pretrained(str(out_dir / "final"))
    log.info("saved to %s", out_dir / "final")


if __name__ == "__main__":
    import typer
    typer.run(main)
