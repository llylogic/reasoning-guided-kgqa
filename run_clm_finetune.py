"""Train one LoRA curriculum stage with Accelerate and AdamW."""

import argparse
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from data_utils import ROOT, read_json, records, resolved
from prompter import Prompter
from rgr.lora import attach


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--paths", default=ROOT / "paths.json")
    p.add_argument(
        "--train-file",
        "--train_file",
        help="optional prepared JSONL or legacy JSON training file",
    )
    p.add_argument("--config", default=ROOT / "config.json")
    opt = p.parse_args()
    import torch
    from accelerate import Accelerator
    from accelerate.utils import set_seed
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        get_scheduler,
    )
    from torch.utils.data import DataLoader
    from rgr.training_data import Encoded, training_row

    cfg = read_json(opt.config)
    paths = resolved(opt.paths)
    ix = opt.stage - 1
    work = Path(paths["output_dir"]) / opt.dataset
    data = opt.train_file or work / f"stage{opt.stage}.jsonl"
    base = paths["backbone"] if ix == 0 else work / f"merged{ix}"
    out = work / f"adapter{opt.stage}"
    if out.exists():
        raise FileExistsError(f"Choose a fresh output or move {out} before rerunning")
    accel = Accelerator(
        gradient_accumulation_steps=cfg["accumulation"],
        step_scheduler_with_optimizer=False,
    )
    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    precision = accel.mixed_precision
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(
        precision, torch.float32
    )
    model = AutoModelForCausalLM.from_pretrained(
        base, local_files_only=True, torch_dtype=dtype
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    model = attach(
        model,
        SimpleNamespace(
            lora_target_modules=cfg["targets"],
            lora_r=cfg["rank"],
            lora_alpha=cfg["alpha"],
            lora_dropout=cfg["dropout"],
        ),
    )
    samples = [training_row(r, i, opt.dataset) for i, r in enumerate(records(data))]
    if not samples or any(r["dataset"] != opt.dataset for r in samples):
        raise ValueError("Empty or mismatched stage data")
    ds = Encoded(
        samples,
        tokenizer,
        Prompter(),
        min(cfg["max_train_tokens"], model.config.max_position_embeddings),
    )
    collator = DataCollatorForSeq2Seq(
        tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
    )
    loader = DataLoader(
        ds, shuffle=True, collate_fn=collator, batch_size=cfg["batch_sizes"][ix]
    )
    base_optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg["learning_rates"][ix],
        weight_decay=0.0,
    )
    model, optimizer, loader = accel.prepare(model, base_optimizer, loader)
    updates = math.ceil(len(loader) / cfg["accumulation"]) * cfg["epochs"][ix]
    scheduler = get_scheduler(
        "linear",
        optimizer=base_optimizer,
        num_warmup_steps=0,
        num_training_steps=updates,
    )
    for epoch in range(cfg["epochs"][ix]):
        model.train()
        for batch in loader:
            with accel.accumulate(model):
                loss = model(**batch).loss
                accel.backward(loss)
                if accel.sync_gradients:
                    accel.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if accel.sync_gradients and not accel.optimizer_step_was_skipped:
                    scheduler.step()
                optimizer.zero_grad()
        accel.print(
            f"stage={ix + 1} epoch={epoch + 1} last_batch_loss={float(loss.detach()):.5g} lr={scheduler.get_last_lr()[0]:.5g}"
        )
    accel.wait_for_everyone()
    state = accel.get_state_dict(model)
    if accel.is_main_process:
        out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=out.parent) as tmp:
            checkpoint = Path(tmp) / "adapter"
            accel.unwrap_model(model).save_pretrained(
                checkpoint, state_dict=state, safe_serialization=True
            )
            tokenizer.save_pretrained(checkpoint)
            checkpoint.rename(out)
    accel.wait_for_everyone()


if __name__ == "__main__":
    main()
