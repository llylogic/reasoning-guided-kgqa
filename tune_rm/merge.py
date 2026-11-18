"""Path-parameterized version of tune_rm/merge_first.py."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory


def merge_final_model(base_model_path, adapter_path, final_output_path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if Path(final_output_path).exists():
        raise FileExistsError(final_output_path)
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype="auto", local_files_only=True
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token

    model = PeftModel.from_pretrained(model, adapter_path, local_files_only=True)
    model = model.merge_and_unload()

    out = Path(final_output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out.parent) as tmp:
        checkpoint = Path(tmp) / "model"
        model.save_pretrained(checkpoint)
        tokenizer.save_pretrained(checkpoint)
        checkpoint.rename(out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("adapter")
    p.add_argument("output")
    args = p.parse_args()
    merge_final_model(args.base, args.adapter, args.output)
