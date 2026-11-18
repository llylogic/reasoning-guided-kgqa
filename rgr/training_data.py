class Encoded:
    def __init__(self, samples, tokenizer, prompter, limit):
        self.samples, self.tokenizer, self.prompter, self.limit = (
            samples,
            tokenizer,
            prompter,
            limit,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        row = self.samples[i]
        prompt = self.prompter.generate_prompt(row["prompt"])
        # Tokenize the two parts independently: the boundary is identical at inference.
        prefix = self.tokenizer.encode(prompt, add_special_tokens=True)
        target = self.tokenizer.encode(row["target"], add_special_tokens=False) + [
            self.tokenizer.eos_token_id
        ]
        ids = prefix + target
        if len(ids) > self.limit:
            raise ValueError(
                f"Overlength training row {row['id']}/{row['task']}: {len(ids)} > {self.limit}. Reduce evidence budget or filter explicitly."
            )
        return dict(
            input_ids=ids,
            attention_mask=[1] * len(ids),
            labels=[-100] * len(prefix) + target,
        )


def training_row(row, index, dataset):
    """Adapt old instruction/input-output and KG-pretraining/text exports."""
    if "dataset" in row and row["dataset"] != dataset:
        raise ValueError("Training file contains a different benchmark")
    if row.get("split", "train") != "train":
        raise ValueError("Training requires split=train")
    if "prompt" in row and "target" in row:
        prompt, target = row["prompt"], row["target"]
    elif "input" in row and "output" in row:
        prompt, target = row["input"], row["output"]
    elif "text" in row:
        prompt, target = "", row["text"]
    else:
        raise ValueError("Expected prompt/target, input/output, or text")
    if not isinstance(prompt, str) or not isinstance(target, str) or not target.strip():
        raise ValueError(f"Invalid training text at record {index}")
    return dict(
        row,
        dataset=dataset,
        id=row.get("id", str(index)),
        task=row.get("task", "imported"),
        prompt=prompt,
        target=target,
    )
