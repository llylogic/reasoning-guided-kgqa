"""Backends are loaded lazily. Offline evaluation uses local checkpoints."""

import json
import re
from collections import Counter
from math import sqrt

ANSWER = "\nAnswer:"


def planning(q):
    return (
        "Write a numbered KG retrieval plan for the question. Refer to earlier results as step 1, step 2, etc. "
        "Use entity or step | relation for each hop. Preserve all constraints.\nQuestion: "
        + q
        + "\nPlan:"
    )


def answering(q, sgs):
    return (
        "Answer from the graph evidence. Give a retrieval plan followed by Answer: and a JSON array of answer labels."
        "\nQuestion: " + q + "\nEvidence:\n" + sgs + "\nPlan:"
    )


def rewrite(q):
    return (
        "Simplify this question into numbered KG retrieval steps. Use entity or step | relation. "
        "Preserve the question's constraints.\nQuestion: " + q + "\nPlan:"
    )


def normalize(s):
    return " ".join(re.findall(r"\w+", s.casefold().replace("_", " ")))


class Lexical:
    """Rank the demo's candidate triples by cosine similarity of word counts."""

    def scores(self, q, texts):
        a = Counter(normalize(q).split())

        def score(t):
            b = Counter(normalize(t).split())
            return sum(v * b[k] for k, v in a.items()) / (
                sqrt(sum(v * v for v in a.values()) * sum(v * v for v in b.values()))
                or 1
            )

        return [score(t) for t in texts]


class BGE:
    def __init__(self, path, device="cpu"):
        from sentence_transformers import SentenceTransformer

        self.encoder = SentenceTransformer(path, device=device, local_files_only=True)
        self.encoder.eval()
        self.last_texts = self.last_vectors = None

    def scores(self, q, texts):
        if not texts:
            return []
        qv = self.encoder.encode(
            ["Represent this sentence for searching relevant passages: " + q],
            normalize_embeddings=True,
        )
        if texts != self.last_texts:
            self.last_vectors = self.encoder.encode(
                texts, normalize_embeddings=True, batch_size=64
            )
            self.last_texts = list(texts)
        return (self.last_vectors @ qv[0]).tolist()


class LocalLM:
    def __init__(self, path, device, max_new=256, trust_local_code=False):
        import torch
        from pathlib import Path
        from transformers import AutoTokenizer, AutoModelForCausalLM

        if not Path(path).is_dir():
            raise FileNotFoundError(f"Missing local model directory: {path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            path, local_files_only=True, trust_remote_code=trust_local_code
        )
        kwargs = {"device_map": "auto"} if device == "auto" else {}
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            local_files_only=True,
            trust_remote_code=trust_local_code,
            torch_dtype="auto" if device != "cpu" else torch.float32,
            **kwargs,
        )
        if device != "auto":
            self.model.to(device)
        self.model.eval()
        self.device = self.model.get_input_embeddings().weight.device
        self.max_new = max_new

    def generate_teacher(self, prompt):
        if self.tokenizer.chat_template:
            formatted = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted = prompt
        result = self.generate(
            formatted, sample=True, formatted_chat=bool(self.tokenizer.chat_template)
        )
        return result.rsplit("</think>", 1)[-1].strip()

    def count(self, s):
        return len(self.tokenizer.encode(s, add_special_tokens=False))

    def generate(self, prompt, stop=False, sample=False, formatted_chat=False):
        import torch
        from transformers import StoppingCriteria, StoppingCriteriaList

        tok = self.tokenizer
        ids = tok(
            prompt, return_tensors="pt", add_special_tokens=not formatted_chat
        ).to(self.device)
        start = ids.input_ids.shape[1]
        limit = getattr(self.model.config, "max_position_embeddings", 4096)
        budget = min(limit - start, self.max_new)
        if budget <= 0:
            raise ValueError("Prompt exceeds checkpoint context window")

        class AnswerStarted(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                return ANSWER in tok.decode(
                    input_ids[0, start:], skip_special_tokens=True
                )

        args = {"temperature": 0.7, "top_p": 0.9} if sample else {}
        with torch.inference_mode():
            out = self.model.generate(
                **ids,
                max_new_tokens=budget,
                do_sample=sample,
                pad_token_id=tok.eos_token_id,
                stopping_criteria=StoppingCriteriaList([AnswerStarted()])
                if stop
                else None,
                **args,
            )
        result = tok.decode(out[0, start:], skip_special_tokens=True)
        return result.partition(ANSWER)[0] if stop else result


def answers(text):
    value = text.rsplit("Answer:", 1)[-1].strip()
    try:
        items = json.loads(value)
    except ValueError:
        return [s.strip() for s in value.splitlines() if s.strip()]
    return (
        list(dict.fromkeys(str(x) for x in items))
        if isinstance(items, list)
        else [str(items)]
    )
