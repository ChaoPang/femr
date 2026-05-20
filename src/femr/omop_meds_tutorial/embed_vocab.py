"""Pre-embed every entry in the MOTOR tokenizer's vocab with a medical
SentenceTransformer (default: NeuML/pubmedbert-base-embeddings) and save
the result as a (vocab_size, hidden_size) torch tensor.

Output tensor is suitable for `--reasoning_embedding_init_path` in
pretrain_motor.py, which warm-starts ReasoningLayer.reasoning_embedding
with clinically-grounded text representations instead of random init.

Text construction per vocab entry:
  * type=code: ontology.get_description(code) if available, else the code string.
  * type=numeric: synthetic phrase "numeric value between {start} and {end}".

The script expects the standard prepare_motor.py outputs in $PRETRAINING_DATA:
  * tokenizer/dictionary.msgpack
  * ontology.pkl  (must carry descriptions — i.e., built without
    prune_all_descriptions=True; see commit d9a6060)

Usage:
  python -m femr.omop_meds_tutorial.embed_vocab \\
      --pretraining_data $PRETRAINING_DATA \\
      --output reasoning_init.pt \\
      --model_name NeuML/pubmedbert-base-embeddings
"""
from __future__ import annotations

import argparse
import math
import pathlib
import pickle
import time
from typing import List, Optional

import msgpack
import numpy as np
import torch


def _build_descriptions(
    vocab: list,
    ontology,
    missing_fallback: str = "{code}",
) -> tuple[list[str], dict[str, int]]:
    """Return (texts, breakdown) where texts[i] is the description used for
    vocab entry i. `breakdown` counts how many entries got each source.
    """
    texts: List[str] = []
    counts = {"code_with_desc": 0, "code_fallback": 0, "numeric": 0, "other": 0}
    for entry in vocab:
        if entry["type"] == "code":
            code = entry["code_string"]
            desc = ontology.get_description(code) if ontology is not None else None
            if desc:
                texts.append(desc)
                counts["code_with_desc"] += 1
            else:
                texts.append(missing_fallback.format(code=code))
                counts["code_fallback"] += 1
        elif entry["type"] == "numeric":
            start = entry.get("val_start")
            end = entry.get("val_end")
            prop = entry.get("property", "value")
            # Render "[-inf, x)" cleanly.
            def fmt(x: Optional[float]) -> str:
                if x is None:
                    return "?"
                if math.isinf(x):
                    return "negative infinity" if x < 0 else "infinity"
                return f"{x:g}"
            texts.append(
                f"numeric {prop} between {fmt(start)} and {fmt(end)}"
            )
            counts["numeric"] += 1
        else:
            texts.append(str(entry))
            counts["other"] += 1
    return texts, counts


def main() -> None:
    p = argparse.ArgumentParser(
        description="Embed every MOTOR vocab entry with a SentenceTransformer."
    )
    p.add_argument("--pretraining_data", required=True,
                   help="Folder with tokenizer/dictionary.msgpack and ontology.pkl")
    p.add_argument("--output", required=True,
                   help="Path to save the (vocab_size, hidden_size) torch tensor")
    p.add_argument("--model_name", default="NeuML/pubmedbert-base-embeddings",
                   help="SentenceTransformer model to use")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Encoding batch size")
    p.add_argument("--device", default=None,
                   help="Device for the encoder ('cuda', 'cpu', or torch.device-compatible)")
    p.add_argument("--target_hidden_size", type=int, default=None,
                   help="If set, the script will assert the encoder output dim equals this "
                        "(useful sanity check vs MOTOR hidden_size).")
    args = p.parse_args()

    pretraining_data = pathlib.Path(args.pretraining_data)
    tok_path = pretraining_data / "tokenizer" / "dictionary.msgpack"
    ont_path = pretraining_data / "ontology.pkl"
    out_path = pathlib.Path(args.output)

    print(f"loading tokenizer vocab from {tok_path}")
    vocab = msgpack.unpackb(tok_path.read_bytes(), raw=False)["vocab"]
    print(f"  vocab_size = {len(vocab):,}")

    print(f"loading ontology from {ont_path}")
    with open(ont_path, "rb") as f:
        ontology = pickle.load(f)
    print(f"  ontology.description_map size = {len(ontology.description_map):,}")

    print("constructing descriptions for each vocab entry ...")
    texts, breakdown = _build_descriptions(vocab, ontology)
    print(f"  breakdown: {breakdown}")
    print(f"  example with description: '{texts[next(i for i,e in enumerate(vocab) if e['type']=='code')]}'")
    print(f"  example numeric:           '{texts[next(i for i,e in enumerate(vocab) if e['type']=='numeric')]}'")

    print(f"loading SentenceTransformer({args.model_name!r})")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model_name, device=args.device)
    emb_dim = model.get_sentence_embedding_dimension()
    print(f"  encoder output dim = {emb_dim}")
    if args.target_hidden_size is not None and emb_dim != args.target_hidden_size:
        raise ValueError(
            f"Encoder emits {emb_dim}-d embeddings but --target_hidden_size={args.target_hidden_size}."
            " They must match for direct use in ReasoningLayer.reasoning_embedding."
        )

    print(f"encoding {len(texts):,} descriptions  batch_size={args.batch_size} ...")
    t0 = time.time()
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=args.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    print(f"  encoded in {time.time()-t0:.1f}s; shape={embeddings.shape}  dtype={embeddings.dtype}")

    tensor = torch.from_numpy(embeddings).to(dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(tensor, out_path)
    print(f"wrote {out_path}  ({out_path.stat().st_size/1e6:.1f} MB)")
    print("Pass this path to pretrain_motor.py via --reasoning_embedding_init_path.")


if __name__ == "__main__":
    main()
