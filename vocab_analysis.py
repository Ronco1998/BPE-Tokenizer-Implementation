import argparse
import os
import sys
from typing import Dict, List
import contextlib

class _Tee:
    """Simple stream-tee: duplicate writes to several streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()

# Ensure all tokenizer classes that might appear in pickled files are imported
# so that pickle can reconstruct them. Importing here has no runtime cost but
# guarantees compatibility with BPETokenizer saved via ner_bpe_tokenizer.py.
try:
    # The import is optional — if the module is absent we simply skip it.
    import ner_bpe_tokenizer  # noqa: F401
except ModuleNotFoundError:
    pass

from base_tokenizer import BaseTokenizer


# ────────────────────────────────────────────────────────────────────────────
# Helper functions
# ────────────────────────────────────────────────────────────────────────────

def print_vocabulary(token_to_id: Dict[str, int]) -> None:
    """Pretty-print the vocabulary sorted by token id."""
    print("\n================ VOCABULARY ================")
    for token, tid in sorted(token_to_id.items(), key=lambda kv: kv[1]):
        print(f"{tid:6d} : {repr(token)}")
    print("===========================================\n")


def analyse_token_types(token_to_id: Dict[str, int], special_tokens: Dict[str, int]) -> None:
    """Analyse how many tokens are chars, words, or bigrams and show examples."""
    stats = {
        "special": 0,
        "char": 0,
        "word": 0,
        "bigram": 0,
    }
    examples: Dict[str, List[str]] = {k: [] for k in stats}

    for token in token_to_id.keys():
        if token in special_tokens:
            stats["special"] += 1
            if len(examples["special"]) < 5:
                examples["special"].append(repr(token))
            continue

        words = token.strip().split()
        if len(token) == 1:
            category = "char"
        elif len(words) == 1:
            category = "word"
        elif len(words) == 2:
            category = "bigram"
        else:
            # Ignore longer n-grams for this simple analysis
            continue

        stats[category] += 1
        if len(examples[category]) < 10:
            examples[category].append(repr(token))

    total = len(token_to_id)

    print("================ TOKEN TYPE ANALYSIS ================")
    for cat in ["special", "char", "word", "bigram"]:
        cnt = stats[cat]
        pct = 100.0 * cnt / total if total else 0.0
        print(f"{cat.capitalize():<8}: {cnt:6d} • {pct:5.1f}%")
        if examples[cat]:
            print(f"    e.g. {', '.join(examples[cat])}")
    print("====================================================\n")


# ────────────────────────────────────────────────────────────────────────────
# Main CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Display vocabulary and simple token-type analysis for a pickled tokenizer",
    )
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Path to the pickled tokenizer (.pkl) file",
    )
    parser.add_argument(
        "--output",
        help="Optional path for the analysis text file (UTF-8). Defaults to <tokenizer>_analysis.txt in the same directory.",
    )

    args = parser.parse_args()

    # Derive output path if not provided
    if args.output:
        out_path = args.output
    else:
        base = os.path.splitext(os.path.basename(args.tokenizer))[0]
        out_path = os.path.join(os.path.dirname(args.tokenizer), f"{base}_analysis.txt")

    # Load the tokenizer (BaseTokenizer will unpickle the exact subclass)
    tokenizer: BaseTokenizer = BaseTokenizer.load(args.tokenizer)

    # Emit to console and file simultaneously
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        tee = _Tee(sys.stdout, fh)
        with contextlib.redirect_stdout(tee):
            print_vocabulary(tokenizer.token_to_id)
            analyse_token_types(tokenizer.token_to_id, tokenizer.special_tokens)

    print(f"Analysis written to {out_path}\n")


if __name__ == "__main__":
    main() 