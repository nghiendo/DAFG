import json
import os


DATASET_DIR = "dataset"
DATASETS = (
    "Laptops_corenlp",
    "Restaurants_corenlp",
    "Tweets_corenlp",
    "MAMS_corenlp",
)
SPLITS = ("train", "test", "valid")


def map_polarity(polarity_value):
    """Map polarity labels to the raw ABSA convention: -1, 0, 1."""
    polarity_str = str(polarity_value).lower().strip()
    if polarity_str in {"1", "+1", "positive"} or "positive" in polarity_str:
        return "1"
    if polarity_str in {"-1", "negative"} or "negative" in polarity_str:
        return "-1"
    return "0"


def build_raw_sentence(tokens, aspect):
    """
    Convert one JSON sample into the classic 3-line raw format sentence.

    The repo expects the aspect term in the sentence to be replaced by a token
    containing '$', typically '$T$'.
    """
    start_idx = aspect.get("from")
    end_idx = aspect.get("to")
    if start_idx is None or end_idx is None:
        return ""
    if not isinstance(start_idx, int) or not isinstance(end_idx, int):
        return ""
    if start_idx < 0 or end_idx > len(tokens) or start_idx >= end_idx:
        return ""

    raw_tokens = tokens[:start_idx] + ["$T$"] + tokens[end_idx:]
    return " ".join(raw_tokens).strip()


def build_aspect_term(tokens, aspect):
    term_tokens = aspect.get("term", []) or []
    term = " ".join(term_tokens).strip()
    if term:
        return term

    start_idx = aspect.get("from")
    end_idx = aspect.get("to")
    if isinstance(start_idx, int) and isinstance(end_idx, int):
        return " ".join(tokens[start_idx:end_idx]).strip()
    return ""


def convert_complex_json_to_raw(json_path, raw_path):
    if not os.path.exists(json_path):
        print(f"[skip] Missing JSON: {json_path}")
        return 0

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dest_dir = os.path.dirname(raw_path)
    if dest_dir:
        os.makedirs(dest_dir, exist_ok=True)

    count = 0
    with open(raw_path, "w", encoding="utf-8", newline="\n") as f_out:
        for entry in data:
            tokens = entry.get("token", []) or []
            if not tokens:
                continue

            for aspect in entry.get("aspects", []) or []:
                sentence = build_raw_sentence(tokens, aspect)
                term = build_aspect_term(tokens, aspect)
                polarity = map_polarity(aspect.get("polarity", "neutral"))

                if not sentence or not term:
                    continue

                f_out.write(f"{sentence}\n")
                f_out.write(f"{term}\n")
                f_out.write(f"{polarity}\n")
                count += 1

    print(f"[ok] {json_path} -> {raw_path} ({count} samples)")
    return count


def convert_all_datasets(base_dir=DATASET_DIR):
    total = 0
    for dataset_name in DATASETS:
        dataset_path = os.path.join(base_dir, dataset_name)
        for split in SPLITS:
            json_path = os.path.join(dataset_path, f"{split}.json")
            raw_path = os.path.join(dataset_path, f"{split}.raw")
            total += convert_complex_json_to_raw(json_path, raw_path)
    print(f"[done] Generated {total} raw samples in total.")


if __name__ == "__main__":
    convert_all_datasets()
