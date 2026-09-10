from __future__ import annotations

import argparse
import json

from src.evaluation import evaluate_records, load_prediction_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate SatQueryX VQA/change-VQA predictions.")
    parser.add_argument("predictions", help="JSONL with prediction, reference and optional task fields")
    args = parser.parse_args()
    metrics = evaluate_records(load_prediction_jsonl(args.predictions))
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
