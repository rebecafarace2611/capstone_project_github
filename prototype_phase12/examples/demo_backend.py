from __future__ import annotations

import argparse
import json
from pathlib import Path

from prototype_phase12.backend import FraudScoringBackend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the prototype backend.")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "artifacts",
    )
    parser.add_argument("--claim-id", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = FraudScoringBackend(args.artifact_dir)
    claim_id = args.claim_id
    if claim_id is None:
        claims = backend.list_claims()
        if not claims:
            raise SystemExit("No claim pool UI file found. Run asset preparation first.")
        claim_id = claims[0]["claim_id"]
    result = backend.score_claim(claim_id)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
