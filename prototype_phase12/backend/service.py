from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "artifacts"


def classify_probability(probability: float, threshold: float) -> bool:
    """Apply the locked q-star threshold used by the prototype."""
    return float(probability) >= float(threshold)


@dataclass(frozen=True)
class FactorContribution:
    feature: str
    label: str
    value: Any
    contribution: float
    direction: str
    direction_label: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "label": self.label,
            "value": _json_safe(self.value),
            "contribution": self.contribution,
            "abs_contribution": abs(self.contribution),
            "direction": self.direction,
            "direction_label": self.direction_label,
        }


class FraudScoringBackend:
    """Load the chosen tree model once and score/explain prototype claims."""

    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> None:
        self.artifact_dir = Path(artifact_dir)
        self.config = _read_json(self.artifact_dir / "model_config.json")
        self.label_map = _read_json(self.artifact_dir / "label_map.json")["labels"]
        self.approved_features = _read_json(
            self.artifact_dir / "approved_features.json"
        )["approved_features"]
        self.threshold = float(self.config["q_star_threshold"])
        self.model_family = str(self.config.get("model_family", "")).lower()
        self.categorical_features = list(self.config.get("categorical_features", []))

        self.claim_pool = self._load_claim_pool()
        self.category_map = self._load_category_map()
        self.models = self._load_models()

    def list_claims(self) -> list[dict[str, Any]]:
        ui_path = self.artifact_dir / self.config["claim_pool_ui_path"]
        if not ui_path.exists():
            return []
        return _read_json(ui_path)["claims"]

    def score_claim(
        self,
        claim: str | int | Mapping[str, Any] | pd.Series | pd.DataFrame,
        *,
        top_n: int = 6,
    ) -> dict[str, Any]:
        frame = self._claim_to_frame(claim)
        probability = float(self._predict_proba(frame)[0])
        flagged = classify_probability(probability, self.threshold)
        top_factors = [
            factor.as_dict() for factor in self._explain(frame, top_n=top_n)
        ]
        claim_id = (
            str(frame["claim_id"].iloc[0])
            if "claim_id" in frame.columns
            else None
        )
        return {
            "claim_id": claim_id,
            "fraud_probability": probability,
            "threshold": self.threshold,
            "flagged": flagged,
            "classification_label": "Flagged for review" if flagged else "Not flagged",
            "top_factors": top_factors,
        }

    def _load_claim_pool(self) -> pd.DataFrame:
        pool_path = self.artifact_dir / self.config["claim_pool_path"]
        if not pool_path.exists():
            return pd.DataFrame()
        return pd.read_parquet(pool_path)

    def _load_models(self) -> list[Any]:
        model_paths = [self.artifact_dir / p for p in self.config.get("model_paths", [])]
        missing = [str(p) for p in model_paths if not p.exists()]
        if not model_paths:
            raise FileNotFoundError(
                "No fitted model files are listed in artifacts/model_config.json. "
                "Add the chosen-track fitted model artifact first."
            )
        if missing:
            raise FileNotFoundError(f"Configured model files are missing: {missing}")
        if self.model_family == "xgboost":
            try:
                import xgboost as xgb
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Install prototype_phase12/requirements.txt before loading XGBoost."
                ) from exc
            models = []
            for path in model_paths:
                model = xgb.XGBClassifier()
                model.load_model(str(path))
                models.append(model)
            return models
        if self.model_family == "lightgbm":
            try:
                import lightgbm as lgb
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Install prototype_phase12/requirements.txt before loading LightGBM."
                ) from exc
            return [lgb.Booster(model_file=str(path)) for path in model_paths]
        raise ValueError(
            "artifacts/model_config.json must set model_family to 'xgboost' or 'lightgbm'."
        )

    def _load_category_map(self) -> dict[str, list[Any]]:
        category_map_path = self.artifact_dir / str(
            self.config.get("category_map_path", "category_map.json")
        )
        if category_map_path.exists():
            payload = _read_json(category_map_path)
            categories = payload.get("categories", {})
            if isinstance(categories, dict):
                return {str(key): value for key, value in categories.items()}
        try:
            first_model = self.artifact_dir / self.config["model_paths"][0]
        except (KeyError, IndexError):
            return {}
        if self.model_family == "lightgbm":
            categories = _read_lightgbm_pandas_categories(first_model)
            if len(categories) == len(self.categorical_features):
                return dict(zip(self.categorical_features, categories))
        return {}

    def _claim_to_frame(
        self,
        claim: str | int | Mapping[str, Any] | pd.Series | pd.DataFrame,
    ) -> pd.DataFrame:
        if isinstance(claim, pd.DataFrame):
            frame = claim.copy()
        elif isinstance(claim, pd.Series):
            frame = claim.to_frame().T
        elif isinstance(claim, Mapping):
            frame = pd.DataFrame([dict(claim)])
        else:
            if self.claim_pool.empty:
                raise ValueError("No claim pool has been prepared yet.")
            claim_id = str(claim)
            matches = self.claim_pool[self.claim_pool["claim_id"].astype(str) == claim_id]
            if matches.empty:
                raise KeyError(f"Claim not found in pool: {claim_id}")
            frame = matches.head(1).copy()

        missing = [f for f in self.approved_features if f not in frame.columns]
        if missing:
            raise ValueError(f"Claim is missing model features: {missing[:10]}")

        metadata_cols = [c for c in ["claim_id", "row_index"] if c in frame.columns]
        ordered = frame[metadata_cols + self.approved_features].copy()
        return self._prepare_feature_frame(ordered)

    def _prepare_feature_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        for feature in self.approved_features:
            if feature in self.categorical_features:
                categories = self.category_map.get(feature)
                if categories:
                    values = frame[feature]
                    sample = next((v for v in categories if v is not None), None)
                    if isinstance(sample, (int, float)) and not isinstance(sample, bool):
                        values = pd.to_numeric(values, errors="coerce")
                    frame[feature] = pd.Categorical(values, categories=categories)
                else:
                    frame[feature] = frame[feature].astype("category")
            else:
                frame[feature] = pd.to_numeric(frame[feature], errors="coerce").astype(
                    "float32"
                )
        return frame

    def _feature_matrix(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame[self.approved_features]

    def _predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        matrix = self._feature_matrix(frame)
        if self.model_family == "xgboost":
            member_scores = [
                np.asarray(model.predict_proba(matrix)[:, 1], dtype=np.float64)
                for model in self.models
            ]
        else:
            member_scores = [
                np.asarray(model.predict(matrix), dtype=np.float64)
                for model in self.models
            ]
        return np.mean(np.vstack(member_scores), axis=0)

    def _explain(self, frame: pd.DataFrame, *, top_n: int) -> list[FactorContribution]:
        matrix = self._feature_matrix(frame)
        if self.model_family == "xgboost":
            try:
                import xgboost as xgb
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Install prototype_phase12/requirements.txt before explaining XGBoost."
                ) from exc
            dmatrix = xgb.DMatrix(matrix, enable_categorical=True)
            member_contribs = [
                np.asarray(
                    model.get_booster().predict(dmatrix, pred_contribs=True),
                    dtype=np.float64,
                )
                for model in self.models
            ]
        else:
            member_contribs = [
                np.asarray(model.predict(matrix, pred_contrib=True), dtype=np.float64)
                for model in self.models
            ]
        mean_contrib = np.mean(np.stack(member_contribs), axis=0)[0, :-1]
        top_indices = np.argsort(np.abs(mean_contrib))[::-1][:top_n]
        row = frame.iloc[0]
        factors: list[FactorContribution] = []
        for idx in top_indices:
            feature = self.approved_features[int(idx)]
            contribution = float(mean_contrib[int(idx)])
            toward_fraud = contribution > 0
            factors.append(
                FactorContribution(
                    feature=feature,
                    label=self.label_map.get(feature, _humanize_feature(feature)),
                    value=row[feature],
                    contribution=contribution,
                    direction="toward_fraud" if toward_fraud else "toward_non_fraud",
                    direction_label=(
                        "Pushes toward fraud"
                        if toward_fraud
                        else "Pushes toward non-fraud"
                    ),
                )
            )
        return factors


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_lightgbm_pandas_categories(model_path: Path) -> list[list[Any]]:
    marker = "pandas_categorical:"
    for line in reversed(model_path.read_text(encoding="utf-8", errors="ignore").splitlines()):
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    return []


def _humanize_feature(feature: str) -> str:
    return feature.replace("_", " ").title()


def _json_safe(value: Any) -> Any:
    if pd.isna(value):
        return None
    if isinstance(value, np.generic):
        return value.item()
    return value
