"""Semantic versioning for registered models (stored as the "semver" tag).

Registry versions are plain integers, so we keep a human-meaningful label:
  MAJOR - the input schema (feature list) changed
  MINOR - a different algorithm than the previous model
  PATCH - same algorithm and schema, i.e. a retrain
"""

import hashlib

INITIAL_SEMVER = "0.1.0"


def schema_hash(features: list[str]) -> str:
    """Short, order-sensitive fingerprint of the feature list."""
    return hashlib.sha256(",".join(features).encode()).hexdigest()[:12]


def bump_semver(previous: dict | None, model_type: str, features_hash: str) -> str:
    """Next semver given the previous version's tags.

    `previous` holds the tags "semver", "model_type" and "schema_hash" of the
    last registered version, or is None/empty for the very first model.
    """
    if not previous or "semver" not in previous:
        return INITIAL_SEMVER
    major, minor, patch = (int(p) for p in previous["semver"].split("."))
    if previous.get("schema_hash") != features_hash:
        return f"{major + 1}.0.0"
    if previous.get("model_type") != model_type:
        return f"{major}.{minor + 1}.0"
    return f"{major}.{minor}.{patch + 1}"
