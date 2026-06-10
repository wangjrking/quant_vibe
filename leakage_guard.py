"""Feature leakage guard for model training.

The model may keep future-return columns in the processed dataset for labels
and backtests, but those columns must never enter the feature matrix.
"""

import re


_ALLOWED_RETURN_FEATURES = {"pre_yield_rate"}


def _is_leaky_feature(name, label=None):
    feature = str(name)
    lowered = feature.lower()

    if label is not None and feature == label:
        return True
    if lowered.startswith("post"):
        return True
    if "post_" in lowered or "_post" in lowered:
        return True
    if lowered == "tag" or lowered.endswith("_tag"):
        return True
    if "yield_rate" in lowered and lowered not in _ALLOWED_RETURN_FEATURES:
        return True
    if re.search(r"(future|forward|lead)", lowered):
        return True
    return False


def find_leaky_features(features, label=None):
    """Return future-looking feature names in their original order."""
    return [feature for feature in features if _is_leaky_feature(feature, label=label)]


def validate_no_leakage(features, label=None):
    """Raise ValueError when future-looking features are present."""
    leaky_features = find_leaky_features(features, label=label)
    if leaky_features:
        preview = ", ".join(map(str, leaky_features[:10]))
        if len(leaky_features) > 10:
            preview += f", ... (+{len(leaky_features) - 10} more)"
        raise ValueError(f"Future-looking features are not allowed in training: {preview}")
    return True
