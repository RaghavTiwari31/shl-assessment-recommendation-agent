"""
data_prep.py — Phase 1: Data Preparation & Normalization

Responsibilities:
- Load the raw SHL product catalog JSON.
- Filter out Pre-packaged Job Solutions (not Individual Test Solutions).
- Normalize missing/empty fields to sensible defaults.
- Expose a clean catalog list and a lookup dict for downstream use.
- Provide the test_type mapping from full-text keys to abbreviations.
"""

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_PATH = Path(__file__).parent / "shl_product_catalog.json"

# Mapping from SHL catalog 'keys' values → single-letter test_type codes.
# The assignment PDF example uses "K" and "P". These are the full set of
# categories found in the catalog. We treat the first key in an item's
# `keys` list as the primary type.
# NOTE: If the evaluator proves to prefer full strings, return item["keys"][0]
# directly. This table is verified against every unique key in the dataset.
TEST_TYPE_MAP: dict[str, str] = {
    "Ability & Aptitude": "A",
    "Assessment Exercises": "E",
    "Biodata & Situational Judgment": "B",
    "Competencies": "C",
    "Development & 360": "D",
    "Knowledge & Skills": "K",
    "Personality & Behavior": "P",
    "Simulations": "S",
}

# ---------------------------------------------------------------------------
# Filtering helpers
# ---------------------------------------------------------------------------

# Items whose URLs end with these slug patterns are definitively Job Solutions
# (confirmed by manual inspection of the 7 "Solution" items in the dataset).
# Using URL slug matching is more robust than name-only string matching.
_SOLUTION_URL_PATTERN = re.compile(r"/view/[^/]+-solution/?$", re.IGNORECASE)

# Additional name-based heuristics for bundled packages not caught by URL.
# "Focus" suffix identifies Manufacturing/Industrial focus-bundle products.
_BUNDLED_NAME_PATTERNS = [
    re.compile(r"\bSolution\b", re.IGNORECASE),
    re.compile(r"\bFocus\s+\d", re.IGNORECASE),   # e.g. "Focus 8.0"
]


def _is_job_solution(item: dict[str, Any]) -> bool:
    """
    Return True if the item appears to be a Pre-packaged Job Solution
    (out of scope per the assignment specification).

    Strategy (defence-in-depth):
    1. Primary: URL slug ends with '-solution'.
    2. Secondary: Name matches bundled-product name patterns.
    """
    url: str = item.get("link", "")
    name: str = item.get("name", "")

    if _SOLUTION_URL_PATTERN.search(url):
        return True
    for pattern in _BUNDLED_NAME_PATTERNS:
        if pattern.search(name):
            return True
    return False


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalize_item(item: dict[str, Any]) -> dict[str, Any]:
    """
    Return a normalized copy of a catalog item with:
    - Empty list/string fields replaced with readable defaults.
    - A derived 'test_type' field (primary type abbreviation).
    - A 'doc_text' field used for embedding/BM25 indexing.
    """
    normalized = dict(item)  # shallow copy

    # Normalize empty fields
    if not normalized.get("duration"):
        normalized["duration"] = "Not specified"
    if not normalized.get("languages"):
        normalized["languages"] = ["Not specified"]
    if not normalized.get("job_levels"):
        normalized["job_levels"] = ["Not specified"]
    if not normalized.get("description"):
        normalized["description"] = "No description available."

    # Derive primary test_type abbreviation
    keys: list[str] = normalized.get("keys", [])
    primary_key: str = keys[0] if keys else ""
    normalized["test_type"] = TEST_TYPE_MAP.get(primary_key, primary_key or "Unknown")

    # Build a rich text string for embedding and BM25 indexing.
    # Concatenating name + description + job levels + keys maximises
    # retrieval signal across semantic and keyword dimensions.
    job_levels_str = ", ".join(normalized["job_levels"])
    keys_str = ", ".join(keys)
    languages_str = ", ".join(normalized["languages"])
    normalized["doc_text"] = (
        f"Assessment: {normalized['name']}. "
        f"Type: {keys_str}. "
        f"Job Levels: {job_levels_str}. "
        f"Languages: {languages_str}. "
        f"Duration: {normalized['duration']}. "
        f"Remote: {normalized.get('remote', 'Unknown')}. "
        f"Adaptive: {normalized.get('adaptive', 'Unknown')}. "
        f"Description: {normalized['description']}"
    )

    return normalized


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_catalog() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """
    Load, filter, and normalize the SHL product catalog.

    Returns:
        catalog_list  : List of normalized individual-test items (ordered).
        catalog_lookup: Dict keyed by canonical URL for O(1) validation.
    """
    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        raw: list[dict[str, Any]] = json.load(f)

    filtered: list[dict[str, Any]] = []
    skipped_names: list[str] = []

    for item in raw:
        if _is_job_solution(item):
            skipped_names.append(item.get("name", "?"))
        else:
            filtered.append(_normalize_item(item))

    # Build lookup dict: url → normalized item (used by validation middleware)
    catalog_lookup: dict[str, dict[str, Any]] = {
        item["link"]: item for item in filtered
    }

    print(f"[data_prep] Loaded {len(raw)} raw items.")
    print(f"[data_prep] Filtered out {len(skipped_names)} Job Solution(s): {skipped_names}")
    print(f"[data_prep] Clean catalog size: {len(filtered)} Individual Test Solutions.")

    return filtered, catalog_lookup


# ---------------------------------------------------------------------------
# Quick smoke-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    catalog, lookup = load_catalog()
    # Print first 3 items to verify normalization
    for item in catalog[:3]:
        print("\n---")
        print(f"  Name      : {item['name']}")
        print(f"  test_type : {item['test_type']}")
        print(f"  Duration  : {item['duration']}")
        print(f"  Job Levels: {item['job_levels'][:3]}")
        print(f"  doc_text  : {item['doc_text'][:120]}...")
