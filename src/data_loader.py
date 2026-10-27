"""Load and clean the bridge-earthquake dataset.

Source file: data/bridge_earthquake_data.csv (kept unmodified).
The raw CSV has 999 rows x 28 columns: 249 real observations, 750 empty
padding rows, 17 empty "Unnamed" columns, and a codebook stored as text in
one cell of the first row. Cleaning steps are listed in clean() and README.
"""

import re
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = PROJECT_ROOT / "data" / "bridge_earthquake_data.csv"

COL_RENAME = {
    "Name of the bridge": "bridge",
    "Earthquake": "earthquake",
    "x1 = Material Type": "material",
    "x2 = Age (yr)": "age",
    "x3 = Bridge Type": "bridge_type",
    "x4 = Earthquake Magnitude": "magnitude",
    "x5 = Distance to epicenter (miles)": "distance_mi",
    "y1 = Damaged or not?": "damaged",
    "y2 = Damage Level": "damage_level",
    "Note": "note",
    "Location": "location",
}

# Codebook copied from the note embedded in the raw CSV.
MATERIAL_NAMES = {"I": "steel", "II": "concrete"}
BRIDGE_TYPE_NAMES = {"I": "beam_girder", "II": "truss", "III": "cable_suspension", "IV": "arch"}
DAMAGE_LEVEL_TO_INT = {"I": 0, "II": 1, "III": 2}
DAMAGE_LEVEL_LABELS = [
    "Level I: No damage",
    "Level II: Moderate/severe damage",
    "Level III: Partial/overall collapse",
]
DAMAGE_LEVEL_SHORT = ["I: No damage", "II: Mod./severe", "III: Collapse"]

FEATURE_COLS = ["material", "age", "bridge_type", "magnitude", "distance_mi"]
CATEGORICAL_COLS = ["material", "bridge_type"]
NUMERIC_COLS = ["age", "magnitude", "distance_mi"]
BINARY_TARGET = "damaged"
MULTI_TARGET = "damage_level_int"
GROUP_COL = "bridge_id"


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Read the CSV exactly as supplied (latin-1 handles the degree symbols)."""
    return pd.read_csv(path, encoding="latin-1")


def bridge_id(name: str) -> str:
    """Canonical physical-bridge name.

    The first (damaging) record of a bridge carries its construction year,
    e.g. "Interstate 10 (over La Cienega Boulevard), 1957", while later
    records of the same bridge omit it. Removing a trailing ", YYYY" and
    whitespace maps both spellings to one bridge (26 raw names -> 20 bridges).
    """
    return re.sub(r",\s*\d{4}\s*$", "", str(name).strip()).strip()


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    """Apply the documented cleaning steps; no rows are imputed or invented."""
    df = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]   # 1. drop 17 empty/codebook columns
    df = df.rename(columns=COL_RENAME)
    df = df.dropna(how="all")                                  # 2. drop 750 empty padding rows
    df = df.dropna(subset=["damaged"]).copy()

    for c in ["bridge", "earthquake", "material", "bridge_type", "damage_level"]:
        df[c] = df[c].astype(str).str.strip()                  # 3. strip stray whitespace

    missing = df[FEATURE_COLS + ["damaged", "damage_level"]].isna().sum()
    assert missing.sum() == 0, f"unexpected missing values:\n{missing}"

    df["damaged"] = df["damaged"].astype(int)
    df["material"] = df["material"].map(MATERIAL_NAMES)        # 4. decode codebook categories
    df["bridge_type"] = df["bridge_type"].map(BRIDGE_TYPE_NAMES)
    df["damage_level_int"] = df["damage_level"].map(DAMAGE_LEVEL_TO_INT)
    assert df[["material", "bridge_type", "damage_level_int"]].notna().all().all(), "unknown code"

    # 5. consistency check: damaged == 1  <=>  damage level II or III
    assert ((df["damage_level_int"] > 0).astype(int) == df["damaged"]).all()

    df[GROUP_COL] = df["bridge"].map(bridge_id)               # 6. physical-bridge group id
    return df.reset_index(drop=True)


def load_clean(path=RAW_CSV) -> pd.DataFrame:
    return clean(load_raw(path))


if __name__ == "__main__":
    d = load_clean()
    print(d.shape)
    print(d["damaged"].value_counts().to_dict(), d["damage_level"].value_counts().to_dict())
    print("raw bridge names:", d["bridge"].nunique(), "| physical bridges:", d[GROUP_COL].nunique())
