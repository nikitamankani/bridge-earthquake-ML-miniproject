# Loading and cleaning the bridge-earthquake dataset.
#
# The raw CSV (data/bridge_earthquake_data.csv) is used as-is. It has 999 rows and
# 28 columns, but only 249 rows are real records - the rest are empty padding rows.
# There are also 17 empty "Unnamed" columns, one of which holds the codebook text.

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

# codes taken from the codebook written inside the CSV
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


def load_raw(path=RAW_CSV):
    # latin-1 because the file has a few degree symbols in it
    return pd.read_csv(path, encoding="latin-1")


def bridge_id(name):
    # The damaging record of a bridge has the build year in its name,
    # e.g. "Interstate 10 (over La Cienega Boulevard), 1957", but its other
    # records don't. Removing the ", 1957" part gives one name per bridge
    # (26 different names -> 20 actual bridges).
    return re.sub(r",\s*\d{4}\s*$", "", str(name).strip()).strip()


def clean(raw):
    # drop the empty "Unnamed" columns and the empty padding rows
    df = raw.loc[:, ~raw.columns.str.startswith("Unnamed")]
    df = df.rename(columns=COL_RENAME)
    df = df.dropna(how="all")
    df = df.dropna(subset=["damaged"]).copy()

    # some text fields have extra spaces
    for c in ["bridge", "earthquake", "material", "bridge_type", "damage_level"]:
        df[c] = df[c].astype(str).str.strip()

    # we don't fill in any missing values, so make sure there aren't any
    missing = df[FEATURE_COLS + ["damaged", "damage_level"]].isna().sum()
    assert missing.sum() == 0, f"unexpected missing values:\n{missing}"

    # convert the roman-numeral codes into readable names / numbers
    df["damaged"] = df["damaged"].astype(int)
    df["material"] = df["material"].map(MATERIAL_NAMES)
    df["bridge_type"] = df["bridge_type"].map(BRIDGE_TYPE_NAMES)
    df["damage_level_int"] = df["damage_level"].map(DAMAGE_LEVEL_TO_INT)
    assert df[["material", "bridge_type", "damage_level_int"]].notna().all().all(), "unknown code"

    # sanity check: damaged = 1 exactly when damage level is II or III
    assert ((df["damage_level_int"] > 0).astype(int) == df["damaged"]).all()

    # bridge id, used to keep each bridge in a single CV fold
    df[GROUP_COL] = df["bridge"].map(bridge_id)
    return df.reset_index(drop=True)


def load_clean(path=RAW_CSV):
    return clean(load_raw(path))


if __name__ == "__main__":
    d = load_clean()
    print("Rows after cleaning:", len(d))
    print("Damaged:", int(d["damaged"].sum()), "| Not damaged:", int((d["damaged"] == 0).sum()))
    print("Damage levels:", d["damage_level"].value_counts().reindex(["I", "II", "III"]).to_dict())
    print("Bridge names in file:", d["bridge"].nunique(), "| Actual bridges:", d[GROUP_COL].nunique())
