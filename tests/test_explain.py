import pandas as pd
import pytest

from sepsis.explain import base_variable, group_masks


@pytest.mark.parametrize(
    "feature, base",
    [
        ("HR_ffill", "HR"), ("HR_mean12", "HR"), ("Bilirubin_direct_hrs_since", "Bilirubin_direct"),
        ("Bilirubin_total_measured", "Bilirubin_total"), ("ICULOS", "ICULOS"), ("Unit1", "Unit1"),
    ],
)  # fmt: skip
def test_base_variable_maps_derived_features(feature, base):
    assert base_variable(feature) == base


def test_group_masks_partition_age_and_sex():
    df = pd.DataFrame({"Age": [30, 40, 59.9, 60, 74, 75, 90], "Gender": [0, 1, 0, 1, 0, 1, 0]})
    m = group_masks(df)
    assert sum(m[f"age {g}"].sum() for g in ["<40", "40-59", "60-74", "75+"]) == len(df)
    assert m["sex female"].sum() + m["sex male"].sum() == len(df)
    assert m["age 40-59"].tolist() == [False, True, True, False, False, False, False]
