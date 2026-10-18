import numpy as np
import pandas as pd
import pytest

from sepsis.features import TIME_VARYING, build_features
from sepsis.splits import assign_splits, patient_table


def make_df(n_patients=6, hours=30, seed=0, hospital="A"):
    rng = np.random.default_rng(seed)
    frames = []
    for p in range(n_patients):
        d = pd.DataFrame(rng.normal(size=(hours, len(TIME_VARYING))), columns=TIME_VARYING)
        d = d.mask(rng.random(d.shape) < 0.6)  # sparse measurements
        d["Age"], d["Gender"] = 60.0, p % 2
        d["Unit1"], d["Unit2"], d["HospAdmTime"] = np.nan, np.nan, -1.0
        d["ICULOS"] = np.arange(1, hours + 1)
        d["SepsisLabel"] = (np.arange(hours) >= 20).astype(int) if p % 3 == 0 else 0
        d.insert(0, "patient_id", f"{hospital}_p{p:06d}")
        d.insert(1, "hospital", hospital)
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


@pytest.mark.parametrize("t", [0, 5, 13, 24])
def test_features_are_causal(t):
    """Changing a patient's rows after hour t must not change any feature at hours <= t."""
    df = make_df()
    base = build_features(df)
    pid = df.patient_id.iloc[0]
    changed = df.copy()
    mask = (changed.patient_id == pid) & (changed.ICULOS > t + 1)  # ICULOS = row index + 1
    rng = np.random.default_rng(1)
    changed.loc[mask, TIME_VARYING] = rng.normal(size=(mask.sum(), len(TIME_VARYING))) * 100
    got = build_features(changed)
    keep = (base.patient_id == pid) & (base.ICULOS <= t + 1)
    assert keep.sum() == t + 1
    cols = [c for c in base.columns if c not in ("SepsisLabel",)]
    pd.testing.assert_frame_equal(base.loc[keep, cols], got.loc[keep, cols])
    # and other patients are untouched entirely
    other = base.patient_id != pid
    pd.testing.assert_frame_equal(base[other], got[other])


def test_feature_values_small_example():
    df = make_df(n_patients=1, hours=8)
    df[TIME_VARYING] = np.nan
    df.loc[[1, 4], "HR"] = [100.0, 110.0]
    f = build_features(df)
    assert np.isnan(f.HR_ffill[0]) and f.HR_ffill[2] == 100 and f.HR_ffill[6] == 110
    assert list(f.HR_measured) == [0, 1, 0, 0, 1, 0, 0, 0]
    assert np.isnan(f.HR_hrs_since[0])
    assert list(f.HR_hrs_since[1:8]) == [0, 1, 2, 0, 1, 2, 3]
    # 6h window at row 4: ffilled values 100,100,100,110
    assert f.HR_mean6[4] == pytest.approx(102.5)
    assert f.HR_slope6[0] != f.HR_slope6[0]  # NaN: fewer than 2 points
    # perfectly linear ramp -> slope 2/h
    df.loc[:, "HR"] = 2.0 * np.arange(8)
    assert build_features(df).HR_slope6.iloc[-1] == pytest.approx(2.0, abs=1e-4)


def test_split_integrity():
    df = pd.concat([make_df(40, 5, 0, "A"), make_df(30, 5, 1, "B")], ignore_index=True)
    pt = patient_table(df)
    split = assign_splits(pt, 0.2, seed=7)
    assert split.index.is_unique
    # hospital B only ever in test; hospital A never in test
    assert set(split[pt.hospital == "B"]) == {"test"}
    assert set(split[pt.hospital == "A"]) == {"train", "validation"}
    # stratified: both splits contain ever-septic patients, in about the same proportion
    a = pt[pt.hospital == "A"].assign(split=split)
    rates = a.groupby("split").ever_septic.mean()
    assert abs(rates["train"] - rates["validation"]) < 0.1
    # row-level: each patient maps to exactly one split
    rows = df.assign(split=df.patient_id.map(split))
    assert rows.groupby("patient_id").split.nunique().max() == 1
    assert assign_splits(pt, 0.2, seed=7).equals(split)  # deterministic
