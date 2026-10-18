import numpy as np
import pandas as pd

from sepsis.features import TIME_VARYING


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
