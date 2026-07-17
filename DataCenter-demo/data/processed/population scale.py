import os
import shutil
import numpy as np

# -----------------------------
# Files to process
# -----------------------------
files = [
    "d_hat.npy",
    "d_hat_azur.npy",
    "d_hat_test.npy",
]

# -----------------------------
# Demand locations, in J order
# -----------------------------
cities = [
    "NewYork_NY",
    "LosAngeles_CA",
    "Chicago_IL",
    "Dallas_TX",
    "Houston_TX",
    "WashingtonDC",
    "Miami_FL",
    "Philadelphia_PA",
    "Atlanta_GA",
    "Phoenix_AZ",
]

# Approximate metro-area populations.
# Only relative magnitudes matter because they are normalized below.
population = np.array([
    19_641_225,  # New York
    13_288_904,  # Los Angeles
    9_879_320,   # Chicago
    7_978_340,   # Dallas
    7_975_220,   # Houston
    6_538_392,   # Washington DC
    6_423_080,   # Miami
    6_245_051,   # Philadelphia
    6_577_299,   # Atlanta
    5_070_110,   # Phoenix
], dtype=float)

# Normalize to [0, 1] by max population
pop_coef = population / population.max()

print("Population coefficients:")
for city, coef in zip(cities, pop_coef):
    print(f"{city:18s}: {coef:.4f}")

# Shape for broadcasting to S * J * T
coef = pop_coef.reshape(1, 10, 1)

for fname in files:
    if not os.path.exists(fname):
        raise FileNotFoundError(f"File not found: {fname}")

    backup_name = fname.replace(".npy", ".bak.npy")
    shutil.copy2(fname, backup_name)
    print(f"Backup saved: {backup_name}")

    data = np.load(fname)

    if data.ndim != 3:
        raise ValueError(f"{fname} should be a 3D array with shape S*J*T, got {data.shape}")

    if data.shape[1] != 10:
        raise ValueError(f"{fname} should have J=10 in axis 1, got shape {data.shape}")

    scaled_data = data * coef

    np.save(fname, scaled_data)
    print(f"Scaled and overwritten: {fname}, shape={scaled_data.shape}")

print("Done.")