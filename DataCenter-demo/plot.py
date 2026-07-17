import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from scipy.ndimage import gaussian_filter
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.lines import Line2D


# =========================================================
# 1. Data
# =========================================================
data_centers = pd.DataFrame({
    "name": [
        "Ashburn_VA",
        "SiliconValley_CA",
        "Chicago_IL",
        "Dallas_TX",
        "Seattle_WA",
        "Phoenix_AZ"
    ],
    "lat": [39.0438, 37.3861, 41.8781, 32.7767, 47.6062, 33.4484],
    "lon": [-77.4874, -122.0839, -87.6298, -96.7970, -122.3321, -112.0740]
})

demand_points = pd.DataFrame({
    "name": [
        "NewYork_NY",
        "LosAngeles_CA",
        "Chicago_IL",
        "Dallas_TX",
        "Houston_TX",
        "WashingtonDC",
        "Miami_FL",
        "Philadelphia_PA",
        "Atlanta_GA",
        "Phoenix_AZ"
    ],
    "lat": [40.7128, 34.0522, 41.8781, 32.7767, 29.7604,
            38.9072, 25.7617, 39.9526, 33.7490, 33.4484],
    "lon": [-74.0060, -118.2437, -87.6298, -96.7970, -95.3698,
            -77.0369, -80.1918, -75.1652, -84.3880, -112.0740]
})

d_hat_path = r"C:\Users\15636\PycharmProjects\DataCenter-demo\data\processed\d_hat.npy"
d_hat = np.load(d_hat_path)

J = len(demand_points)
demand_vec = d_hat[0].reshape(J, -1).sum(axis=1)

demand_points["demand"] = demand_vec


# =========================================================
# 2. Utility: nearest data center for each demand point
# =========================================================
def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


nearest_links = []
for _, d in demand_points.iterrows():
    distances = data_centers.apply(
        lambda c: haversine(d["lat"], d["lon"], c["lat"], c["lon"]),
        axis=1
    )
    c = data_centers.loc[distances.idxmin()]
    nearest_links.append((d["lon"], d["lat"], c["lon"], c["lat"]))


# =========================================================
# 3. Data center interconnection network
#    You can adjust this according to your experimental setting.
# =========================================================
dc_edges = [
    ("Seattle_WA", "SiliconValley_CA"),
    ("Seattle_WA", "Chicago_IL"),
    ("SiliconValley_CA", "Phoenix_AZ"),
    ("Phoenix_AZ", "Dallas_TX"),
    ("Dallas_TX", "Chicago_IL"),
    ("Chicago_IL", "Ashburn_VA"),
    ("Dallas_TX", "Ashburn_VA"),
    ("Phoenix_AZ", "Chicago_IL"),
]

dc_dict = data_centers.set_index("name")[["lat", "lon"]].to_dict("index")


# =========================================================
# 4. Plot
# =========================================================
fig = plt.figure(figsize=(14, 8))
ax = plt.axes(projection=ccrs.LambertConformal(
    central_longitude=-96,
    central_latitude=39
))

ax.set_extent([-126, -66, 24, 50], crs=ccrs.PlateCarree())

# Base map
ax.add_feature(cfeature.LAND, facecolor="#f4f1e8")
ax.add_feature(cfeature.OCEAN, facecolor="#cfe8f3")
ax.add_feature(cfeature.LAKES, facecolor="#cfe8f3", edgecolor="none")
ax.add_feature(cfeature.STATES, edgecolor="lightgray", linewidth=0.6)
ax.add_feature(cfeature.BORDERS, edgecolor="gray", linewidth=0.6)
ax.add_feature(cfeature.COASTLINE, edgecolor="gray", linewidth=0.8)

# =========================================================
# Demand heatmap
# =========================================================
lon_min, lon_max = -126, -66
lat_min, lat_max = 24, 50

grid_lon = np.linspace(lon_min, lon_max, 500)
grid_lat = np.linspace(lat_min, lat_max, 300)
LON, LAT = np.meshgrid(grid_lon, grid_lat)

heat = np.zeros_like(LON)

sigma = 1.6   # 控制热力扩散范围，可调大/调小

for _, row in demand_points.iterrows():
    dist2 = (LON - row["lon"]) ** 2 + (LAT - row["lat"]) ** 2
    heat += row["demand"] * np.exp(-dist2 / (2 * sigma ** 2))

heat = heat / heat.max()

ax.contourf(
    LON, LAT, heat,
    levels=20,
    cmap="OrRd",
    alpha=0.35,
    transform=ccrs.PlateCarree(),
    zorder=1
)

# Demand-to-data-center links
for lon1, lat1, lon2, lat2 in nearest_links:
    ax.plot(
        [lon1, lon2], [lat1, lat2],
        linestyle="--",
        linewidth=1.1,
        color="gray",
        alpha=0.75,
        transform=ccrs.PlateCarree(),
        zorder=2
    )

# Data center interconnections
for a, b in dc_edges:
    ax.plot(
        [dc_dict[a]["lon"], dc_dict[b]["lon"]],
        [dc_dict[a]["lat"], dc_dict[b]["lat"]],
        linestyle="--",
        linewidth=1.5,
        color="#1f77ff",
        alpha=0.9,
        transform=ccrs.PlateCarree(),
        zorder=3
    )

# Plot data centers
ax.scatter(
    data_centers["lon"], data_centers["lat"],
    s=180,
    marker="s",
    color="#1565c0",
    edgecolor="white",
    linewidth=1.2,
    transform=ccrs.PlateCarree(),
    zorder=5
)

# Plot demand points
ax.scatter(
    demand_points["lon"], demand_points["lat"],
    s=110,
    marker="o",
    color="#d7301f",
    edgecolor="white",
    linewidth=1.0,
    transform=ccrs.PlateCarree(),
    zorder=6
)

# Labels for data centers
for _, row in data_centers.iterrows():
    ax.text(
        row["lon"] + 0.5, row["lat"] + 0.4,
        row["name"].replace("_", ", "),
        fontsize=9,
        color="#0b3d91",
        weight="bold",
        transform=ccrs.PlateCarree(),
        zorder=7
    )

# Labels for demand points
for _, row in demand_points.iterrows():
    ax.text(
        row["lon"] + 0.5, row["lat"] - 0.3,
        row["name"].replace("_", ", "),
        fontsize=8.5,
        color="#b2182b",
        weight="bold",
        transform=ccrs.PlateCarree(),
        zorder=7
    )

# Legend
legend_elements = [
    Line2D([0], [0], marker="s", color="w",
           label="Candidate Data Center Sites (6)",
           markerfacecolor="#1565c0", markeredgecolor="white", markersize=12),
    Line2D([0], [0], marker="o", color="w",
           label="Demand Locations (10)",
           markerfacecolor="#d7301f", markeredgecolor="white", markersize=10),
    Line2D([0], [0], linestyle="--", color="#1f77ff",
           label="Data Center Interconnections"),
    Line2D([0], [0], linestyle="--", color="gray",
           label="Demand-to-Data Center Connections")
]

plt.tight_layout()
plt.savefig("us_data_center_demand_map.png", dpi=300, bbox_inches="tight")
plt.savefig("us_data_center_demand_map.pdf", bbox_inches="tight")
plt.show()