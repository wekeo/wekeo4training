# Calculate the average NDVI map for each calendar month
monthly_climatology = {}

for month in range(1, 13):
    monthly_data = [da for date, da in rasters.items() if date.month == month]

    # Calculate monthly climatology
    monthly_climatology[month] = xr.concat(monthly_data, dim="observation").mean(dim="observation", skipna=True)

# Choose the month to analyse
date = pd.Timestamp("2023-09-21")

# Calculate the anomaly relative to the climatology
anomaly_map = rasters[date] - monthly_climatology[date.month]

# Plot the anomaly map
fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={"projection": ccrs.PlateCarree()})

basic_geo_features(ax, lon_min, lon_max, lat_min, lat_max)

im = ax.pcolormesh(anomaly_map.lon, anomaly_map.lat, anomaly_map,
    cmap="RdBu_r", vmin=-0.15, vmax=0.15, shading="auto")

fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.12, shrink=0.75, label="Monthly NDVI anomaly")

ax.set_title(f"NDVI anomaly on {date.date()}, relative to 2015–2025 climatology")
plt.show()