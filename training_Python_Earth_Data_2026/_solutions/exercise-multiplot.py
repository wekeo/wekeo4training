# Choose years to plot
years = [2015, 2020, 2025]
months = ["01", "06", "09"]

fig, axes = plt.subplots(
    3, 3, figsize=(14, 8),
    subplot_kw={"projection": ccrs.PlateCarree()}
)

for ax, (year, month) in zip(axes.flat, [(y, m) for y in years for m in months]):
    path = f"data/NDVI_21-{month}-{year}.nc"

    # Clean NDVI for this year
    ndvi_clean = clean_ndvi(path)

    # Add geographic features
    basic_geo_features(ax, lon_min, lon_max, lat_min, lat_max)

    # Plot NDVI
    im = ax.pcolormesh(
        ndvi_clean.lon,
        ndvi_clean.lat,
        ndvi_clean,
        cmap="YlGn",
        vmin=-0.1,
        vmax=0.9,
        shading="auto"
    )

    # Set title for each subplot
    ax.set_title(f"NDVI - {month}/{year}")

# Adjusting subplots position
fig.subplots_adjust(
    left=0.06, right=0.7,
    top=0.9, bottom=0.16,
    wspace=0.7, hspace=0.6
)

# Shared colorbar
cbar_ax = fig.add_axes([0.25, 0.001, 0.50, 0.025])
fig.colorbar(im, cax=cbar_ax, orientation="horizontal", label="NDVI")

plt.tight_layout()
plt.show()