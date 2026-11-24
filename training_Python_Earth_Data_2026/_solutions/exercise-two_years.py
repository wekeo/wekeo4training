# Select two years
years = [2015, 2025]

# Extract month and year
df = df_monthly
df["year"] = df_monthly["date"].dt.year
df["month"] = df_monthly["date"].dt.month

# Create one dataset combining both years
df_selected = df[df["year"].isin(years)]

# Plot
fig, ax = plt.subplots(figsize=(9, 5))

for year in years:
    data = df_selected[df_selected["year"] == year]
    ax.plot(data["month"], data["mean_ndvi"], marker="o", label=str(year))

ax.set_xlabel("Month")
ax.set_ylabel("Mean NDVI")
ax.set_title("Seasonal Vegetation Greenness: 2015 vs 2025")
ax.set_xticks(range(1, 13))
ax.legend(title="Year")
ax.grid(True, alpha=0.3)

plt.show()