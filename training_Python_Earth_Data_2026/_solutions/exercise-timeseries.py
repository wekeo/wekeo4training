# Create figure and primary axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot NDVI on the left y-axis
ax1.plot(df_combined["date_x"], df_combined["mean_ndvi"], color="forestgreen", label="Mean NDVI")
ax1.set_ylabel("Mean NDVI", color="forestgreen")
ax1.tick_params(axis="y", labelcolor="forestgreen")
ax1.grid(True, alpha=0.3)

# Plot precipitation on the right y-axis
ax2 = ax1.twinx()
ax2.plot(df_combined["date_x"], df_combined["precipitation"], color="cyan", label="Temperature")
ax2.set_ylabel("Precipitation (mm)", color="cyan")
ax2.tick_params(axis="y", labelcolor="cyan")

# Add title and format dates
plt.title("Monthly NDVI and Precipitation in the Tabernas Area")
fig.autofmt_xdate()
plt.tight_layout()
plt.show()