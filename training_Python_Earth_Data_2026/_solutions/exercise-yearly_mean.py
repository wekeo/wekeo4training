# Extract the year from each date
df = df_monthly
df["year"] = df_monthly["date"].dt.year

# Calculate the mean NDVI for each year
df_yearly = df.groupby("year")["mean_ndvi"].mean().reset_index()

# Display the results
print(df_yearly)

# Plot yearly mean evolution for 2015-2025
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df_yearly["year"], df_yearly["mean_ndvi"], marker="o")

ax.set_xlabel("Year")
ax.set_ylabel("Mean NDVI")
ax.set_title("Yearly Mean Vegetation Greenness")
ax.grid(True, alpha=0.3)

plt.show()