# Open another query
with open("./data/req_NDVI_09-2020.json", "r") as f:
    query_ndvi_2020 = json.load(f)
 
# Search   
matches = c.search(query_ndvi_2020)

# Download
matches.download(output_path)
