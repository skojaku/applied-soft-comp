"""
This script is used to balance the dataset by downsampling the top queues.
"""
import pandas as pd

# Load the dataset
# Download the dataset from https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets?resource=download
# And unpack it in the data/service-ticket-dataset directory
df = pd.read_csv("dataset-tickets-multi-lang3-4k.csv")

# Filter for English tickets
dg = df.query("language == 'en'")

# Count the number of tickets per queue
queue_counts = dg.groupby("queue").size().sort_values(ascending=False).reset_index(name="count")

# First find the top queues whose cumulative counts are more than 90% of the total
cumulative_counts = queue_counts["count"].cumsum()
threshold = 0.9 * queue_counts["count"].sum()
top_queues = queue_counts[cumulative_counts <= threshold]

# Downsample the top queues to match the counts of the top queues
min_count = top_queues["count"].min()
focal_queues = top_queues["queue"].tolist()

dg_balanced = dg[dg["queue"].isin(focal_queues)]
dg_balanced = dg_balanced.groupby("queue").apply(lambda x: x.sample(min_count))
dg_balanced.reset_index(drop=True, inplace=True)

# Verify the balance
queue_counts = dg_balanced.groupby("queue").size().sort_values(ascending=False).reset_index(name="count")
queue_counts

# Save the balanced dataset
dg_balanced.to_csv("data.csv", index=False)
# %%
