import pandas as pd
import glob
import os

# Grab the latest benchmark CSV generated
list_of_files = glob.glob("benchmark_results_*.csv")
if not list_of_files:
    # Fallback to default output name if timestamp isn't present
    latest_file = "benchmark_results.csv"
else:
    latest_file = max(list_of_files, key=os.path.getctime)

print(f"Analyzing latest benchmark run: {latest_file}\n")
df = pd.read_csv(latest_file)

# Exclude error rows if any exist
df_valid = df[df["Decision"] != "ERROR"].copy()

# 1. Overall Acceptance Rates
print("=== ACCEPTANCE RATES BY CONDITION ===")
summary = df_valid.groupby("Condition")["Decision"].value_counts(normalize=True).unstack().fillna(0) * 100
print(summary.round(2))

# 2. Average Quality Metric Improvements
print("\n=== MEAN QUALITY IMPROVEMENT (%) BY CONDITION ===")
quality_summary = df_valid.groupby("Condition")["Quality Improvement"].mean() * 100
print(quality_summary.round(2))

# 3. Average AI Confidence Delta
print("\n=== MEAN AI CONFIDENCE DELTA BY CONDITION ===")
conf_summary = df_valid.groupby("Condition")["AI Confidence Delta"].mean()
print(conf_summary.round(4))