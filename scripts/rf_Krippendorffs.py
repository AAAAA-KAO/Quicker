import krippendorff  # pip install krippendorff
import pandas as pd
import os
import numpy as np

# Data shape: Rater × Sample, e.g., 3 x 65, with elements in the range 1–5
np.random.seed(42)
database = 'rf_data'
rate_result = []
for file in os.listdir(f'{database}'):
    if file.endswith('.csv'):
        raw_data = pd.read_csv(f'{database}/{file}')
        np_data = raw_data.values
        # np_data.shape should be 3 x 65 (Raters x Samples)
        np_data = np_data.reshape(1, -1)
        rate_result.append(np_data)  # Stored here

np_rate_result = np.concatenate(
    (rate_result[0], rate_result[1], rate_result[2]), axis=0
)

bootstrap_results = []
for _ in range(1000):
    idx = np.random.choice(
        np_rate_result.shape[1], np_rate_result.shape[1], replace=True
    )
    bootstrap_results.append(
        krippendorff.alpha(
            reliability_data=np_rate_result[:, idx], level_of_measurement='ordinal'
        )
    )
ci = np.percentile(bootstrap_results, [2.5, 97.5])
print("95% Confidence Interval:", ci)
# print(f"Krippendorff's alpha is: {alpha_ord}")
