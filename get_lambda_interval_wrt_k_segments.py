import pandas as pd
import numpy as np
from opart import opart_k_segments

def squared_error(segment):
    if len(segment) == 0:
        return 0
    return np.sum((segment - np.mean(segment)) ** 2)

def calculate_distance(y, y_k):
    return np.sum((y - y_k) ** 2)

def get_stat(y):
    plausible_k = set(range(len(y)))
    kc = max(plausible_k)
    plausible_k.remove(kc)

    results = []  # To store (kc, next_lambda)

    while plausible_k:
        next_lambda = float('inf')
        next_k = 0

        for k in plausible_k:
            _, y_kc = opart_k_segments(y, kc)
            _, y_k = opart_k_segments(y, k)

            dist_diff = calculate_distance(y, y_kc) - calculate_distance(y, y_k)
            if k != kc:  # Prevent division by zero
                hit_time = dist_diff / (k - kc)

                if hit_time < next_lambda:
                    next_lambda = hit_time
                    next_k = k

        # Save kc and next_lambda for analysis
        results.append((kc, next_lambda))

        kc = next_k
        plausible_k = {k for k in plausible_k if k < kc}  # Remove k >= kc
    
    data = []
    for i in range(len(results)):
        k_segments = results[i][0]
        if i == 0:
            interval_start = 0
        else:
            interval_start = results[i - 1][1]
        interval_end = results[i][1]
        data.append({
            'Number of Changepoints': k_segments,
            'Interval Start': interval_start,
            'Interval End': interval_end
        })

    df = pd.DataFrame(data)
    return df