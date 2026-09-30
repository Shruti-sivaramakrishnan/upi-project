import numpy as np
import pandas as pd

mu_floor = 6903.325617283951
mu_sens_high = 12425.986111111111
mu_needed_fixed_target = 11630.518859552725

mu_grid = np.linspace(mu_floor, mu_sens_high, 200)
growth_pct = (mu_needed_fixed_target / mu_grid - 1) * 100

df = pd.DataFrame({"mu_value": mu_grid, "growth_pct_required": growth_pct})
df.to_csv("processed/capacity_growth_sweep_200.csv", index=False)

median = np.median(growth_pct)
p5, p95 = np.percentile(growth_pct, [5, 95])
print(f"Median: {median:.1f}%")
print(f"5th percentile: {p5:.1f}%")
print(f"95th percentile: {p95:.1f}%")
print(f"Saved {len(df)} rows to processed/capacity_growth_sweep_200.csv")
