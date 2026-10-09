import pandas as pd, numpy as np
df = pd.read_csv("processed/mm1_queuing_results.csv", parse_dates=["month"]).sort_values("month").reset_index(drop=True)
print(len(df), "months", df.month.iloc[0].date(), "to", df.month.iloc[-1].date())
rho = df["rho_peak_baseline"].values

def devs(s):
    tr = list(range(s-6, s)) + list(range(s+3, s+9))
    t = np.array(tr, float); y = rho[tr]
    b, a = np.polyfit(t, y, 1)
    sd = (y - (a + b*t)).std(ddof=1)
    return [(rho[s+k] - (a + b*(s+k))) / sd for k in range(3)]

rows = []
for s in range(6, 58):
    d = devs(s)
    rows.append(dict(start=df.month[s].strftime("%Y-%m"), end=df.month[s+2].strftime("%Y-%m"), d1=d[0], d2=d[1], d3=d[2], mean_dev=np.mean(d), max_dev=max(d)))
W = pd.DataFrame(rows)
print("windows:", len(W))
c = W[W.start == "2025-03"].iloc[0]
print("GATE Mar-May 2025:", round(c.d1,2), round(c.d2,2), round(c.d3,2), "| expected +1.38 +0.55 -0.29")
for st in ["mean_dev", "max_dev"]:
    r = int((W[st] > c[st]).sum() + 1)
    print(st, "rank", r, "of", len(W), "p =", round(r/len(W), 3))
print(W.sort_values("mean_dev", ascending=False).head(5)[["start","end","mean_dev"]].round(3).to_string(index=False))
W.round(4).to_csv("processed/permutation_windows.csv", index=False)

print()
for n in ["rho_floor","rho_sens_low","rho_peak_baseline","rho_sens_high"]:
    v = df[n]; ge = df[v >= 1.0]
    print(n, "| rho>=1:", len(ge), "| rho>1:", int((v > 1).sum()), "| first:", ge.month.iloc[0].strftime("%Y-%m") if len(ge) else "none", "| max:", round(v.max(), 4), df.month[v.idxmax()].strftime("%Y-%m"))
