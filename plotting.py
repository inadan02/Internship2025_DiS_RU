import pandas as pd
import matplotlib.pyplot as plt

# Load your logs
df_old = pd.read_csv("C:/MASTER/INTERNSHIP/CODE/parameter_backdoor/results/asr_log_bilevel.csv")
df_new = pd.read_csv("C:/MASTER/INTERNSHIP/CODE/parameter_backdoor/results/asr_log_bilevel_v2.csv")

# Average per step
mean_old = df_old.groupby("step")[["asr_val", "sat"]].mean()
mean_new = df_new.groupby("step")[["asr_val", "sat"]].mean()

# Plot with distinct colors
fig, ax1 = plt.subplots(figsize=(10, 6))

# ASR (validation)
ax1.plot(mean_old.index, mean_old["asr_val"], color="#1f77b4", linestyle="--", linewidth=2, label="ASR (old)")
ax1.plot(mean_new.index, mean_new["asr_val"], color="#2ca02c", linestyle="-", linewidth=2, label="ASR (v2)")
ax1.set_xlabel("Step", fontsize=12)
ax1.set_ylabel("ASR (val)", fontsize=12, color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.grid(True, linestyle="--", alpha=0.5)

# Saturation (on right y-axis)
ax2 = ax1.twinx()
ax2.plot(mean_old.index, mean_old["sat"], color="#ff7f0e", linestyle="--", linewidth=2, label="Saturation (old)")
ax2.plot(mean_new.index, mean_new["sat"], color="#d62728", linestyle="-", linewidth=2, label="Saturation (v2)")
ax2.set_ylabel("Saturation", fontsize=12, color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")

# Title & legends
fig.suptitle("ASR vs Saturation: Bilevel (old) vs Bilevel v2 (λ_sat_mean Added)", fontsize=14, weight="bold")
ax1.legend(loc="upper left", frameon=True)
ax2.legend(loc="upper right", frameon=True)

plt.tight_layout()
plt.show()
