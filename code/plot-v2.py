import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# =========================
# Read Excel File
# =========================
# excel_file = "results_batch_8_BS-radius_200_RIS-radius_100_vary_Pmax-seed-0/summary_P.xlsx"
excel_file = "results_batch_8_BS-radius_200_RIS-radius_100_vary_M-seed-0/summary_M.xlsx"
df = pd.read_excel(excel_file, index_col=0)

# Convert column headers to float (x-axis)
x = df.columns.astype(float)

# =========================
# Plot Styling (Paper Ready)
# =========================
plt.rcParams.update({
    "font.size": 13,
    "font.family": "serif",
    "axes.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 3,
    "ytick.minor.size": 3,
})

# styles = {
#     # ===== Centralized (Red) =====
#     "centralized": {
#         "linestyle": "-",
#         "marker": "o",
#         "color": "tab:red"
#     },
#     "centralized_discrete": {
#         "linestyle": "--",
#         "marker": "o",
#         "color": "tab:red"
#     },

#     # ===== Decentralized (Blue) =====
#     "decentralized": {
#         "linestyle": "-",
#         "marker": "s",
#         "color": "tab:blue"
#     },
#     "decentralized_discrete": {
#         "linestyle": "--",
#         "marker": "s",
#         "color": "tab:blue"
#     },
# }

# styles = {
#     # ===== Centralized (Red) =====
#     "centralized": {
#         "linestyle": "-",
#         "marker": "o",
#         "color": "tab:red"
#     },
#     "centralized_discrete": {
#         "linestyle": "--",
#         "marker": "o",
#         "color": "tab:red"
#     },
#     "centralized_random_phase_discrete": {
#         "linestyle": ":",
#         "marker": "o",
#         "color": "tab:red"
#     },

#     # ===== Decentralized (Blue) =====
#     "decentralized": {
#         "linestyle": "-",
#         "marker": "s",
#         "color": "tab:blue"
#     },
#     "decentralized_discrete": {
#         "linestyle": "--",
#         "marker": "s",
#         "color": "tab:blue"
#     },
#     "decentralized_random_phase_discrete": {
#         "linestyle": ":",
#         "marker": "s",
#         "color": "tab:blue"
#     },
# }



styles = {
    # ===== Centralized (Red) =====
    "centralized": {
        "linestyle": "-",
        "marker": "o",
        "color": "tab:red"
    },
    "centralized_discrete": {
        "linestyle": "--",
        "marker": "o",
        "color": "tab:red"
    },
    # "centralized_random_phase": {
        # "linestyle": "-.",
        # "marker": "o",
        # "color": "tab:red"
    # },
    "centralized_random_phase_discrete": {
        "linestyle": ":",
        "marker": "o",
        "color": "tab:red"
    },


    # ===== Decentralized (Blue) =====
    "decentralized": {
        "linestyle": "-",
        "marker": "s",
        "color": "tab:blue"
    },
    "decentralized_discrete": {
        "linestyle": "--",
        "marker": "s",
        "color": "tab:blue"
    },
    # "decentralized_random_phase": {
        # "linestyle": "-.",
        # "marker": "s",
        # "color": "tab:blue"
    # },
    "decentralized_random_phase_discrete": {
        "linestyle": ":",
        "marker": "s",
        "color": "tab:blue"
    },
}



fig, ax = plt.subplots(figsize=(7.2, 5.4))  # slightly more "paper" aspect

# =========================
# Plot Selected Curves
# =========================

label_map = {
    "centralized": "Centralized",
    "centralized_discrete": "Centralized (D)",
    "centralized_random_phase": "Centralized (R)",
    "centralized_random_phase_discrete": "Centralized (R-D)",
    "decentralized": "Decentralized",
    "decentralized_discrete": "Decentralized (D)",
    "decentralized_random_phase": "Decentralized (R)",
    "decentralized_random_phase_discrete": "Decentralized (R-D)",
}


for key in styles:
    if key in df.index:
        ax.plot(
            x,
            df.loc[key].values,
            color=styles[key]["color"],   
            linestyle=styles[key]["linestyle"],
            marker=styles[key]["marker"],
            linewidth=2.2,
            markersize=8,
            markerfacecolor="none",     # hollow markers like the reference
            markeredgewidth=1.6,        # thick marker edge
            label=label_map.get(key, key)
        )

ax.set_xlabel("$M$")
# ax.set_xlabel(r"$P_{\mathrm{max}}$ (dBm)")
ax.set_ylabel("Sum Rate (bps/Hz)")

ax.set_xticks(x)  # force integer ticks exactly at your data points

# Minor ticks + dotted major/minor grid (no alpha -> EPS friendly)
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.grid(True, which="major", linestyle=":", linewidth=0.9)
ax.grid(True, which="minor", linestyle=":", linewidth=0.6)

# Legend: boxed white panel with black border (like the reference)
leg = ax.legend(loc="lower right", fontsize=10, frameon=True, fancybox=False, framealpha=1.0)
# leg = ax.legend(loc="upper left", fontsize=10, frameon=True, fancybox=False, framealpha=1.0)
leg.get_frame().set_edgecolor("black")
leg.get_frame().set_linewidth(1.0)
leg.get_frame().set_facecolor("white")

fig.tight_layout()

# =========================
# Save Figure
# =========================
# fig.savefig("grouped_plot2-seed-0.eps", format="eps", dpi=300)
fig.savefig("grouped_plot2-seed-0.pdf", dpi=300)
# fig.savefig("grouped_plot2-seed-0-pmax.pdf", dpi=300)

# No plt.show() on headless machine
plt.close(fig)
print("Saved: grouped_plot.eps and grouped_plot.pdf")
