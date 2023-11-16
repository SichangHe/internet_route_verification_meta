"""Run at `scripts/` with `python3 -m scripts.fig.as_spec_stacked_area`.
Data are from here:
<https://github.com/SichangHe/internet_route_verification/issues/89>
"""
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scripts.csv_files import as_stats

FILE = as_stats
TAGS = (
    "spec_export_customers",
    "spec_as_is_origin_but_no_route",
    "spec_as_set_contains_origin_but_no_route",
    "spec_import_from_neighbor",
    "spec_uphill",
    "spec_uphill_tier1",
    "spec_tier1_pair",
    "spec_import_peer_oifps",
    "spec_import_customer_oifps",
)


def plot():
    df = pd.read_csv(FILE.path)

    d = pd.DataFrame({"total": sum(df[tag] for tag in TAGS)})
    for tag in TAGS:
        d[f"%{tag}"] = df[tag] / d["total"] * 100.0
    d.dropna(inplace=True)
    d = d.sort_values(
        by=[f"%{tag}" for tag in TAGS],
        ascending=False,
        ignore_index=True,
    )

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(figsize=(16, 9))
    fig.tight_layout()
    ax.stackplot(
        d.index,
        [d[f"%{tag}"] for tag in TAGS],
        labels=[f"%{tag}" for tag in TAGS],
    )
    ax.set_xlabel("AS", fontsize=16)
    ax.set_ylabel(f"Percentage of Special Case", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid()
    ax.legend(loc="lower center", fontsize=14)

    # For checking.
    # fig.show()

    return fig, ax, d


def main():
    FILE.download_if_missing()
    fig, _, _ = plot()

    pdf_name = f"AS-special-case-percentages-stacked-area.pdf"
    fig.savefig(pdf_name, bbox_inches="tight")
    fig.set_size_inches(8, 6)
    fig.savefig(pdf_name.replace(".pdf", "-squared.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()
