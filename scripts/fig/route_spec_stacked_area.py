"""Run at `scripts/` with `python3 -m fig.route_spec_stacked_area`.
Data are from here:
<https://github.com/SichangHe/internet_route_verification/issues/88>

Adopted from `as_spec_stacked_area.py`.
"""
import matplotlib.pyplot as plt
import pandas as pd
from fig import download_if_missing
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FILE = "route_stats1.csv.gz"
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
    df = pd.read_csv(FILE, dtype="uint16")

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
        rasterized=True,
    )
    ax.set_xlabel("Route", fontsize=16)
    ax.set_ylabel(f"Percentage of Special Case", fontsize=16)
    ax.tick_params(axis="both", labelsize=14)
    ax.grid()
    ax.legend(loc="lower center", fontsize=14)

    # For checking.
    # fig.show()

    return fig, ax, d


def main():
    download_if_missing(
        "https://github.com/SichangHe/internet_route_verification/releases/download/data-88/route_stats1.csv.gz",
        FILE,
    )
    fig, _, _ = plot()

    pdf_name = f"route-special-case-percentages-stacked-area.pdf"
    fig.savefig(pdf_name, bbox_inches="tight")
    fig.set_size_inches(8, 6)
    fig.savefig(pdf_name.replace(".pdf", "-squared.pdf"), bbox_inches="tight")


if __name__ == "__main__":
    main()