"""Run at `scripts/` with `python3 -m scripts.stats.as_set_size_fitting`."""
import pandas as pd
from scipy.stats import fit, zipf
from scipy.stats._fit import FitResult
from scripts.csv_files import as_set_graph_stats

FILE = as_set_graph_stats


def main():
    FILE.download_if_missing()
    df_raw = pd.read_csv(FILE.path)
    df_wo_hash = df_raw[~df_raw["as_set"].str.contains("#")]
    total = len(df_wo_hash)
    print("Overview:")
    print(df_wo_hash.describe().applymap("{0:.2f}".format))

    print("\nAS Set sizes in AS Num counts.")
    df = df_wo_hash[df_wo_hash["n_nums"] > 0]
    empty = total - len(df)
    print(f"{empty} ({(empty * 100 / total):.2f}%) AS Sets have no AS Num.")
    res: FitResult = fit(zipf, df["n_nums"], [(1.0, 10.0)])
    print(f"Fitting Zipf distribution: Negative log-likelihood {res.nllf()}.")
    print(res)

    print("\nAS Set nesting depths.")
    df = df_wo_hash
    res = fit(zipf, df["depth"], [(1.0, 10.0)])
    print(f"Fitting Zipf distribution: Negative log-likelihood {res.nllf()}.")
    print(res)

    print("\nAS Set with cycles.")
    df = df_wo_hash
    total = len(df)
    n_w_cycle = len(df[df["has_cycle"]])
    print(f"{n_w_cycle} ({(n_w_cycle * 100 / total):.2f}%) AS Sets have cycles.")

    df = df_wo_hash[df_wo_hash["n_sets"] > 0]
    total = len(df)
    n_w_cycle = len(df[df["has_cycle"]])
    print(
        f"{n_w_cycle} ({(n_w_cycle * 100 / total):.2f}%) have cycles among {total} AS Sets containing other AS Sets."
    )


main() if __name__ == "__main__" else None
