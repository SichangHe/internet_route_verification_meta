"""Run at `scripts/` with `python3 -m fig.route_all`.
Data are from here:
<https://github.com/SichangHe/internet_route_verification/issues/88>
"""
import pandas as pd
from fig import download_if_missing

FILE = "route_stats1.csv.gz"
PORTS = ("import", "export")
TAGS = ("ok", "skip", "unrec", "meh", "err")


def main():
    download_if_missing(
        "https://github.com/SichangHe/internet_route_verification/releases/download/data-88/route_stats1.csv.gz",
        FILE,
    )
    df = pd.read_csv(FILE, dtype="uint16")
    n_route = len(df)
    df["total"] = sum((df[f"{port}_{tag}"] for tag in TAGS for port in PORTS))
    df_all = {}
    for tag in TAGS:
        df_all[tag] = df[
            df[f"import_{tag}"] + df[f"export_{tag}"] == df["total"]
        ].dropna()
        count = df_all[tag].__len__()
        percentage = count / n_route * 100
        print(f"{count} all {tag}, {percentage}%.")
    for port in PORTS:
        print()
        df[f"total_{port}"] = sum((df[f"{port}_{tag}"] for tag in TAGS))

        df_all[f"{port}_dne"] = df[df[f"total_{port}"] == 0].dropna()
        count = df_all[f"{port}_dne"].__len__()
        percentage = count / n_route * 100
        print(f"{count} have no {port}, {percentage}%")

        for tag in TAGS:
            df_all[f"{port}_{tag}"] = df[
                (df[f"total_{port}"] != 0)
                & (df[f"{port}_{tag}"] == df[f"total_{port}"])
            ].dropna()
            count = df_all[f"{port}_{tag}"].__len__()
            percentage = count / n_route * 100
            print(f"{count} all {tag} in {port}, {percentage}%.")


if __name__ == "__main__":
    main()