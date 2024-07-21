import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def read_spreeadsheet(file_path):
    data = pd.read_excel(file_path)
    data["FECHA"] = pd.to_datetime(data["FECHA"], dayfirst=True, errors="coerce")
    data = data[data["FECHA"].notnull()]
    data = data[data["FECHA"] > datetime(2023, 8, 1)]
    return data


def data_preprocessing(data: pd.DataFrame):
    data = data.set_index("FECHA")
    data = data.dropna()
    data = data.sort_index()
    data = data.resample("D").ffill()
    data = data.reset_index()
    # adding dates without data until july 2025

    return data


def forecast_data(m, b, target):

    # proyectar datos hasta julio 2025
    start_date = datetime(2024, 8, 1)
    end_date = datetime(2025, 7, 31)
    date_range = pd.date_range(start_date, end_date, freq="ME")

    df = pd.DataFrame(date_range, columns=["FECHA"])
    df[target] = m * np.array(range(1, len(df) + 1)) + b
    df["NOMBRE_MES_ANNO"] = df["FECHA"].dt.strftime("%B-%Y")
    df = df.drop_duplicates(subset=["NOMBRE_MES_ANNO"])
    df.drop(columns=["FECHA"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def create_features(data: pd.DataFrame):
    df = pd.DataFrame()
    df["COMPRA"] = (
        data.groupby(pd.Series(data["FECHA"].dt.month, name="MONTH"))["COMPRA"]
        .transform("mean")
        .unique()
    )
    df["VENTA"] = (
        data.groupby(pd.Series(data["FECHA"].dt.month, name="MONTH"))["VENTA"]
        .transform("mean")
        .unique()
    )
    df["NOMBRE_MES_ANNO"] = data["FECHA"].dt.strftime("%B-%Y").unique()

    return df


def linear_regression(data: pd.DataFrame, feature: str, target: str):
    import numpy as np

    x = np.array(range(1, len(data[feature]) + 1))
    y = data[target]

    m = (len(x) * np.sum(x * y) - np.sum(x) * np.sum(y)) / (
        len(x) * np.sum(x**2) - np.sum(x) ** 2
    )
    b = (np.sum(y) - m * np.sum(x)) / len(x)

    return m, b


def plot_data(data: pd.DataFrame):
    plt.title("Tipo de cambio CRC COLON - USD DOLAR").figure.set_size_inches(10, 5)
    plt.scatter(data["NOMBRE_MES_ANNO"], data["COMPRA"], label="Compra", color="red")
    plt.scatter(data["NOMBRE_MES_ANNO"], data["VENTA"], label="Venta")

    m, b = linear_regression(data, "NOMBRE_MES_ANNO", "COMPRA")
    df_compra = forecast_data(m, b, "COMPRA")
    print("Compra: y = {}x + {}".format(m, b))

    m, b = linear_regression(data, "NOMBRE_MES_ANNO", "VENTA")
    df_venta = forecast_data(m, b, "VENTA")

    df = pd.merge(df_compra, df_venta, on="NOMBRE_MES_ANNO")
    data = pd.concat([data, df], ignore_index=True)
    print("Venta: y = {}x + {}".format(m, b))

    plt.plot(
        data["NOMBRE_MES_ANNO"], data["COMPRA"], label="Compra (LR)", color="orange"
    )
    plt.plot(data["NOMBRE_MES_ANNO"], data["VENTA"], label="Venta (LR)", color="purple")

    for i, txt in enumerate(data["COMPRA"]):
        plt.annotate(
            f"₡{txt:.2f}",
            (data["NOMBRE_MES_ANNO"][i], data["COMPRA"][i]),
            fontsize=8,
            color="black",
        )

    for i, txt in enumerate(data["VENTA"]):
        plt.annotate(
            f"₡{txt:.2f}",
            (data["NOMBRE_MES_ANNO"][i], data["VENTA"][i]),
            fontsize=8,
            color="black",
        )

    plt.ylabel(
        "Tipo de cambio",
        fontsize=12,
        color="black",
        fontweight="bold",
        labelpad=10,
        rotation=90,
    )

    plt.xlabel(
        "Meses",
        fontsize=15,
        color="black",
        fontweight="bold",
        labelpad=5,
    )

    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.ylim(495, 545)
    plt.xlim(-1, len(data))
    plt.xticks(rotation=90)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = read_spreeadsheet("DATA/usd_crc_exchange.xlsx")
    data = data_preprocessing(data)
    data = create_features(data)
    plot_data(data)
