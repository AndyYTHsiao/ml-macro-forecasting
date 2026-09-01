import json
import os

import pandas as pd
import requests


def download_hk_econ_data(
    parameters: dict, api_url: str = "https://www.censtatd.gov.hk/api/post.php"
) -> pd.DataFrame:
    """
    Download economic data from the Hong Kong Census and Statistics Department API.

    Args:
        parameters (dict): A dictionary of parameters to be sent in the POST request. Refer to the API documentation for details.
        api_url (str): The API endpoint URL.

    Returns:
        pd.DataFrame: A DataFrame containing the retrieved data.
    """
    data = {"query": json.dumps(parameters)}
    r = requests.post(api_url, data=data, timeout=20)

    if r.status_code == 200:
        data = r.json()
        return pd.DataFrame(data["dataSet"])
    else:
        raise ConnectionError(f"Failed to retrieve data: {r.status_code}")


if __name__ == "__main__":
    output_path = "../data/"
    os.makedirs(output_path, exist_ok=True)

    # --- Download Hong Kong CPI data ---
    hk_cpi_parameters = {
        "cv": {"GROUP": ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"]},
        "sv": {
            "CC_CM_1920": ["MoM_1dp_%_s", "YoY_1dp_%_s"],
            "B_CM_1920": ["MoM_1dp_%_s", "YoY_1dp_%_s"],
            "C_CM_1920": ["MoM_1dp_%_s", "YoY_1dp_%_s"],
            "A_CM_1920": ["MoM_1dp_%_s", "YoY_1dp_%_s"],
        },
        "period": {"start": "199301", "end": "202412"},
        "id": "510-60001A",
        "lang": "en",
    }
    hk_cpi = download_hk_econ_data(hk_cpi_parameters)

    # Missing values are changes less than 0.05%
    hk_cpi["figure"] = hk_cpi["figure"].fillna(0)
    hk_cpi = hk_cpi.loc[hk_cpi["freq"] == "M", :]
    hk_cpi = hk_cpi.drop(columns=["GROUP", "freq", "sd_value"])
    hk_cpi["period"] = (
        pd.to_datetime(hk_cpi["period"], format="%Y%m")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )
    hk_cpi["svDesc"] = hk_cpi["svDesc"].replace(
        {
            "Month-to-month % change": "MoM",
            "Year-on-year % change": "YoY",
        }
    )
    hk_cpi["sv"] = hk_cpi["sv"].replace(
        {
            "CC_CM_1920": "Composite CPI",
            "A_CM_1920": "CPI (A)",
            "B_CM_1920": "CPI (B)",
            "C_CM_1920": "CPI (C)",
        }
    )
    hk_cpi["sv"] = hk_cpi["sv"] + " " + hk_cpi["GROUPDesc"] + " " + hk_cpi["svDesc"]
    hk_cpi = hk_cpi.drop(columns=["sv", "GROUPDesc", "svDesc"])
    hk_cpi = hk_cpi.pivot(index="period", columns="sv", values="figure")
    hk_cpi.columns.name = None
    order_prefix = [
        "Composite CPI Total",
        "CPI (A) Total",
        "CPI (B) Total",
        "CPI (C) Total",
        "Composite CPI Food",
        "CPI (A) Food",
        "CPI (B) Food",
        "CPI (C) Food",
        "Composite CPI Housing",
        "CPI (A) Housing",
        "CPI (B) Housing",
        "CPI (C) Housing",
        "Composite CPI Electricity, gas and water",
        "CPI (A) Electricity, gas and water",
        "CPI (B) Electricity, gas and water",
        "CPI (C) Electricity, gas and water",
        "Composite CPI Alcoholic drinks and tobacco",
        "CPI (A) Alcoholic drinks and tobacco",
        "CPI (B) Alcoholic drinks and tobacco",
        "CPI (C) Alcoholic drinks and tobacco",
        "Composite CPI Clothing and footwear",
        "CPI (A) Clothing and footwear",
        "CPI (B) Clothing and footwear",
        "CPI (C) Clothing and footwear",
        "Composite CPI Durable goods",
        "CPI (A) Durable goods",
        "CPI (B) Durable goods",
        "CPI (C) Durable goods",
        "Composite CPI Miscellaneous goods",
        "CPI (A) Miscellaneous goods",
        "CPI (B) Miscellaneous goods",
        "CPI (C) Miscellaneous goods",
        "Composite CPI Transport",
        "CPI (A) Transport",
        "CPI (B) Transport",
        "CPI (C) Transport",
        "Composite CPI Miscellaneous services",
        "CPI (A) Miscellaneous services",
        "CPI (B) Miscellaneous services",
        "CPI (C) Miscellaneous services",
    ]
    desired_order = []
    for cat in ["YoY", "MoM"]:
        desired_order += [f"{item} {cat}" for item in order_prefix]
    hk_cpi = hk_cpi[desired_order].dropna()

    new_cpi_cols = {}
    for col in hk_cpi.columns:
        # Add lag features for CPI columns
        if "CPI" in col:
            for i in [1, 3, 6, 12]:
                new_cpi_cols[f"{col} {i}m lag"] = hk_cpi[col].shift(i)

        # Add diff features for YoY columns
        if "YoY" in col:
            for i in [3, 6]:
                new_cpi_cols[f"{col} {i}m diff"] = hk_cpi[col].diff(i)

    hk_cpi = pd.concat([hk_cpi, pd.DataFrame(new_cpi_cols, index=hk_cpi.index)], axis=1)
    hk_cpi.to_csv("../data/hk_cpi.csv", index=True)

    # --- Download Hong Kong unemployment data ---
    hk_unemp_parameters = {
        "cv": {"SEX": ["M", "F"]},
        "sv": {
            "UDR": ["Rate_1dp_%_n"],
            "LFPR": ["Rate_1dp_%_n"],
            "UR": ["Rate_1dp_%_n"],
            "SAUR": ["Rate_1dp_%_n"],
        },
        "period": {"start": "199307", "end": "202412"},
        "id": "210-06101",
        "lang": "en",
    }
    hk_unemp = download_hk_econ_data(hk_unemp_parameters)
    hk_unemp = hk_unemp.loc[hk_unemp["freq"] == "M3M", :]
    hk_unemp["sv"] = hk_unemp["sv"].map(
        {
            "UDR": "Underemployment rate",
            "LFPR": "Labour force participation rate",
            "UR": "Unemployment rate",
        }
    )
    hk_unemp["sv"] = hk_unemp["SEXDesc"] + " " + hk_unemp["sv"].str.lower()
    hk_unemp = hk_unemp[["sv", "period", "figure"]]
    hk_unemp["period"] = (
        pd.to_datetime(hk_unemp["period"], format="%Y%m")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )
    hk_unemp = hk_unemp.pivot(index="period", columns="sv", values="figure")
    hk_unemp.columns.name = None
    new_unemp_cols = {}
    for col in hk_unemp.columns:
        # Add lag features for unemployment rate columns
        if "unemployment rate" in col:
            for i in [1, 3, 6]:
                new_unemp_cols[f"{col} {i}m lag"] = hk_unemp[col].shift(i)

    hk_unemp = pd.concat(
        [hk_unemp, pd.DataFrame(new_unemp_cols, index=hk_unemp.index)], axis=1
    )
    hk_unemp.to_csv("../data/hk_unemp.csv", index=True)
