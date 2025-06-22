# Overview

This directory contains datasets or modeling and forecasting key macroeconomic indicators (CPI YoY and unemployment rate) for Hong Kong and Taiwan. To benchmark model performance, [`analyst_forecasts.csv`](./analyst_forecasts.csv) includes analyst forecasts sourced from Bloomberg.

# Data source

## Hong Kong

Data are collected from the [Census and Statistics Department of Hong Kong](https://www.censtatd.gov.hk/en/). The script ([`download_data.py`](../src/dataprep/download_data.py)) automates data retrieval through the department's API and performs basic preprocessing.

## Taiwan

Data are collected from the [National Statistics of Taiwan](https://eng.stat.gov.tw/Default.aspx). While data is comprehensive, the official download interface is only available in **Traditional Chinese**.

## Analyst forecasts

Analyst forecasts for the target indicators were obtained from Bloomberg Terminal as of the end of 2022.