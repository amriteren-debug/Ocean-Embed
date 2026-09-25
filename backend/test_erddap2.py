import requests
import pandas as pd
import io

url = "https://www.ifremer.fr/erddap/tabledap/ArgoFloats.csv?platform_number,time,latitude,longitude&time>=2026-08-20T00:00:00Z&time<=2026-08-20T23:59:59Z&latitude>=5&latitude<=30&longitude>=45&longitude<=105"

try:
    response = requests.get(url, verify=False, timeout=10)
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text), skiprows=[1])
    print(f"Success! Fetched {len(df)} rows.")
    print(df.head())
except Exception as e:
    print(f"Error: {e}")
