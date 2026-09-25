import urllib.request
import ssl
import pandas as pd
import io

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://www.ifremer.fr/erddap/tabledap/ArgoFloats.csv?time,latitude,longitude,pres,temp&time>=2024-08-20T00:00:00Z&time<=2024-08-20T23:59:59Z&latitude>=5&latitude<=30&longitude>=45&longitude<=105"

try:
    response = urllib.request.urlopen(url, context=ctx)
    data = response.read().decode('utf-8')
    df = pd.read_csv(io.StringIO(data), skiprows=[1]) # ERDDAP puts units on line 2
    print(f"Success! Fetched {len(df)} rows.")
    print(df.head())
    
    # Also count unique floats (usually they have platform_number, but here we can just count unique lat/lon)
    unique_floats = df.drop_duplicates(subset=['latitude', 'longitude'])
    print(f"Unique float profiles: {len(unique_floats)}")
except Exception as e:
    print(f"Error: {e}")
