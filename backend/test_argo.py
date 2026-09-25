import copernicusmarine
import datetime
import traceback

def test_fetch(dataset_id):
    print(f"Testing {dataset_id}...")
    try:
        copernicusmarine.subset(
            dataset_id=dataset_id,
            variables=["TEMP", "PSAL"], # uppercase in this dataset usually
            start_datetime="2020-08-20T00:00:00",
            end_datetime="2020-08-20T23:59:59",
            minimum_longitude=45.0,
            maximum_longitude=105.0,
            minimum_latitude=5.0,
            maximum_latitude=30.0,
            minimum_depth=0.0,
            maximum_depth=1000.0,
            output_filename=f"test_argo_{dataset_id.replace(':', '_')}.nc",
            output_directory=".",
            force_download=True
        )
        print(f"SUCCESS: {dataset_id}")
    except Exception as e:
        print(f"FAILED: {dataset_id} -> {e}")

test_fetch("cmems_obs-ins_glo_phy-temp-sal_my_cora-oa_P1M")
