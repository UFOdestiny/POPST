"""Small raw CSV/parquet fixtures exercise every city Flow/OD pipeline."""

import importlib
import json
from pathlib import Path
import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from config.paths import ROOT

GEO = {
    "dc": "Neighborhood_Clusters.geojson",
    "sf": "SF Analysis Neighborhoods.geojson",
    "nyc": "NYC Taxi Zones.geojson",
    "chicago": "Chicago Community Areas.geojson",
}


@pytest.mark.parametrize("city", GEO)
def test_download_uses_explicit_raw_directory(city, tmp_path, monkeypatch):
    import requests
    monkeypatch.setattr(requests, "get", lambda *a, **k: pytest.fail("No downloads were requested"))
    raw = tmp_path / "raw"
    module = importlib.import_module(f"data.cities.{city}")
    ctx = module.process_download(
        {"downloads": [], "year": 2025}, raw_root=raw,
        asset_root=tmp_path / "assets", processed_root=tmp_path / "processed",
    )
    assert Path(ctx["DIR"]).parent == raw
    assert not (tmp_path / "processed").exists()


@pytest.mark.parametrize("city", GEO)
@pytest.mark.parametrize("task", ["flow", "od"])
def test_raw_to_dataset_and_adjacency(city, task, tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    assets = tmp_path / "assets"
    processed = tmp_path / "processed"
    geo = gpd.read_file(ROOT / "notebooks/assets/geo" / GEO[city]).to_crs("EPSG:4326")
    if city == "nyc":
        geo = geo[geo.borough == "Manhattan"]
    geo = geo.head(4).copy()
    (assets / "geo").mkdir(parents=True)
    geo.to_file(assets / "geo" / GEO[city], driver="GeoJSON")
    points = geo.geometry.representative_point()
    x0, y0, x1, y1 = points.iloc[0].x, points.iloc[0].y, points.iloc[1].x, points.iloc[1].y
    name = {"dc": "DC", "sf": "SF", "nyc": "NYC", "chicago": "Chicago"}[city]
    folder = raw / name

    def write(relative, record, parquet=False):
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame = pd.DataFrame([record, record])  # two trips, same O/D/time bin
        if parquet:
            frame.to_parquet(path, index=False)
        else:
            frame.to_csv(path, index=False)

    time = "2025-01-01 02:00:00"
    bike = dict(start_lng=x0, start_lat=y0, end_lng=x1, end_lat=y1, started_at=time, ended_at=time)
    provider = {"dc": "capitalbikeshare", "sf": "baywheels", "nyc": "citibike", "chicago": "divvy"}[
        city
    ]
    write(f"bike/202501-{provider}-tripdata.csv", bike)
    if city == "dc":
        write(
            "taxi/taxi_2025_01.csv",
            dict(
                ORIGIN_BLOCK_LONGITUDE=x0,
                ORIGIN_BLOCK_LATITUDE=y0,
                DESTINATION_BLOCK_LONG=x1,
                DESTINATION_BLOCK_LAT=y1,
                ORIGINDATETIME_TR="01/01/2025 02:00",
                DESTINATIONDATETIME_TR="01/01/2025 02:00",
            ),
        )
    elif city == "sf":
        write(
            "sf_taxi_2025.csv",
            dict(
                pickup_location_longitude=x0,
                pickup_location_latitude=y0,
                dropoff_location_longitude=x1,
                dropoff_location_latitude=y1,
                start_time_local=time,
                end_time_local=time,
            ),
        )
    elif city == "nyc":
        loc = dict(
            PULocationID=int(geo.LocationID.iloc[0]), DOLocationID=int(geo.LocationID.iloc[1])
        )
        write(
            "taxi/yellow_tripdata_2025-01.parquet",
            dict(**loc, tpep_pickup_datetime=time, tpep_dropoff_datetime=time),
            True,
        )
        write(
            "taxi/fhvhv_tripdata_2025-01.parquet",
            dict(**loc, pickup_datetime=time, dropoff_datetime=time),
            True,
        )
    else:
        record = dict(
            pickup_community_area=int(geo.area_numbe.iloc[0]),
            dropoff_community_area=int(geo.area_numbe.iloc[1]),
            trip_start_timestamp=time,
            trip_end_timestamp=time,
        )
        write("chi_taxi_2025_15min.csv", record)
        write("TNP/chi_TNP_2025_01_15min.csv", record)
        write(
            "chi_scooter_2025_60min.csv",
            dict(
                start_community_area_number=record["pickup_community_area"],
                end_community_area_number=record["dropoff_community_area"],
                start_time=time,
                end_time=time,
            ),
        )
    module = importlib.import_module(f"data.cities.{city}")
    context = getattr(module, f"process_{task}")(
        dict(year=2025, start="2025-01-01", end="2025-01-04", horizons=[1]),
        raw_root=raw, asset_root=assets, processed_root=processed,
    )
    directory = processed / context["DATASET"] / "2025_12to1"
    meta = json.loads((directory / "meta.json").read_text())
    from data.preprocessing import reconstruct_scaler

    with np.load(directory / "his.npz") as f:
        restored = reconstruct_scaler(meta).inverse_transform(f["data"])
    channels = 3 if city in ("nyc", "chicago") else 2
    np.testing.assert_allclose(
        restored.sum(axis=tuple(range(restored.ndim - 1))), np.full(channels, 2), atol=1e-6
    )
    assert restored.ndim == (4 if task == "od" else 3)
    assert restored.shape[1] == 4
    for adjacency in (processed / context["DATASET"]).glob("*.npy"):
        adj = np.load(adjacency)
        assert adj.shape == (4, 4) and np.isfinite(adj).all()
    assert list((processed / context["DATASET"]).glob("*.npy"))
    for split in ["train", "val", "test"]:
        assert np.load(directory / f"idx_{split}.npy").size > 0
    # Run the preserved visualization cells against the same tiny processed data.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    context.update(np=np, pd=pd, plt=plt)
    notebook = json.loads(
        (
            ROOT / "notebooks" / name / f"{name}_{task.upper() if task == 'od' else 'Flow'}.ipynb"
        ).read_text()
    )
    for cell in notebook["cells"][3:]:
        if cell["cell_type"] == "code":
            exec(compile("".join(cell["source"]), f"{city}_{task}_plot", "exec"), context)
    plt.close("all")
