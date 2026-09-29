from data.downloads import download_from_api

from data.spatial import make_step_idx_fn, map_points_to_regions, accumulate_nt, accumulate_od
"""SF raw data download and Flow/OD preparation."""

from data.prepare import prepare_array


def process_flow(options=None, *, raw_root, asset_root, processed_root):
    options = dict(options or {})
    from pathlib import Path
    import os
    import numpy as np
    import pandas as pd
    import geopandas as geopd

    DIR = str(Path(raw_root))
    DIR = os.path.join(DIR, "SF")
    TAXI_FILE = Path(DIR) / f"sf_taxi_{options.get('year', 2023)}.csv"
    BIKE_DIR = Path(DIR) / "bike"
    FLOW_MODE = options.get("flow_mode", "arrival")
    assert FLOW_MODE in {"arrival", "departure"}
    NHOODS_PATH = str(Path(asset_root) / "geo" / "SF Analysis Neighborhoods.geojson")
    OUT_DIR = Path(processed_root) / "intermediate" / "SF"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_NHOODS = options.get("target_nhoods", None)
    TARGET_TAG = options.get("target_tag", "subset")
    YEAR = options.get("year", 2023)
    FREQ = options.get("freq", "15min")
    TIME_START = pd.Timestamp(options.get("start", f"{YEAR}-01-01 00:00:00"))
    TIME_END = pd.Timestamp(options.get("end", f"{YEAR + 1}-01-01 00:00:00"))
    STEPS = pd.date_range(TIME_START, TIME_END, freq=FREQ, inclusive="left")
    T = len(STEPS)
    FREQ_TAG = FREQ
    print(f"YEAR={YEAR}, FREQ={FREQ}, T={T} steps, FLOW_MODE={FLOW_MODE}")
    nhoods = geopd.read_file(NHOODS_PATH).to_crs("EPSG:4326")
    nhoods = nhoods.sort_values("nhood").reset_index(drop=True)
    nhoods["region_idx"] = np.arange(len(nhoods), dtype=np.int32)
    nhoods_for_join = nhoods[["region_idx", "geometry"]]
    N = len(nhoods)
    bounds = nhoods.total_bounds
    name_to_region = {str(nm): idx for idx, nm in enumerate(nhoods["nhood"].tolist())}
    print(f"N={N} analysis neighborhoods, bbox={bounds}")
    step_idx_15 = make_step_idx_fn(FREQ, T, time_start=TIME_START)

    def process_gps_csv(
        files, lon_col, lat_col, time_col, n_t, step_idx_fn, dt_fmt=None, chunksize=500000
    ):
        """Bin a single trip endpoint (lon/lat + time) into an (N, n_t) flow matrix.

        Used for both SF mobilities since taxi and bike are both GPS-based.
        """
        nt = np.zeros((N, n_t), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        for fp in files:
            print(f"Processing {fp.name} [{lon_col}, {lat_col}, {time_col}]")
            reader = pd.read_csv(
                fp, usecols=[lon_col, lat_col, time_col], chunksize=chunksize, low_memory=False
            )
            for chunk in reader:
                total_rows += len(chunk)
                chunk = chunk.dropna(subset=[lon_col, lat_col, time_col])
                if chunk.empty:
                    continue
                lon = pd.to_numeric(chunk[lon_col], errors="coerce").to_numpy(dtype=np.float64)
                lat = pd.to_numeric(chunk[lat_col], errors="coerce").to_numpy(dtype=np.float64)
                valid_xy = np.isfinite(lon) & np.isfinite(lat)
                if not np.any(valid_xy):
                    continue
                lon = lon[valid_xy]
                lat = lat[valid_xy]
                ts = chunk.loc[valid_xy, time_col].reset_index(drop=True)
                step_idx, time_valid = step_idx_fn(ts, fmt=dt_fmt)
                region_idx = map_points_to_regions(lon, lat, nhoods_for_join, bounds)
                kept_rows += accumulate_nt(region_idx, step_idx, time_valid, nt, n_t)
        print(f"Rows seen: {total_rows:,}, kept in (N,T): {kept_rows:,}")
        return nt

    if FLOW_MODE == "arrival":
        taxi_lon, taxi_lat, taxi_time = (
            "dropoff_location_longitude",
            "dropoff_location_latitude",
            "end_time_local",
        )
    else:
        taxi_lon, taxi_lat, taxi_time = (
            "pickup_location_longitude",
            "pickup_location_latitude",
            "start_time_local",
        )
    if not TAXI_FILE.exists():
        raise FileNotFoundError(f"SF taxi file not found: {TAXI_FILE}")
    sf_taxi_nt = process_gps_csv([TAXI_FILE], taxi_lon, taxi_lat, taxi_time, T, step_idx_15)
    taxi_out = OUT_DIR / f"sf_taxi_{YEAR}_{FREQ_TAG}.npy"
    np.save(taxi_out, sf_taxi_nt)
    print(f"Saved {taxi_out} shape={sf_taxi_nt.shape}, total={sf_taxi_nt.sum():,}")
    if FLOW_MODE == "arrival":
        bike_lon, bike_lat, bike_time = ("end_lng", "end_lat", "ended_at")
    else:
        bike_lon, bike_lat, bike_time = ("start_lng", "start_lat", "started_at")
    bike_files = sorted(BIKE_DIR.glob(f"{YEAR}??-baywheels-tripdata.csv"))
    if not bike_files:
        raise FileNotFoundError(f"No Bay Wheels CSV files found under {BIKE_DIR}")
    sf_bike_nt = process_gps_csv(bike_files, bike_lon, bike_lat, bike_time, T, step_idx_15)
    bike_out = OUT_DIR / f"sf_bike_{YEAR}_{FREQ_TAG}.npy"
    np.save(bike_out, sf_bike_nt)
    print(f"Saved {bike_out} shape={sf_bike_nt.shape}, total={sf_bike_nt.sum():,}")

    def summarize_nt(name, arr):
        arr = np.asarray(arr)
        total = float(arr.sum())
        nnz = int(np.count_nonzero(arr))
        density = 100.0 * nnz / arr.size if arr.size else 0.0
        active_regions = int((arr.sum(axis=1) > 0).sum())
        active_steps = int((arr.sum(axis=0) > 0).sum())
        print(
            f"[{name}] total={total:,.0f}, nnz={nnz:,} ({density:.3f}%), active_regions={active_regions}/{arr.shape[0]}, active_steps={active_steps}/{arr.shape[1]}"
        )

    for name, arr in [("taxi", sf_taxi_nt), ("bike", sf_bike_nt)]:
        summarize_nt(name, arr)
    if TARGET_NHOODS is not None and (
        not isinstance(TARGET_NHOODS, str) or TARGET_NHOODS.lower() != "all"
    ):
        target_set = {str(c) for c in TARGET_NHOODS}
        mask = nhoods["nhood"].astype(str).isin(target_set)
        sel_idx = np.where(mask.to_numpy())[0]
        if sel_idx.size == 0:
            raise ValueError(f"No neighborhoods matched TARGET_NHOODS={TARGET_NHOODS}")
        nhoods_sel = nhoods.iloc[sel_idx].reset_index(drop=True)
        nhood_tag = TARGET_TAG.lower().replace(" ", "_")
        geo_path = OUT_DIR / f"{nhood_tag}.geojson"
        nhoods_sel.drop(columns=["region_idx"], errors="ignore").to_file(geo_path, driver="GeoJSON")
        print(f"TARGET_NHOODS kept {len(sel_idx)}/{N} neighborhoods -> {geo_path}")
        for name, arr in [("taxi", sf_taxi_nt), ("bike", sf_bike_nt)]:
            sub = arr[sel_idx]
            out = OUT_DIR / f"sf_{nhood_tag}_{name}_{YEAR}_{FREQ_TAG}.npy"
            np.save(out, sub)
            print(f"  {out} shape={sub.shape}, total={sub.sum():,}")
    else:
        sel_idx = np.arange(N)
        nhoods_sel = nhoods.copy()
        nhood_tag = "all"
        print(f"Neighborhood filter disabled, keeping all {N} neighborhoods")
    MERGE_MOBILITIES = ["taxi", "bike"]
    MOBILITY_ARRS = {"taxi": sf_taxi_nt, "bike": sf_bike_nt}
    missing = [m for m in MERGE_MOBILITIES if m not in MOBILITY_ARRS]
    if missing:
        raise ValueError(f"Unknown mobilities in MERGE_MOBILITIES: {missing}")
    merged = np.stack([MOBILITY_ARRS[m] for m in MERGE_MOBILITIES], axis=1)
    merged_tag = "_".join(MERGE_MOBILITIES)
    merged_out = OUT_DIR / f"sf_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
    np.save(merged_out, merged)
    print(f"{merged_out} shape={merged.shape} (N,M,T)")
    if TARGET_NHOODS is not None and (
        not isinstance(TARGET_NHOODS, str) or TARGET_NHOODS.lower() != "all"
    ):
        merged_c = merged[sel_idx, :, :]
        merged_c_out = OUT_DIR / f"sf_{nhood_tag}_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        np.save(merged_c_out, merged_c)
        print(f"{merged_c_out} shape={merged_c.shape} (N,M,T)")
    if TARGET_NHOODS is not None and (
        not isinstance(TARGET_NHOODS, str) or TARGET_NHOODS.lower() != "all"
    ):
        src_path = OUT_DIR / f"sf_{nhood_tag}_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"sf_{nhood_tag}_{FREQ}"
    else:
        src_path = OUT_DIR / f"sf_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"sf_{FREQ}"
    merged_arr = np.load(src_path)
    M = len(MERGE_MOBILITIES)
    if merged_arr.shape[0] == M and merged_arr.shape[1] != M:
        ndt = merged_arr.transpose(1, 0, 2)
    elif merged_arr.shape[1] == M:
        ndt = merged_arr
    else:
        raise ValueError(f"Cannot locate mobility axis (M={M}) in {merged_arr.shape}")
    ndt_path = src_path.with_name(src_path.stem + "_NDT.npy")
    np.save(ndt_path, ndt)
    print(f"{ndt_path} shape={tuple(ndt.shape)} (N, D, T) with D={M} mobilities")
    for hy in options.get("horizons", [1, 3, 6, 9, 12]):
        prepare_array(
            output_root=processed_root,
            data_path=str(ndt_path),
            fmt="NDT",
            clip_neg=True,
            per_channel=True,
            log1p=True,
            dataset=DATASET,
            years=f"{YEAR}_12to{hy}",
            seq_length_x=int("12"),
            seq_length_y=int(str(hy)),
        )
    from data.graph import get_adjacency_matrix

    ADJ_DIR = Path(processed_root) / DATASET
    ADJ_DIR.mkdir(parents=True, exist_ok=True)
    ADJ_OUT = ADJ_DIR / "sf.npy"
    adj_gdf = nhoods_sel
    ctr = adj_gdf.set_geometry("geometry").to_crs("EPSG:3310").centroid.reset_index(drop=True)
    N_adj = len(ctr)
    ids = list(range(N_adj))
    distance = [[i, j, ctr[i].distance(ctr[j])] for i in ids for j in ids]
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ids)
    np.save(ADJ_OUT, adj_mx)
    print(f"Saved {ADJ_OUT} shape={adj_mx.shape}")
    return locals()


def process_od(options=None, *, raw_root, asset_root, processed_root):
    options = dict(options or {})
    from pathlib import Path
    import os
    import numpy as np
    import pandas as pd
    import geopandas as geopd

    DIR = str(Path(raw_root))
    DIR = os.path.join(DIR, "SF")
    TAXI_FILE = Path(DIR) / f"sf_taxi_{options.get('year', 2023)}.csv"
    BIKE_DIR = Path(DIR) / "bike"
    FLOW_MODE = options.get("flow_mode", "departure")
    assert FLOW_MODE in {"arrival", "departure"}
    NHOODS_PATH = str(Path(asset_root) / "geo" / "SF Analysis Neighborhoods.geojson")
    OUT_DIR = Path(processed_root) / "intermediate" / "SF"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_NHOODS = options.get("target_nhoods", None)
    TARGET_TAG = options.get("target_tag", "subset")
    YEAR = options.get("year", 2023)
    FREQ = options.get("freq", "15min")
    TIME_START = pd.Timestamp(options.get("start", f"{YEAR}-01-01 00:00:00"))
    TIME_END = pd.Timestamp(options.get("end", f"{YEAR + 1}-01-01 00:00:00"))
    STEPS = pd.date_range(TIME_START, TIME_END, freq=FREQ, inclusive="left")
    T = len(STEPS)
    FREQ_TAG = FREQ
    print(f"YEAR={YEAR}, FREQ={FREQ}, T={T} steps, FLOW_MODE={FLOW_MODE}")
    nhoods = geopd.read_file(NHOODS_PATH).to_crs("EPSG:4326")
    nhoods = nhoods.sort_values("nhood").reset_index(drop=True)
    nhoods["region_idx"] = np.arange(len(nhoods), dtype=np.int32)
    nhoods_for_join = nhoods[["region_idx", "geometry"]]
    N = len(nhoods)
    bounds = nhoods.total_bounds
    name_to_region = {str(nm): idx for idx, nm in enumerate(nhoods["nhood"].tolist())}
    print(f"N={N} analysis neighborhoods, bbox={bounds}")
    step_idx_15 = make_step_idx_fn(FREQ, T, time_start=TIME_START)

    def process_gps_od_csv(
        files,
        o_lon_c,
        o_lat_c,
        d_lon_c,
        d_lat_c,
        time_col,
        n_t,
        step_idx_fn,
        dt_fmt=None,
        chunksize=500000,
    ):
        """OD binning for CSVs whose origin & destination are both lon/lat points."""
        od = np.zeros((N, N, n_t), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        usecols = [o_lon_c, o_lat_c, d_lon_c, d_lat_c, time_col]
        for fp in files:
            print(
                f"Processing {fp.name} [O=({o_lon_c},{o_lat_c}), D=({d_lon_c},{d_lat_c}), t={time_col}]"
            )
            reader = pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                total_rows += len(chunk)
                chunk = chunk.dropna(subset=usecols)
                if chunk.empty:
                    continue
                o_lon = pd.to_numeric(chunk[o_lon_c], errors="coerce").to_numpy(dtype=np.float64)
                o_lat = pd.to_numeric(chunk[o_lat_c], errors="coerce").to_numpy(dtype=np.float64)
                d_lon = pd.to_numeric(chunk[d_lon_c], errors="coerce").to_numpy(dtype=np.float64)
                d_lat = pd.to_numeric(chunk[d_lat_c], errors="coerce").to_numpy(dtype=np.float64)
                valid_xy = (
                    np.isfinite(o_lon)
                    & np.isfinite(o_lat)
                    & np.isfinite(d_lon)
                    & np.isfinite(d_lat)
                )
                if not np.any(valid_xy):
                    continue
                o_lon, o_lat = (o_lon[valid_xy], o_lat[valid_xy])
                d_lon, d_lat = (d_lon[valid_xy], d_lat[valid_xy])
                ts = chunk.loc[valid_xy, time_col].reset_index(drop=True)
                o_region = map_points_to_regions(o_lon, o_lat, nhoods_for_join, bounds)
                d_region = map_points_to_regions(d_lon, d_lat, nhoods_for_join, bounds)
                step_idx, time_valid = step_idx_fn(ts, fmt=dt_fmt)
                kept_rows += accumulate_od(o_region, d_region, step_idx, time_valid, od, n_t)
        print(f"Rows seen: {total_rows:,}, kept in OD: {kept_rows:,}")
        return od

    taxi_time_col = "end_time_local" if FLOW_MODE == "arrival" else "start_time_local"
    if not TAXI_FILE.exists():
        raise FileNotFoundError(f"SF taxi file not found: {TAXI_FILE}")
    sf_taxi_od = process_gps_od_csv(
        [TAXI_FILE],
        "pickup_location_longitude",
        "pickup_location_latitude",
        "dropoff_location_longitude",
        "dropoff_location_latitude",
        taxi_time_col,
        T,
        step_idx_15,
    )
    taxi_out = OUT_DIR / f"sf_taxi_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(taxi_out, sf_taxi_od)
    print(f"Saved {taxi_out} shape={sf_taxi_od.shape} (O,D,T), total={sf_taxi_od.sum():,}")
    bike_time_col = "ended_at" if FLOW_MODE == "arrival" else "started_at"
    bike_files = sorted(BIKE_DIR.glob(f"{YEAR}??-baywheels-tripdata.csv"))
    if not bike_files:
        raise FileNotFoundError(f"No Bay Wheels CSV files found under {BIKE_DIR}")
    sf_bike_od = process_gps_od_csv(
        bike_files, "start_lng", "start_lat", "end_lng", "end_lat", bike_time_col, T, step_idx_15
    )
    bike_out = OUT_DIR / f"sf_bike_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(bike_out, sf_bike_od)
    print(f"Saved {bike_out} shape={sf_bike_od.shape} (O,D,T), total={sf_bike_od.sum():,}")

    def summarize_od(name, arr):
        arr = np.asarray(arr)
        total = float(arr.sum())
        nnz = int(np.count_nonzero(arr))
        density = 100.0 * nnz / arr.size if arr.size else 0.0
        pair_flow = arr.sum(axis=2)
        active_pairs = int((pair_flow > 0).sum())
        active_o = int((arr.sum(axis=(1, 2)) > 0).sum())
        active_d = int((arr.sum(axis=(0, 2)) > 0).sum())
        active_steps = int((arr.sum(axis=(0, 1)) > 0).sum())
        print(
            f"[{name}] shape={arr.shape}, total={total:,.0f}, nnz={nnz:,} ({density:.3f}%), active_pairs={active_pairs}/{arr.shape[0] * arr.shape[1]}, active_O={active_o}/{arr.shape[0]}, active_D={active_d}/{arr.shape[1]}, active_steps={active_steps}/{arr.shape[2]}"
        )

    for name, arr in [("taxi", sf_taxi_od), ("bike", sf_bike_od)]:
        summarize_od(name, arr)
    if TARGET_NHOODS is not None and (
        not isinstance(TARGET_NHOODS, str) or TARGET_NHOODS.lower() != "all"
    ):
        target_set = {str(c) for c in TARGET_NHOODS}
        mask = nhoods["nhood"].astype(str).isin(target_set)
        sel_idx = np.where(mask.to_numpy())[0]
        if sel_idx.size == 0:
            raise ValueError(f"No neighborhoods matched TARGET_NHOODS={TARGET_NHOODS}")
        nhoods_sel = nhoods.iloc[sel_idx].reset_index(drop=True)
        nhood_tag = TARGET_TAG.lower().replace(" ", "_")
        geo_path = OUT_DIR / f"{nhood_tag}.geojson"
        nhoods_sel.drop(columns=["region_idx"], errors="ignore").to_file(geo_path, driver="GeoJSON")
        print(f"TARGET_NHOODS kept {len(sel_idx)}/{N} neighborhoods -> {geo_path}")
        for name, arr in [("taxi", sf_taxi_od), ("bike", sf_bike_od)]:
            sub = arr[np.ix_(sel_idx, sel_idx)]
            out = OUT_DIR / f"sf_{nhood_tag}_{name}_od_{YEAR}_{FREQ_TAG}.npy"
            np.save(out, sub)
            print(f"  {out} shape={sub.shape}, total={sub.sum():,}")
    else:
        sel_idx = np.arange(N)
        nhoods_sel = nhoods.copy()
        nhood_tag = "all"
        print(f"Neighborhood filter disabled, keeping all {N} neighborhoods")
    MERGE_MOBILITIES = ["taxi", "bike"]
    MOBILITY_ARRS = {"taxi": sf_taxi_od, "bike": sf_bike_od}
    missing = [m for m in MERGE_MOBILITIES if m not in MOBILITY_ARRS]
    if missing:
        raise ValueError(f"Unknown mobilities in MERGE_MOBILITIES: {missing}")
    merged = np.stack([MOBILITY_ARRS[m] for m in MERGE_MOBILITIES], axis=2)
    merged_tag = "_".join(MERGE_MOBILITIES)
    M_dim = len(MERGE_MOBILITIES)
    merged_out = OUT_DIR / f"sf_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(merged_out, merged)
    print(f"{merged_out} shape={merged.shape} (N, N, M, T) with M={M_dim} mobilities")
    if TARGET_NHOODS is not None and (
        not isinstance(TARGET_NHOODS, str) or TARGET_NHOODS.lower() != "all"
    ):
        merged_c = merged[np.ix_(sel_idx, sel_idx)]
        merged_c_out = OUT_DIR / f"sf_{nhood_tag}_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
        np.save(merged_c_out, merged_c)
        print(f"{merged_c_out} shape={merged_c.shape} (N, N, M, T)")
    for i in MOBILITY_ARRS:
        i_out = OUT_DIR / f"sf_{i}_od_{YEAR}_{FREQ_TAG}.npy"
        np.save(i_out, MOBILITY_ARRS[i])
        print(f"{i_out} shape={MOBILITY_ARRS[i].shape} (N, N, T) with 1 mobilities")
    if TARGET_NHOODS is not None and (
        not isinstance(TARGET_NHOODS, str) or TARGET_NHOODS.lower() != "all"
    ):
        src_path = OUT_DIR / f"sf_{nhood_tag}_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"sf_{nhood_tag}_od_{FREQ}"
    else:
        src_path = OUT_DIR / f"sf_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"sf_od_{FREQ}"
    print(f"Source {src_path} shape={tuple(np.load(src_path, mmap_mode='r').shape)} (N, N, M, T)")
    for hy in options.get("horizons", [1, 3, 6, 9, 12]):
        prepare_array(
            output_root=processed_root,
            data_path=str(src_path),
            fmt="NNDT",
            clip_neg=True,
            per_channel=True,
            log1p=True,
            dataset=DATASET,
            years=f"{YEAR}_12to{hy}",
            seq_length_x=int("12"),
            seq_length_y=int(str(hy)),
        )
    DATASET_list = []
    for i in MOBILITY_ARRS:
        d = DATASET + "_" + i
        DATASET_list.append(d)
        i_out = OUT_DIR / f"sf_{i}_od_{YEAR}_{FREQ_TAG}.npy"
        np.save(i_out, MOBILITY_ARRS[i])
        print(f"{i_out} shape={MOBILITY_ARRS[i].shape} (N, N, T) with 1 mobilities")
        prepare_array(
            output_root=processed_root,
            data_path=str(i_out),
            fmt="NNT",
            clip_neg=True,
            per_channel=True,
            log1p=True,
            dataset=d,
            years=f"{YEAR}_12to1",
            seq_length_x=int("12"),
            seq_length_y=int("1"),
        )
    from data.graph import get_adjacency_matrix

    ADJ_DIR = Path(processed_root) / DATASET
    ADJ_DIR.mkdir(parents=True, exist_ok=True)
    ADJ_OUT = ADJ_DIR / "sf.npy"
    adj_gdf = nhoods_sel
    ctr = adj_gdf.set_geometry("geometry").to_crs("EPSG:3310").centroid.reset_index(drop=True)
    N_adj = len(ctr)
    ids = list(range(N_adj))
    distance = [[i, j, ctr[i].distance(ctr[j])] for i in ids for j in ids]
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ids)
    np.save(ADJ_OUT, adj_mx)
    for i in DATASET_list:
        ADJ_DIR = Path(processed_root) / i
        ADJ_DIR.mkdir(parents=True, exist_ok=True)
        ADJ_OUT = ADJ_DIR / f"sf.npy"
        np.save(ADJ_OUT, adj_mx)
    print(f"Saved {ADJ_OUT} shape={adj_mx.shape}")
    return locals()


def process_download(options=None, *, raw_root, asset_root, processed_root):
    options = dict(options or {})
    from pathlib import Path
    import os
    import zipfile
    import requests

    DIR = str(Path(raw_root))
    os.makedirs(DIR, exist_ok=True)
    DIR = os.path.join(DIR, "SF")
    os.makedirs(DIR, exist_ok=True)
    taxi_orig_file = os.path.join(DIR, f"sf_taxi_{options.get('year', 2023)}.csv")
    bike_orig_dir = os.path.join(DIR, "bike")

    DOWNLOAD = "taxi" in options.get("downloads", ["taxi", "bike"])
    if DOWNLOAD:
        API_URL = "https://data.sfgov.org/resource/m8hk-2ipk.csv"
        base_where = f"start_time_local >= '{options.get('year', 2023)}-01-01T00:00:00.000' AND start_time_local < '{options.get('year', 2023) + 1}-01-01T00:00:00.000'"
        select_cols = ":id, start_time_local, end_time_local, pickup_location_latitude, pickup_location_longitude, dropoff_location_latitude, dropoff_location_longitude"
        download_from_api(
            API_URL,
            base_where,
            taxi_orig_file,
            order_col="start_time_local",
            tie_col=":id",
            select_cols=select_cols,
            page_size=20000,
        )

    DOWNLOAD = "bike" in options.get("downloads", ["taxi", "bike"])
    year = options.get("year", 2023)
    base_url_template = (
        "https://s3.amazonaws.com/baywheels-data/{year}{month:02d}-baywheels-tripdata.csv.zip"
    )
    os.makedirs(bike_orig_dir, exist_ok=True)
    if DOWNLOAD:
        for month in options.get("months", list(range(1, 13))):
            url = base_url_template.format(year=year, month=month)
            zip_filename = f"{year}{month:02d}-baywheels-tripdata.csv.zip"
            zip_filepath = os.path.join(bike_orig_dir, zip_filename)
            try:
                response = requests.get(url, stream=True, timeout=120)
                response.raise_for_status()
                if response.status_code == 200:
                    with open(zip_filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192 * 1024):
                            if chunk:
                                f.write(chunk)
                    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                        zip_ref.extractall(bike_orig_dir)
                    os.remove(zip_filepath)
            except requests.exceptions.RequestException:
                raise
    return locals()
