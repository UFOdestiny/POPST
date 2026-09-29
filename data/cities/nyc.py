"""NYC raw data download and Flow/OD preparation."""

from data.spatial import map_points_to_regions
from data.prepare import prepare_array


def process_flow(options=None, *, raw_root, asset_root, processed_root):
    options = dict(options or {})
    from pathlib import Path
    import os
    import numpy as np
    import pandas as pd
    import geopandas as geopd

    DIR = str(Path(raw_root))
    DIR = os.path.join(DIR, "NYC")
    TAXI_DIR = Path(DIR) / "taxi"
    BIKE_DIR = Path(DIR) / "bike"
    FLOW_MODE = options.get("flow_mode", "arrival")
    assert FLOW_MODE in {"arrival", "departure"}
    ZONES_PATH = str(Path(asset_root) / "geo" / "NYC Taxi Zones.geojson")
    OUT_DIR = Path(processed_root) / "intermediate" / "NYC"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BOROUGH_GEOJSON_PATH = "./Manhattan.geojson"
    TARGET_BOROUGH = options.get("target_borough", "Manhattan")
    YEAR = options.get("year", 2025)
    FREQ = options.get("freq", "15min")
    TIME_START = pd.Timestamp(options.get("start", f"{YEAR}-01-01 00:00:00"))
    TIME_END = pd.Timestamp(options.get("end", f"{YEAR + 1}-01-01 00:00:00"))
    STEPS = pd.date_range(TIME_START, TIME_END, freq=FREQ, inclusive="left")
    T = len(STEPS)
    FREQ_TAG = FREQ
    print(f"YEAR={YEAR}, FREQ={FREQ}, T={T} steps, FLOW_MODE={FLOW_MODE}")
    zones = geopd.read_file(ZONES_PATH).to_crs("EPSG:4326")
    zones = zones.sort_values("LocationID").reset_index(drop=True)
    zones["region_idx"] = np.arange(len(zones), dtype=np.int32)
    zones_for_join = zones[["region_idx", "geometry"]]
    N = len(zones)
    bounds = zones.total_bounds
    locationid_to_region = {
        int(loc): idx for idx, loc in enumerate(zones["LocationID"].tolist()) if pd.notna(loc)
    }
    print(f"N={N} regions, bbox={bounds}")

    def timestamps_to_step_idx(ts_series):
        ts = pd.to_datetime(ts_series, errors="coerce")
        step = ts.dt.floor(FREQ)
        delta_minutes = (step - TIME_START) / pd.Timedelta(FREQ)
        idx_float = delta_minutes.to_numpy(dtype=np.float64)
        valid = np.isfinite(idx_float) & (idx_float >= 0) & (idx_float < T)
        step_idx = np.zeros(len(idx_float), dtype=np.int64)
        step_idx[valid] = idx_float[valid].astype(np.int64)
        return (step_idx, valid)

    def accumulate_nt(region_idx, step_idx, time_valid, nt_matrix):
        valid = time_valid & (region_idx >= 0)
        if not np.any(valid):
            return 0
        flat_idx = region_idx[valid].astype(np.int64) * T + step_idx[valid].astype(np.int64)
        binc = np.bincount(flat_idx, minlength=N * T)
        nt_matrix += binc.reshape(N, T)
        return int(valid.sum())

    def process_taxi_parquet(files, schema_map):
        """schema_map: {pattern_prefix: {'pu_id','do_id','pu_time','do_time'}}."""
        import pyarrow.parquet as pq

        nt = np.zeros((N, T), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        for fp in files:
            prefix = next((k for k in schema_map if fp.name.startswith(k)), None)
            if prefix is None:
                print(f"  skip (no schema): {fp.name}")
                continue
            cols = schema_map[prefix]
            if FLOW_MODE == "arrival":
                loc_col, time_col = (cols["do_id"], cols["do_time"])
            else:
                loc_col, time_col = (cols["pu_id"], cols["pu_time"])
            print(f"Processing {fp.name} [{loc_col}, {time_col}]")
            pq_file = pq.ParquetFile(fp)
            for rg in range(pq_file.num_row_groups):
                table = pq_file.read_row_group(rg, columns=[loc_col, time_col])
                df = table.to_pandas()
                total_rows += len(df)
                df = df.dropna(subset=[loc_col, time_col])
                if df.empty:
                    continue
                loc = pd.to_numeric(df[loc_col], errors="coerce")
                valid_loc = loc.notna()
                if not valid_loc.any():
                    continue
                loc = loc[valid_loc].astype(np.int64).reset_index(drop=True)
                ts = df.loc[valid_loc, time_col].reset_index(drop=True)
                region_idx = loc.map(locationid_to_region).fillna(-1).to_numpy(dtype=np.int32)
                step_idx, time_valid = timestamps_to_step_idx(ts)
                kept_rows += accumulate_nt(region_idx, step_idx, time_valid, nt)
        print(f"Rows seen: {total_rows:,}, kept in (N,T): {kept_rows:,}")
        return nt

    taxi_schema = {
        "yellow_tripdata": {
            "pu_id": "PULocationID",
            "do_id": "DOLocationID",
            "pu_time": "tpep_pickup_datetime",
            "do_time": "tpep_dropoff_datetime",
        },
        "green_tripdata": {
            "pu_id": "PULocationID",
            "do_id": "DOLocationID",
            "pu_time": "lpep_pickup_datetime",
            "do_time": "lpep_dropoff_datetime",
        },
    }
    taxi_files = sorted(
        [p for p in TAXI_DIR.glob(f"yellow_tripdata_{YEAR}-*.parquet")]
        + [p for p in TAXI_DIR.glob(f"green_tripdata_{YEAR}-*.parquet")]
    )
    if not taxi_files:
        raise FileNotFoundError(f"No yellow/green taxi parquet files found under {TAXI_DIR}")
    nyc_taxi_nt = process_taxi_parquet(taxi_files, taxi_schema)
    taxi_out = OUT_DIR / f"nyc_taxi_{YEAR}_{FREQ_TAG}.npy"
    np.save(taxi_out, nyc_taxi_nt)
    print(f"Saved {taxi_out} shape={nyc_taxi_nt.shape}, total={nyc_taxi_nt.sum():,}")
    fhv_schema = {
        "fhvhv_tripdata": {
            "pu_id": "PULocationID",
            "do_id": "DOLocationID",
            "pu_time": "pickup_datetime",
            "do_time": "dropoff_datetime",
        },
        "fhv_tripdata": {
            "pu_id": "PUlocationID",
            "do_id": "DOlocationID",
            "pu_time": "pickup_datetime",
            "do_time": "dropOff_datetime",
        },
    }
    fhv_schema_ordered = {k: fhv_schema[k] for k in sorted(fhv_schema, key=len, reverse=True)}
    fhv_files = sorted(
        [p for p in TAXI_DIR.glob(f"fhvhv_tripdata_{YEAR}-*.parquet")]
        + [p for p in TAXI_DIR.glob(f"fhv_tripdata_{YEAR}-*.parquet")]
    )
    if not fhv_files:
        raise FileNotFoundError(f"No fhv/fhvhv parquet files found under {TAXI_DIR}")
    nyc_fhv_nt = process_taxi_parquet(fhv_files, fhv_schema_ordered)
    fhv_out = OUT_DIR / f"nyc_fhv_{YEAR}_{FREQ_TAG}.npy"
    np.save(fhv_out, nyc_fhv_nt)
    print(f"Saved {fhv_out} shape={nyc_fhv_nt.shape}, total={nyc_fhv_nt.sum():,}")

    def process_bike_csv(files, chunksize=500000):
        nt = np.zeros((N, T), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        if FLOW_MODE == "arrival":
            lon_col, lat_col, time_col = ("end_lng", "end_lat", "ended_at")
        else:
            lon_col, lat_col, time_col = ("start_lng", "start_lat", "started_at")
        for fp in files:
            print(f"Processing {fp.name}")
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
                step_idx, time_valid = timestamps_to_step_idx(ts)
                region_idx = map_points_to_regions(lon, lat, zones_for_join, bounds)
                kept_rows += accumulate_nt(region_idx, step_idx, time_valid, nt)
        print(f"Rows seen: {total_rows:,}, kept in (N,T): {kept_rows:,}")
        return nt

    bike_files = sorted(BIKE_DIR.glob(f"{YEAR}??-citibike-tripdata*.csv"))
    if not bike_files:
        raise FileNotFoundError(f"No bike CSV files found under {BIKE_DIR}")
    nyc_bike_nt = process_bike_csv(bike_files)
    bike_out = OUT_DIR / f"nyc_bike_{YEAR}_{FREQ_TAG}.npy"
    np.save(bike_out, nyc_bike_nt)
    print(f"Saved {bike_out} shape={nyc_bike_nt.shape}, total={nyc_bike_nt.sum():,}")

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

    for name, arr in [("taxi", nyc_taxi_nt), ("fhv", nyc_fhv_nt), ("bike", nyc_bike_nt)]:
        summarize_nt(name, arr)
    if TARGET_BOROUGH and str(TARGET_BOROUGH).lower() != "all":
        mask = zones["borough"].astype(str).str.lower() == TARGET_BOROUGH.lower()
        sel_idx = np.where(mask.to_numpy())[0]
        if sel_idx.size == 0:
            raise ValueError(f"No zones found for borough={TARGET_BOROUGH}")
        zones_sel = zones.iloc[sel_idx].reset_index(drop=True)
        borough_tag = TARGET_BOROUGH.lower().replace(" ", "_")
        bgeo_path = OUT_DIR / f"{borough_tag}.geojson"
        zones_sel.drop(columns=["region_idx"], errors="ignore").to_file(bgeo_path, driver="GeoJSON")
        print(f"Borough={TARGET_BOROUGH}, kept {len(sel_idx)}/{N} regions -> {bgeo_path}")
        for name, arr in [("taxi", nyc_taxi_nt), ("fhv", nyc_fhv_nt), ("bike", nyc_bike_nt)]:
            sub = arr[sel_idx]
            out = OUT_DIR / f"nyc_{borough_tag}_{name}_{YEAR}_{FREQ_TAG}.npy"
            np.save(out, sub)
            print(f"  {out} shape={sub.shape}, total={sub.sum():,}")
    else:
        sel_idx = np.arange(N)
        zones_sel = zones.copy()
        borough_tag = "all"
        print(f"Borough filter disabled, keeping all {N} regions")
    MERGE_MOBILITIES = ["taxi", "fhv", "bike"]
    MOBILITY_ARRS = {"taxi": nyc_taxi_nt, "fhv": nyc_fhv_nt, "bike": nyc_bike_nt}
    missing = [m for m in MERGE_MOBILITIES if m not in MOBILITY_ARRS]
    if missing:
        raise ValueError(f"Unknown mobilities in MERGE_MOBILITIES: {missing}")
    merged = np.stack([MOBILITY_ARRS[m] for m in MERGE_MOBILITIES], axis=1)
    merged_tag = "_".join(MERGE_MOBILITIES)
    merged_out = OUT_DIR / f"nyc_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
    np.save(merged_out, merged)
    print(f"{merged_out} shape={merged.shape} (N,M,T)")
    if TARGET_BOROUGH and str(TARGET_BOROUGH).lower() != "all":
        merged_b = merged[sel_idx, :, :]
        merged_b_out = OUT_DIR / f"nyc_{borough_tag}_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        np.save(merged_b_out, merged_b)
        print(f"{merged_b_out} shape={merged_b.shape} (N,M,T)")
    if TARGET_BOROUGH and str(TARGET_BOROUGH).lower() != "all":
        src_path = OUT_DIR / f"nyc_{borough_tag}_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"nyc_{borough_tag}_{FREQ}"
    else:
        src_path = OUT_DIR / f"nyc_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"nyc_{FREQ}"
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

    if TARGET_BOROUGH and str(TARGET_BOROUGH).lower() != "all":
        ADJ_OUT = Path(processed_root) / f"nyc_{borough_tag}_{FREQ}" / f"{TARGET_BOROUGH}.npy"
        ADJ_OUT.parent.mkdir(parents=True, exist_ok=True)
        ctr = zones_sel.set_geometry("geometry").centroid.reset_index(drop=True)
        N_adj = len(ctr)
        ids = list(range(N_adj))
        distance = [[i, j, ctr[i].distance(ctr[j])] for i in ids for j in ids]
        adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ids)
        np.save(ADJ_OUT, adj_mx)
        print(f"Saved {ADJ_OUT} shape={adj_mx.shape}")
    else:
        ADJ_OUT = Path(processed_root) / f"nyc_{FREQ}" / "nyc.npy"
        ADJ_OUT.parent.mkdir(parents=True, exist_ok=True)
        ctr = zones.set_geometry("geometry").centroid.reset_index(drop=True)
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
    DIR = os.path.join(DIR, "NYC")
    TAXI_DIR = Path(DIR) / "taxi"
    BIKE_DIR = Path(DIR) / "bike"
    FLOW_MODE = options.get("flow_mode", "departure")
    assert FLOW_MODE in {"arrival", "departure"}
    ZONES_PATH = str(Path(asset_root) / "geo" / "NYC Taxi Zones.geojson")
    OUT_DIR = Path(processed_root) / "intermediate" / "NYC"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_BOROUGH = options.get("target_borough", "Manhattan")
    YEAR = options.get("year", 2025)
    FREQ = options.get("freq", "15min")
    TIME_START = pd.Timestamp(options.get("start", f"{YEAR}-01-01 00:00:00"))
    TIME_END = pd.Timestamp(options.get("end", f"{YEAR + 1}-01-01 00:00:00"))
    STEPS = pd.date_range(TIME_START, TIME_END, freq=FREQ, inclusive="left")
    T = len(STEPS)
    FREQ_TAG = FREQ
    print(f"YEAR={YEAR}, FREQ={FREQ}, T={T} steps, FLOW_MODE={FLOW_MODE}")
    zones = geopd.read_file(ZONES_PATH).to_crs("EPSG:4326")
    zones = zones.sort_values("LocationID").reset_index(drop=True)
    zones["region_idx"] = np.arange(len(zones), dtype=np.int32)
    zones_for_join = zones[["region_idx", "geometry"]]
    N = len(zones)
    bounds = zones.total_bounds
    locationid_to_region = {
        int(loc): idx for idx, loc in enumerate(zones["LocationID"].tolist()) if pd.notna(loc)
    }
    print(f"N={N} regions, bbox={bounds}")
    if TARGET_BOROUGH and str(TARGET_BOROUGH).lower() != "all":
        mask = zones["borough"].astype(str).str.lower() == TARGET_BOROUGH.lower()
        sel_idx = np.where(mask.to_numpy())[0]
        if sel_idx.size == 0:
            raise ValueError(f"No zones found for borough={TARGET_BOROUGH}")
        borough_tag = TARGET_BOROUGH.lower().replace(" ", "_")
    else:
        sel_idx = np.arange(N)
        borough_tag = "all"
    zones_sel = zones.iloc[sel_idx].reset_index(drop=True)
    n_sel = len(sel_idx)
    region_to_local = np.full(N, -1, dtype=np.int32)
    region_to_local[sel_idx] = np.arange(n_sel, dtype=np.int32)
    print(
        f"TARGET_BOROUGH={TARGET_BOROUGH}, selected {n_sel}/{N} regions, OD matrix per step = {n_sel} x {n_sel}"
    )

    def timestamps_to_step_idx(ts_series):
        ts = pd.to_datetime(ts_series, errors="coerce")
        step = ts.dt.floor(FREQ)
        delta_minutes = (step - TIME_START) / pd.Timedelta(FREQ)
        idx_float = delta_minutes.to_numpy(dtype=np.float64)
        valid = np.isfinite(idx_float) & (idx_float >= 0) & (idx_float < T)
        step_idx = np.zeros(len(idx_float), dtype=np.int64)
        step_idx[valid] = idx_float[valid].astype(np.int64)
        return (step_idx, valid)

    def accumulate_od(o_local, d_local, step_idx, time_valid, od_matrix):
        """Bin trips into an (n, n, T) origin-destination matrix (in-place).

        o_local / d_local are local region indices (0..n-1, or -1 to drop).
        A trip is kept only when both endpoints fall inside the selected set.
        """
        n = od_matrix.shape[0]
        T_ = od_matrix.shape[2]
        valid = time_valid & (o_local >= 0) & (d_local >= 0)
        if not np.any(valid):
            return 0
        flat_idx = (
            o_local[valid].astype(np.int64) * n + d_local[valid].astype(np.int64)
        ) * T_ + step_idx[valid].astype(np.int64)
        binc = np.bincount(flat_idx, minlength=n * n * T_)
        od_matrix += binc.reshape(n, n, T_)
        return int(valid.sum())

    def process_taxi_od_parquet(files, schema_map):
        """schema_map: {pattern_prefix: {'pu_id','do_id','pu_time','do_time'}}."""
        import pyarrow.parquet as pq

        od = np.zeros((n_sel, n_sel, T), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        for fp in files:
            prefix = next((k for k in schema_map if fp.name.startswith(k)), None)
            if prefix is None:
                print(f"  skip (no schema): {fp.name}")
                continue
            cols = schema_map[prefix]
            time_col = cols["do_time"] if FLOW_MODE == "arrival" else cols["pu_time"]
            pu_col, do_col = (cols["pu_id"], cols["do_id"])
            print(f"Processing {fp.name} [O={pu_col}, D={do_col}, t={time_col}]")
            pq_file = pq.ParquetFile(fp)
            for rg in range(pq_file.num_row_groups):
                table = pq_file.read_row_group(rg, columns=[pu_col, do_col, time_col])
                df = table.to_pandas()
                total_rows += len(df)
                df = df.dropna(subset=[pu_col, do_col, time_col])
                if df.empty:
                    continue
                pu = pd.to_numeric(df[pu_col], errors="coerce")
                do = pd.to_numeric(df[do_col], errors="coerce")
                valid_loc = pu.notna() & do.notna()
                if not valid_loc.any():
                    continue
                pu = pu[valid_loc].astype(np.int64).reset_index(drop=True)
                do = do[valid_loc].astype(np.int64).reset_index(drop=True)
                ts = df.loc[valid_loc, time_col].reset_index(drop=True)
                o_region = pu.map(locationid_to_region).fillna(-1).to_numpy(dtype=np.int64)
                d_region = do.map(locationid_to_region).fillna(-1).to_numpy(dtype=np.int64)
                o_local = np.where(o_region >= 0, region_to_local[o_region.clip(min=0)], -1).astype(
                    np.int32
                )
                d_local = np.where(d_region >= 0, region_to_local[d_region.clip(min=0)], -1).astype(
                    np.int32
                )
                step_idx, time_valid = timestamps_to_step_idx(ts)
                kept_rows += accumulate_od(o_local, d_local, step_idx, time_valid, od)
        print(f"Rows seen: {total_rows:,}, kept in OD: {kept_rows:,}")
        return od

    taxi_schema = {
        "yellow_tripdata": {
            "pu_id": "PULocationID",
            "do_id": "DOLocationID",
            "pu_time": "tpep_pickup_datetime",
            "do_time": "tpep_dropoff_datetime",
        },
        "green_tripdata": {
            "pu_id": "PULocationID",
            "do_id": "DOLocationID",
            "pu_time": "lpep_pickup_datetime",
            "do_time": "lpep_dropoff_datetime",
        },
    }
    taxi_files = sorted(
        [p for p in TAXI_DIR.glob(f"yellow_tripdata_{YEAR}-*.parquet")]
        + [p for p in TAXI_DIR.glob(f"green_tripdata_{YEAR}-*.parquet")]
    )
    if not taxi_files:
        raise FileNotFoundError(f"No yellow/green taxi parquet files found under {TAXI_DIR}")
    nyc_taxi_od = process_taxi_od_parquet(taxi_files, taxi_schema)
    taxi_out = OUT_DIR / f"nyc_{borough_tag}_taxi_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(taxi_out, nyc_taxi_od)
    print(f"Saved {taxi_out} shape={nyc_taxi_od.shape} (O,D,T), total={nyc_taxi_od.sum():,}")
    fhv_schema = {
        "fhvhv_tripdata": {
            "pu_id": "PULocationID",
            "do_id": "DOLocationID",
            "pu_time": "pickup_datetime",
            "do_time": "dropoff_datetime",
        },
        "fhv_tripdata": {
            "pu_id": "PUlocationID",
            "do_id": "DOlocationID",
            "pu_time": "pickup_datetime",
            "do_time": "dropOff_datetime",
        },
    }
    fhv_schema_ordered = {k: fhv_schema[k] for k in sorted(fhv_schema, key=len, reverse=True)}
    fhv_files = sorted(
        [p for p in TAXI_DIR.glob(f"fhvhv_tripdata_{YEAR}-*.parquet")]
        + [p for p in TAXI_DIR.glob(f"fhv_tripdata_{YEAR}-*.parquet")]
    )
    if not fhv_files:
        raise FileNotFoundError(f"No fhv/fhvhv parquet files found under {TAXI_DIR}")
    nyc_fhv_od = process_taxi_od_parquet(fhv_files, fhv_schema_ordered)
    fhv_out = OUT_DIR / f"nyc_{borough_tag}_fhv_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(fhv_out, nyc_fhv_od)
    print(f"Saved {fhv_out} shape={nyc_fhv_od.shape} (O,D,T), total={nyc_fhv_od.sum():,}")

    def process_bike_od_csv(files, chunksize=500000):
        od = np.zeros((n_sel, n_sel, T), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        time_col = "ended_at" if FLOW_MODE == "arrival" else "started_at"
        usecols = ["start_lng", "start_lat", "end_lng", "end_lat", time_col]
        for fp in files:
            print(f"Processing {fp.name}")
            reader = pd.read_csv(fp, usecols=usecols, chunksize=chunksize, low_memory=False)
            for chunk in reader:
                total_rows += len(chunk)
                chunk = chunk.dropna(subset=usecols)
                if chunk.empty:
                    continue
                o_lon = pd.to_numeric(chunk["start_lng"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                o_lat = pd.to_numeric(chunk["start_lat"], errors="coerce").to_numpy(
                    dtype=np.float64
                )
                d_lon = pd.to_numeric(chunk["end_lng"], errors="coerce").to_numpy(dtype=np.float64)
                d_lat = pd.to_numeric(chunk["end_lat"], errors="coerce").to_numpy(dtype=np.float64)
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
                o_region = map_points_to_regions(o_lon, o_lat, zones_for_join, bounds)
                d_region = map_points_to_regions(d_lon, d_lat, zones_for_join, bounds)
                o_local = np.where(o_region >= 0, region_to_local[o_region.clip(min=0)], -1).astype(
                    np.int32
                )
                d_local = np.where(d_region >= 0, region_to_local[d_region.clip(min=0)], -1).astype(
                    np.int32
                )
                step_idx, time_valid = timestamps_to_step_idx(ts)
                kept_rows += accumulate_od(o_local, d_local, step_idx, time_valid, od)
        print(f"Rows seen: {total_rows:,}, kept in OD: {kept_rows:,}")
        return od

    bike_files = sorted(BIKE_DIR.glob(f"{YEAR}??-citibike-tripdata*.csv"))
    if not bike_files:
        raise FileNotFoundError(f"No bike CSV files found under {BIKE_DIR}")
    nyc_bike_od = process_bike_od_csv(bike_files)
    bike_out = OUT_DIR / f"nyc_{borough_tag}_bike_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(bike_out, nyc_bike_od)
    print(f"Saved {bike_out} shape={nyc_bike_od.shape} (O,D,T), total={nyc_bike_od.sum():,}")

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

    for name, arr in [("taxi", nyc_taxi_od), ("fhv", nyc_fhv_od), ("bike", nyc_bike_od)]:
        summarize_od(name, arr)
    MERGE_MOBILITIES = ["taxi", "fhv", "bike"]
    MOBILITY_ARRS = {"taxi": nyc_taxi_od, "fhv": nyc_fhv_od, "bike": nyc_bike_od}
    missing = [m for m in MERGE_MOBILITIES if m not in MOBILITY_ARRS]
    if missing:
        raise ValueError(f"Unknown mobilities in MERGE_MOBILITIES: {missing}")
    merged = np.stack([MOBILITY_ARRS[m] for m in MERGE_MOBILITIES], axis=2)
    merged_tag = "_".join(MERGE_MOBILITIES)
    M_dim = len(MERGE_MOBILITIES)
    merged_out = OUT_DIR / f"nyc_{borough_tag}_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(merged_out, merged)
    print(f"{merged_out} shape={merged.shape} (N, N, M, T) with M={M_dim} mobilities")
    for i in MOBILITY_ARRS:
        i_out = OUT_DIR / f"nyc_{borough_tag}_{i}_od_{YEAR}_{FREQ_TAG}.npy"
        np.save(i_out, MOBILITY_ARRS[i])
        print(f"{i_out} shape={MOBILITY_ARRS[i].shape} (N, N, T) with 1 mobilities")
    if TARGET_BOROUGH and str(TARGET_BOROUGH).lower() != "all":
        DATASET = f"nyc_{borough_tag}_od_{FREQ}"
    else:
        DATASET = f"nyc_od_{FREQ}"
    src_path = OUT_DIR / f"nyc_{borough_tag}_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
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
        i_out = OUT_DIR / f"nyc_{borough_tag}_{i}_od_{YEAR}_{FREQ_TAG}.npy"
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
    ADJ_OUT = ADJ_DIR / f"{borough_tag}.npy"
    ctr = zones_sel.set_geometry("geometry").to_crs("EPSG:2263").centroid.reset_index(drop=True)
    N_adj = len(ctr)
    ids = list(range(N_adj))
    distance = [[i, j, ctr[i].distance(ctr[j])] for i in ids for j in ids]
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ids)
    np.save(ADJ_OUT, adj_mx)
    for i in DATASET_list:
        ADJ_DIR = Path(processed_root) / i
        ADJ_DIR.mkdir(parents=True, exist_ok=True)
        ADJ_OUT = ADJ_DIR / f"{borough_tag}.npy"
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
    DIR = os.path.join(DIR, "NYC")
    os.makedirs(DIR, exist_ok=True)
    taxi_orig_dir = os.path.join(DIR, "taxi")
    bike_orig_file = os.path.join(DIR, "bike")

    DOWNLOAD = "taxi" in options.get("downloads", ["taxi", "bike"])
    year = options.get("year", 2025)
    taxi_types = ["yellow", "green", "fhvhv", "fhv"]
    base_url_template = (
        "https://d37ci6vzurychx.cloudfront.net/trip-data/{type}_tripdata_{year}-{month:02d}.parquet"
    )
    os.makedirs(taxi_orig_dir, exist_ok=True)
    if DOWNLOAD:
        for taxi_type in taxi_types:
            for month in options.get("months", list(range(1, 13))):
                url = base_url_template.format(type=taxi_type, year=year, month=month)
                file_name = f"{taxi_type}_tripdata_{year}-{month:02d}.parquet"
                save_path = os.path.join(taxi_orig_dir, file_name)
                try:
                    response = requests.get(url, stream=True, timeout=30)
                    response.raise_for_status()
                    if response.status_code == 200:
                        with open(save_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192 * 1024):
                                if chunk:
                                    f.write(chunk)
                except requests.exceptions.RequestException:
                    raise

    DOWNLOAD = "bike" in options.get("downloads", ["taxi", "bike"])
    year = options.get("year", 2025)
    base_url_template = "https://s3.amazonaws.com/tripdata/{year}{month:02d}-citibike-tripdata.zip"
    os.makedirs(bike_orig_file, exist_ok=True)
    if DOWNLOAD:
        for month in options.get("months", list(range(1, 13))):
            url = base_url_template.format(year=year, month=month)
            zip_filename = f"{year}{month:02d}-citibike-tripdata.zip"
            zip_filepath = os.path.join(bike_orig_file, zip_filename)
            try:
                response = requests.get(url, stream=True, timeout=30)
                response.raise_for_status()
                if response.status_code == 200:
                    with open(zip_filepath, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192 * 1024):
                            if chunk:
                                f.write(chunk)
                    with zipfile.ZipFile(zip_filepath, "r") as zip_ref:
                        zip_ref.extractall(bike_orig_file)
                    os.remove(zip_filepath)
            except requests.exceptions.RequestException:
                raise
    return locals()
