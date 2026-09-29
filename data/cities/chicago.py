from data.downloads import download_from_api

from data.spatial import map_points_to_regions, accumulate_nt, accumulate_od
"""CHICAGO raw data download and Flow/OD preparation."""

from data.prepare import prepare_array


def process_flow(options=None, *, raw_root, asset_root, processed_root):
    options = dict(options or {})
    from pathlib import Path
    import os
    import numpy as np
    import pandas as pd
    import geopandas as geopd

    DIR = str(Path(raw_root))
    DIR = os.path.join(DIR, "Chicago")
    TAXI_FILE = Path(DIR) / f"chi_taxi_{options.get('year', 2025)}_15min.csv"
    TNP_DIR = Path(DIR) / "TNP"
    SCOOTER_FILE = Path(DIR) / f"chi_scooter_{options.get('year', 2025)}_60min.csv"
    BIKE_DIR = Path(DIR) / "bike"
    FLOW_MODE = options.get("flow_mode", "arrival")
    assert FLOW_MODE in {"arrival", "departure"}
    AREAS_PATH = str(Path(asset_root) / "geo" / "Chicago Community Areas.geojson")
    OUT_DIR = Path(processed_root) / "intermediate" / "Chicago"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_AREAS = options.get("target_areas", None)
    TARGET_TAG = options.get("target_tag", "central")
    YEAR = options.get("year", 2025)
    FREQ = options.get("freq", "15min")
    FREQ_SCOOTER = "60min"
    TIME_START = pd.Timestamp(options.get("start", f"{YEAR}-01-01 00:00:00"))
    TIME_END = pd.Timestamp(options.get("end", f"{YEAR + 1}-01-01 00:00:00"))

    def make_grid(freq):
        steps = pd.date_range(TIME_START, TIME_END, freq=freq, inclusive="left")
        return (steps, len(steps))

    STEPS, T = make_grid(FREQ)
    STEPS_SCOOTER, T_SCOOTER = make_grid(FREQ_SCOOTER)
    FREQ_TAG = FREQ
    FREQ_TAG_SCOOTER = FREQ_SCOOTER
    print(
        f"YEAR={YEAR}, FREQ={FREQ} (T={T}), FREQ_SCOOTER={FREQ_SCOOTER} (T={T_SCOOTER}), FLOW_MODE={FLOW_MODE}"
    )
    areas = geopd.read_file(AREAS_PATH).to_crs("EPSG:4326")
    areas["area_number"] = pd.to_numeric(areas["area_numbe"], errors="coerce").astype("Int64")
    areas = areas.dropna(subset=["area_number"]).sort_values("area_number").reset_index(drop=True)
    areas["region_idx"] = np.arange(len(areas), dtype=np.int32)
    areas_for_join = areas[["region_idx", "geometry"]]
    N = len(areas)
    bounds = areas.total_bounds
    area_to_region = {
        int(a): idx for idx, a in enumerate(areas["area_number"].astype(int).tolist())
    }
    print(f"N={N} community areas, bbox={bounds}")

    def make_step_idx_fn(freq, n_t):
        def fn(ts_series):
            ts = pd.to_datetime(ts_series, errors="coerce")
            step = ts.dt.floor(freq)
            delta = (step - TIME_START) / pd.Timedelta(freq)
            idx_float = delta.to_numpy(dtype=np.float64)
            valid = np.isfinite(idx_float) & (idx_float >= 0) & (idx_float < n_t)
            step_idx = np.zeros(len(idx_float), dtype=np.int64)
            step_idx[valid] = idx_float[valid].astype(np.int64)
            return (step_idx, valid)

        return fn

    step_idx_15 = make_step_idx_fn(FREQ, T)
    step_idx_60 = make_step_idx_fn(FREQ_SCOOTER, T_SCOOTER)

    def process_area_csv(path, area_col, time_col, step_idx_fn, n_t, chunksize=500000):
        """For CSVs that already carry a community area column."""
        nt = np.zeros((N, n_t), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        print(f"Processing {path.name} [{area_col}, {time_col}]")
        reader = pd.read_csv(
            path, usecols=[area_col, time_col], chunksize=chunksize, low_memory=False
        )
        for chunk in reader:
            total_rows += len(chunk)
            chunk = chunk.dropna(subset=[area_col, time_col])
            if chunk.empty:
                continue
            area = pd.to_numeric(chunk[area_col], errors="coerce")
            valid_area = area.notna()
            if not valid_area.any():
                continue
            area = area[valid_area].astype(np.int64).reset_index(drop=True)
            ts = chunk.loc[valid_area, time_col].reset_index(drop=True)
            region_idx = area.map(area_to_region).fillna(-1).to_numpy(dtype=np.int32)
            step_idx, time_valid = step_idx_fn(ts)
            kept_rows += accumulate_nt(region_idx, step_idx, time_valid, nt, n_t)
        print(f"Rows seen: {total_rows:,}, kept in (N,T): {kept_rows:,}")
        return nt

    if FLOW_MODE == "arrival":
        taxi_area_col, taxi_time_col = ("dropoff_community_area", "trip_end_timestamp")
    else:
        taxi_area_col, taxi_time_col = ("pickup_community_area", "trip_start_timestamp")
    if not TAXI_FILE.exists():
        raise FileNotFoundError(f"Taxi file not found: {TAXI_FILE}")
    chi_taxi_nt = process_area_csv(TAXI_FILE, taxi_area_col, taxi_time_col, step_idx_15, T)
    taxi_out = OUT_DIR / f"chi_taxi_{YEAR}_{FREQ_TAG}.npy"
    np.save(taxi_out, chi_taxi_nt)
    print(f"Saved {taxi_out} shape={chi_taxi_nt.shape}, total={chi_taxi_nt.sum():,}")
    if FLOW_MODE == "arrival":
        tnp_area_col, tnp_time_col = ("dropoff_community_area", "trip_end_timestamp")
    else:
        tnp_area_col, tnp_time_col = ("pickup_community_area", "trip_start_timestamp")
    tnp_files = sorted(TNP_DIR.glob(f"chi_TNP_{YEAR}_*_15min.csv"))
    if not tnp_files:
        raise FileNotFoundError(f"No TNP CSV files found under {TNP_DIR}")
    chi_tnp_nt = np.zeros((N, T), dtype=np.int64)
    for fp in tnp_files:
        chi_tnp_nt += process_area_csv(fp, tnp_area_col, tnp_time_col, step_idx_15, T)
    tnp_out = OUT_DIR / f"chi_tnp_{YEAR}_{FREQ_TAG}.npy"
    np.save(tnp_out, chi_tnp_nt)
    print(f"Saved {tnp_out} shape={chi_tnp_nt.shape}, total={chi_tnp_nt.sum():,}")
    if FLOW_MODE == "arrival":
        scooter_area_col, scooter_time_col = ("end_community_area_number", "end_time")
    else:
        scooter_area_col, scooter_time_col = ("start_community_area_number", "start_time")
    if not SCOOTER_FILE.exists():
        raise FileNotFoundError(f"Scooter file not found: {SCOOTER_FILE}")
    chi_scooter_nt = process_area_csv(
        SCOOTER_FILE, scooter_area_col, scooter_time_col, step_idx_60, T_SCOOTER
    )
    scooter_out = OUT_DIR / f"chi_scooter_{YEAR}_{FREQ_TAG_SCOOTER}.npy"
    np.save(scooter_out, chi_scooter_nt)
    print(f"Saved {scooter_out} shape={chi_scooter_nt.shape}, total={chi_scooter_nt.sum():,}")

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
                step_idx, time_valid = step_idx_15(ts)
                region_idx = map_points_to_regions(lon, lat, areas_for_join, bounds)
                kept_rows += accumulate_nt(region_idx, step_idx, time_valid, nt, T)
        print(f"Rows seen: {total_rows:,}, kept in (N,T): {kept_rows:,}")
        return nt

    bike_files = sorted(BIKE_DIR.glob(f"{YEAR}??-divvy-tripdata.csv"))
    if not bike_files:
        raise FileNotFoundError(f"No Divvy bike CSV files found under {BIKE_DIR}")
    chi_bike_nt = process_bike_csv(bike_files)
    bike_out = OUT_DIR / f"chi_bike_{YEAR}_{FREQ_TAG}.npy"
    np.save(bike_out, chi_bike_nt)
    print(f"Saved {bike_out} shape={chi_bike_nt.shape}, total={chi_bike_nt.sum():,}")

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

    for name, arr in [
        ("taxi", chi_taxi_nt),
        ("tnp", chi_tnp_nt),
        ("bike", chi_bike_nt),
        ("scooter", chi_scooter_nt),
    ]:
        summarize_nt(name, arr)
    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        target_set = {int(a) for a in TARGET_AREAS}
        mask = areas["area_number"].astype(int).isin(target_set)
        sel_idx = np.where(mask.to_numpy())[0]
        if sel_idx.size == 0:
            raise ValueError(f"No community areas matched TARGET_AREAS={TARGET_AREAS}")
        areas_sel = areas.iloc[sel_idx].reset_index(drop=True)
        area_tag = TARGET_TAG.lower().replace(" ", "_")
        geo_path = OUT_DIR / f"{area_tag}.geojson"
        areas_sel.drop(columns=["region_idx"], errors="ignore").to_file(geo_path, driver="GeoJSON")
        print(f"TARGET_AREAS={sorted(target_set)}, kept {len(sel_idx)}/{N} areas -> {geo_path}")
        for name, arr, freq_tag in [
            ("taxi", chi_taxi_nt, FREQ_TAG),
            ("tnp", chi_tnp_nt, FREQ_TAG),
            ("bike", chi_bike_nt, FREQ_TAG),
            ("scooter", chi_scooter_nt, FREQ_TAG_SCOOTER),
        ]:
            sub = arr[sel_idx]
            out = OUT_DIR / f"chi_{area_tag}_{name}_{YEAR}_{freq_tag}.npy"
            np.save(out, sub)
            print(f"  {out} shape={sub.shape}, total={sub.sum():,}")
    else:
        sel_idx = np.arange(N)
        areas_sel = areas.copy()
        area_tag = "all"
        print(f"Area filter disabled, keeping all {N} community areas")
    MERGE_MOBILITIES = ["taxi", "tnp", "bike"]
    MOBILITY_ARRS = {"taxi": chi_taxi_nt, "tnp": chi_tnp_nt, "bike": chi_bike_nt}
    missing = [m for m in MERGE_MOBILITIES if m not in MOBILITY_ARRS]
    if missing:
        raise ValueError(f"Unknown mobilities in MERGE_MOBILITIES: {missing}")
    merged = np.stack([MOBILITY_ARRS[m] for m in MERGE_MOBILITIES], axis=1)
    merged_tag = "_".join(MERGE_MOBILITIES)
    merged_out = OUT_DIR / f"chi_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
    np.save(merged_out, merged)
    print(f"{merged_out} shape={merged.shape} (N,M,T)")
    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        merged_a = merged[sel_idx, :, :]
        merged_a_out = OUT_DIR / f"chi_{area_tag}_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        np.save(merged_a_out, merged_a)
        print(f"{merged_a_out} shape={merged_a.shape} (N,M,T)")
    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        src_path = OUT_DIR / f"chi_{area_tag}_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"chicago_{area_tag}_{FREQ}"
    else:
        src_path = OUT_DIR / f"chi_{merged_tag}_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"chicago_{FREQ}"
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

    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        ADJ_OUT = Path(processed_root) / f"chicago_{area_tag}_{FREQ}" / f"{TARGET_TAG}.npy"
        ADJ_OUT.parent.mkdir(parents=True, exist_ok=True)
        ctr = areas_sel.set_geometry("geometry").centroid.reset_index(drop=True)
        N_adj = len(ctr)
        ids = list(range(N_adj))
        distance = [[i, j, ctr[i].distance(ctr[j])] for i in ids for j in ids]
        adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ids)
        np.save(ADJ_OUT, adj_mx)
        print(f"Saved {ADJ_OUT} shape={adj_mx.shape}")
    else:
        ADJ_OUT = Path(processed_root) / f"chicago_{FREQ}" / "chicago.npy"
        ADJ_OUT.parent.mkdir(parents=True, exist_ok=True)
        ctr = areas.set_geometry("geometry").centroid.reset_index(drop=True)
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
    DIR = os.path.join(DIR, "Chicago")
    TAXI_FILE = Path(DIR) / f"chi_taxi_{options.get('year', 2025)}_15min.csv"
    TNP_DIR = Path(DIR) / "TNP"
    SCOOTER_FILE = Path(DIR) / f"chi_scooter_{options.get('year', 2025)}_60min.csv"
    BIKE_DIR = Path(DIR) / "bike"
    FLOW_MODE = options.get("flow_mode", "departure")
    assert FLOW_MODE in {"arrival", "departure"}
    AREAS_PATH = str(Path(asset_root) / "geo" / "Chicago Community Areas.geojson")
    OUT_DIR = Path(processed_root) / "intermediate" / "Chicago"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TARGET_AREAS = options.get("target_areas", None)
    TARGET_TAG = options.get("target_tag", "central")
    YEAR = options.get("year", 2025)
    FREQ = options.get("freq", "15min")
    FREQ_SCOOTER = "60min"
    TIME_START = pd.Timestamp(options.get("start", f"{YEAR}-01-01 00:00:00"))
    TIME_END = pd.Timestamp(options.get("end", f"{YEAR + 1}-01-01 00:00:00"))

    def make_grid(freq):
        steps = pd.date_range(TIME_START, TIME_END, freq=freq, inclusive="left")
        return (steps, len(steps))

    STEPS, T = make_grid(FREQ)
    STEPS_SCOOTER, T_SCOOTER = make_grid(FREQ_SCOOTER)
    FREQ_TAG = FREQ
    FREQ_TAG_SCOOTER = FREQ_SCOOTER
    print(
        f"YEAR={YEAR}, FREQ={FREQ} (T={T}), FREQ_SCOOTER={FREQ_SCOOTER} (T={T_SCOOTER}), FLOW_MODE={FLOW_MODE}"
    )
    areas = geopd.read_file(AREAS_PATH).to_crs("EPSG:4326")
    areas["area_number"] = pd.to_numeric(areas["area_numbe"], errors="coerce").astype("Int64")
    areas = areas.dropna(subset=["area_number"]).sort_values("area_number").reset_index(drop=True)
    areas["region_idx"] = np.arange(len(areas), dtype=np.int32)
    areas_for_join = areas[["region_idx", "geometry"]]
    N = len(areas)
    bounds = areas.total_bounds
    area_to_region = {
        int(a): idx for idx, a in enumerate(areas["area_number"].astype(int).tolist())
    }
    print(f"N={N} community areas, bbox={bounds}")

    def make_step_idx_fn(freq, n_t):
        def fn(ts_series):
            ts = pd.to_datetime(ts_series, errors="coerce")
            step = ts.dt.floor(freq)
            delta = (step - TIME_START) / pd.Timedelta(freq)
            idx_float = delta.to_numpy(dtype=np.float64)
            valid = np.isfinite(idx_float) & (idx_float >= 0) & (idx_float < n_t)
            step_idx = np.zeros(len(idx_float), dtype=np.int64)
            step_idx[valid] = idx_float[valid].astype(np.int64)
            return (step_idx, valid)

        return fn

    step_idx_15 = make_step_idx_fn(FREQ, T)
    step_idx_60 = make_step_idx_fn(FREQ_SCOOTER, T_SCOOTER)

    def process_area_od_csv(
        path, pu_area_col, do_area_col, time_col, step_idx_fn, n_t, chunksize=500000
    ):
        """OD binning for CSVs that already carry pickup/dropoff community area columns."""
        od = np.zeros((N, N, n_t), dtype=np.int64)
        total_rows = 0
        kept_rows = 0
        print(f"Processing {path.name} [O={pu_area_col}, D={do_area_col}, t={time_col}]")
        reader = pd.read_csv(
            path,
            usecols=[pu_area_col, do_area_col, time_col],
            chunksize=chunksize,
            low_memory=False,
        )
        for chunk in reader:
            total_rows += len(chunk)
            chunk = chunk.dropna(subset=[pu_area_col, do_area_col, time_col])
            if chunk.empty:
                continue
            pu = pd.to_numeric(chunk[pu_area_col], errors="coerce")
            do = pd.to_numeric(chunk[do_area_col], errors="coerce")
            valid_area = pu.notna() & do.notna()
            if not valid_area.any():
                continue
            pu = pu[valid_area].astype(np.int64).reset_index(drop=True)
            do = do[valid_area].astype(np.int64).reset_index(drop=True)
            ts = chunk.loc[valid_area, time_col].reset_index(drop=True)
            o_region = pu.map(area_to_region).fillna(-1).to_numpy(dtype=np.int32)
            d_region = do.map(area_to_region).fillna(-1).to_numpy(dtype=np.int32)
            step_idx, time_valid = step_idx_fn(ts)
            kept_rows += accumulate_od(o_region, d_region, step_idx, time_valid, od, n_t)
        print(f"Rows seen: {total_rows:,}, kept in OD: {kept_rows:,}")
        return od

    taxi_pu_col, taxi_do_col = ("pickup_community_area", "dropoff_community_area")
    taxi_time_col = "trip_end_timestamp" if FLOW_MODE == "arrival" else "trip_start_timestamp"
    if not TAXI_FILE.exists():
        raise FileNotFoundError(f"Taxi file not found: {TAXI_FILE}")
    chi_taxi_od = process_area_od_csv(
        TAXI_FILE, taxi_pu_col, taxi_do_col, taxi_time_col, step_idx_15, T
    )
    taxi_out = OUT_DIR / f"chi_taxi_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(taxi_out, chi_taxi_od)
    print(f"Saved {taxi_out} shape={chi_taxi_od.shape} (O,D,T), total={chi_taxi_od.sum():,}")
    tnp_pu_col, tnp_do_col = ("pickup_community_area", "dropoff_community_area")
    tnp_time_col = "trip_end_timestamp" if FLOW_MODE == "arrival" else "trip_start_timestamp"
    tnp_files = sorted(TNP_DIR.glob(f"chi_TNP_{YEAR}_*_15min.csv"))
    if not tnp_files:
        raise FileNotFoundError(f"No TNP CSV files found under {TNP_DIR}")
    chi_tnp_od = np.zeros((N, N, T), dtype=np.int64)
    for fp in tnp_files:
        chi_tnp_od += process_area_od_csv(fp, tnp_pu_col, tnp_do_col, tnp_time_col, step_idx_15, T)
    tnp_out = OUT_DIR / f"chi_tnp_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(tnp_out, chi_tnp_od)
    print(f"Saved {tnp_out} shape={chi_tnp_od.shape} (O,D,T), total={chi_tnp_od.sum():,}")
    scooter_pu_col, scooter_do_col = ("start_community_area_number", "end_community_area_number")
    scooter_time_col = "end_time" if FLOW_MODE == "arrival" else "start_time"
    if not SCOOTER_FILE.exists():
        raise FileNotFoundError(f"Scooter file not found: {SCOOTER_FILE}")
    chi_scooter_od = process_area_od_csv(
        SCOOTER_FILE, scooter_pu_col, scooter_do_col, scooter_time_col, step_idx_60, T_SCOOTER
    )
    scooter_out = OUT_DIR / f"chi_scooter_od_{YEAR}_{FREQ_TAG_SCOOTER}.npy"
    np.save(scooter_out, chi_scooter_od)
    print(
        f"Saved {scooter_out} shape={chi_scooter_od.shape} (O,D,T), total={chi_scooter_od.sum():,}"
    )

    def process_bike_od_csv(files, chunksize=500000):
        od = np.zeros((N, N, T), dtype=np.int64)
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
                o_region = map_points_to_regions(o_lon, o_lat, areas_for_join, bounds)
                d_region = map_points_to_regions(d_lon, d_lat, areas_for_join, bounds)
                step_idx, time_valid = step_idx_15(ts)
                kept_rows += accumulate_od(o_region, d_region, step_idx, time_valid, od, T)
        print(f"Rows seen: {total_rows:,}, kept in OD: {kept_rows:,}")
        return od

    bike_files = sorted(BIKE_DIR.glob(f"{YEAR}??-divvy-tripdata.csv"))
    if not bike_files:
        raise FileNotFoundError(f"No Divvy bike CSV files found under {BIKE_DIR}")
    chi_bike_od = process_bike_od_csv(bike_files)
    bike_out = OUT_DIR / f"chi_bike_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(bike_out, chi_bike_od)
    print(f"Saved {bike_out} shape={chi_bike_od.shape} (O,D,T), total={chi_bike_od.sum():,}")

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

    for name, arr in [
        ("taxi", chi_taxi_od),
        ("tnp", chi_tnp_od),
        ("bike", chi_bike_od),
        ("scooter", chi_scooter_od),
    ]:
        summarize_od(name, arr)
    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        target_set = {int(a) for a in TARGET_AREAS}
        mask = areas["area_number"].astype(int).isin(target_set)
        sel_idx = np.where(mask.to_numpy())[0]
        if sel_idx.size == 0:
            raise ValueError(f"No community areas matched TARGET_AREAS={TARGET_AREAS}")
        areas_sel = areas.iloc[sel_idx].reset_index(drop=True)
        area_tag = TARGET_TAG.lower().replace(" ", "_")
        geo_path = OUT_DIR / f"{area_tag}.geojson"
        areas_sel.drop(columns=["region_idx"], errors="ignore").to_file(geo_path, driver="GeoJSON")
        print(f"TARGET_AREAS={sorted(target_set)}, kept {len(sel_idx)}/{N} areas -> {geo_path}")
        for name, arr, freq_tag in [
            ("taxi", chi_taxi_od, FREQ_TAG),
            ("tnp", chi_tnp_od, FREQ_TAG),
            ("bike", chi_bike_od, FREQ_TAG),
            ("scooter", chi_scooter_od, FREQ_TAG_SCOOTER),
        ]:
            sub = arr[np.ix_(sel_idx, sel_idx)]
            out = OUT_DIR / f"chi_{area_tag}_{name}_od_{YEAR}_{freq_tag}.npy"
            np.save(out, sub)
            print(f"  {out} shape={sub.shape}, total={sub.sum():,}")
    else:
        sel_idx = np.arange(N)
        areas_sel = areas.copy()
        area_tag = "all"
        print(f"Area filter disabled, keeping all {N} community areas")
    MERGE_MOBILITIES = ["taxi", "tnp", "bike"]
    MOBILITY_ARRS = {"taxi": chi_taxi_od, "tnp": chi_tnp_od, "bike": chi_bike_od}
    missing = [m for m in MERGE_MOBILITIES if m not in MOBILITY_ARRS]
    if missing:
        raise ValueError(f"Unknown mobilities in MERGE_MOBILITIES: {missing}")
    merged = np.stack([MOBILITY_ARRS[m] for m in MERGE_MOBILITIES], axis=2)
    merged_tag = "_".join(MERGE_MOBILITIES)
    M_dim = len(MERGE_MOBILITIES)
    merged_out = OUT_DIR / f"chi_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
    np.save(merged_out, merged)
    print(f"{merged_out} shape={merged.shape} (N, N, M, T) with M={M_dim} mobilities")
    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        merged_a = merged[np.ix_(sel_idx, sel_idx)]
        merged_a_out = OUT_DIR / f"chi_{area_tag}_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
        np.save(merged_a_out, merged_a)
        print(f"{merged_a_out} shape={merged_a.shape} (N, N, M, T)")
    for i in MOBILITY_ARRS:
        i_out = OUT_DIR / f"chi_{i}_od_{YEAR}_{FREQ_TAG}.npy"
        np.save(i_out, MOBILITY_ARRS[i])
        print(f"{i_out} shape={MOBILITY_ARRS[i].shape} (N, N, T) with 1 mobilities")
    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        src_path = OUT_DIR / f"chi_{area_tag}_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"chicago_{area_tag}_od_{FREQ}"
    else:
        src_path = OUT_DIR / f"chi_{merged_tag}_od_{YEAR}_{FREQ_TAG}.npy"
        DATASET = f"chicago_od_{FREQ}"
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
        i_out = OUT_DIR / f"chi_{i}_od_{YEAR}_{FREQ_TAG}.npy"
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
    if TARGET_AREAS is not None and (
        not isinstance(TARGET_AREAS, str) or TARGET_AREAS.lower() != "all"
    ):
        ADJ_OUT = ADJ_DIR / f"{TARGET_TAG}.npy"
        adj_gdf = areas_sel
    else:
        ADJ_OUT = ADJ_DIR / "chicago.npy"
        adj_gdf = areas
    ctr = adj_gdf.set_geometry("geometry").to_crs("EPSG:3435").centroid.reset_index(drop=True)
    N_adj = len(ctr)
    ids = list(range(N_adj))
    distance = [[i, j, ctr[i].distance(ctr[j])] for i in ids for j in ids]
    adj_mx = get_adjacency_matrix(distance_df=distance, sensor_ids=ids)
    np.save(ADJ_OUT, adj_mx)
    for i in DATASET_list:
        ADJ_DIR = Path(processed_root) / i
        ADJ_DIR.mkdir(parents=True, exist_ok=True)
        ADJ_OUT = ADJ_DIR / f"chicago.npy"
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
    DIR = os.path.join(DIR, "Chicago")
    os.makedirs(DIR, exist_ok=True)
    taxi_orig_file = os.path.join(DIR, f"chi_taxi_{options.get('year', 2025)}_15min.csv")
    bike_orig_dir = os.path.join(DIR, "bike")
    scooter_orig_file = os.path.join(DIR, f"chi_scooter_{options.get('year', 2025)}_60min.csv")

    DOWNLOAD = "taxi" in options.get("downloads", ["taxi", "scooter", "bike", "tnp"])
    if DOWNLOAD:
        API_URL = "https://data.cityofchicago.org/resource/b8xg-w8bx.csv"
        base_where = f"trip_start_timestamp >= '{options.get('year', 2025)}-01-01T00:00:00.000' AND trip_start_timestamp < '{options.get('year', 2025) + 1}-01-01T00:00:00.000'"
        download_from_api(
            API_URL, base_where, taxi_orig_file, order_col="trip_start_timestamp", tie_col="trip_id"
        )

    DOWNLOAD = "scooter" in options.get("downloads", ["taxi", "scooter", "bike", "tnp"])
    if DOWNLOAD:
        API_URL = "https://data.cityofchicago.org/resource/2i5w-ykuw.csv"
        base_where = f"start_time >= '{options.get('year', 2025)}-01-01T00:00:00.000' AND start_time < '{options.get('year', 2025) + 1}-01-01T00:00:00.000'"
        download_from_api(
            API_URL, base_where, scooter_orig_file, order_col="start_time", tie_col="trip_id"
        )

    DOWNLOAD = "bike" in options.get("downloads", ["taxi", "scooter", "bike", "tnp"])
    year = options.get("year", 2025)
    base_url_template = (
        "https://divvy-tripdata.s3.amazonaws.com/{year}{month:02d}-divvy-tripdata.zip"
    )
    os.makedirs(bike_orig_dir, exist_ok=True)
    if DOWNLOAD:
        for month in options.get("months", list(range(1, 13))):
            url = base_url_template.format(year=year, month=month)
            zip_filename = f"{year}{month:02d}-divvy-tripdata.zip"
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

    from concurrent.futures import ThreadPoolExecutor, as_completed

    DOWNLOAD = "tnp" in options.get("downloads", ["taxi", "scooter", "bike", "tnp"])
    MAX_WORKERS = options.get("max_workers", 4)
    if DOWNLOAD:
        API_URL = "https://data.cityofchicago.org/resource/6dvr-xwnh.csv"
        year = options.get("year", 2025)
        TNP_dir = os.path.join(DIR, "TNP")
        os.makedirs(TNP_dir, exist_ok=True)

        def _download_one_month(month):
            start = f"{year}-{month:02d}-01T00:00:00.000"
            if month == 12:
                end = f"{year + 1}-01-01T00:00:00.000"
            else:
                end = f"{year}-{month + 1:02d}-01T00:00:00.000"
            base_where = f"trip_start_timestamp >= '{start}' AND trip_start_timestamp < '{end}'"
            month_file = os.path.join(TNP_dir, f"chi_TNP_{year}_{month:02d}_15min.csv")
            print(f"[{year}-{month:02d}] start -> {month_file}")
            download_from_api(
                API_URL, base_where, month_file, order_col="trip_start_timestamp", tie_col="trip_id"
            )
            print(f"[{year}-{month:02d}] done")
            return month

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {
                ex.submit(_download_one_month, m): m
                for m in options.get("months", list(range(1, 13)))
            }
            for fut in as_completed(futures):
                m = futures[fut]
                try:
                    fut.result()
                except Exception:
                    raise
    return locals()
