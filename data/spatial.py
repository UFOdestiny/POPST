"""Shared spatial joins, time binning, and exact count accumulation."""

import numpy as np
import pandas as pd
import geopandas as geopd


def make_step_idx_fn(freq, n_t, time_start):
    def fn(ts_series, fmt=None):
        ts = pd.to_datetime(ts_series, format=fmt, errors="coerce")
        step = ts.dt.floor(freq)
        delta = (step - time_start) / pd.Timedelta(freq)
        idx_float = delta.to_numpy(dtype=np.float64)
        valid = np.isfinite(idx_float) & (idx_float >= 0) & (idx_float < n_t)
        step_idx = np.zeros(len(idx_float), dtype=np.int64)
        step_idx[valid] = idx_float[valid].astype(np.int64)
        return (step_idx, valid)

    return fn


def map_points_to_regions(lon_arr, lat_arr, join_gdf, bounds_):
    minx, miny, maxx, maxy = bounds_
    region_idx = np.full(len(lon_arr), -1, dtype=np.int32)
    in_bbox = (lon_arr >= minx) & (lon_arr <= maxx) & (lat_arr >= miny) & (lat_arr <= maxy)
    if not np.any(in_bbox):
        return region_idx
    candidate_ids = np.where(in_bbox)[0]
    pts = geopd.GeoDataFrame(
        {"row_id": candidate_ids},
        geometry=geopd.points_from_xy(lon_arr[in_bbox], lat_arr[in_bbox]),
        crs="EPSG:4326",
    )
    joined = geopd.sjoin(pts, join_gdf, how="left", predicate="intersects")
    matched = joined.dropna(subset=["region_idx"]).drop_duplicates(subset=["row_id"], keep="first")
    region_idx[matched["row_id"].to_numpy(dtype=np.int64)] = matched["region_idx"].to_numpy(
        dtype=np.int32
    )
    return region_idx


def accumulate_od(o_region, d_region, step_idx, time_valid, od_matrix, n_t):
    """Bin trips into an (N, N, n_t) origin-destination matrix (in-place).

    o_region / d_region are shapefile region indices (0..N-1, or -1 to drop).
    A trip is kept only when both endpoints map to a valid region.
    """
    n = od_matrix.shape[0]
    valid = time_valid & (o_region >= 0) & (d_region >= 0)
    if not np.any(valid):
        return 0
    flat_idx = (
        o_region[valid].astype(np.int64) * n + d_region[valid].astype(np.int64)
    ) * n_t + step_idx[valid].astype(np.int64)
    binc = np.bincount(flat_idx, minlength=n * n * n_t)
    od_matrix += binc.reshape(n, n, n_t)
    return int(valid.sum())


def accumulate_nt(region_idx, step_idx, time_valid, nt_matrix, n_t):
    n = nt_matrix.shape[0]
    valid = time_valid & (region_idx >= 0)
    if not np.any(valid):
        return 0
    flat_idx = region_idx[valid].astype(np.int64) * n_t + step_idx[valid].astype(np.int64)
    binc = np.bincount(flat_idx, minlength=n * n_t)
    nt_matrix += binc.reshape(n, n_t)
    return int(valid.sum())
