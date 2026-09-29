"""Resumable keyset pagination shared by the city data downloaders."""

import os
import csv
import time
import random
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


def _last_nonempty_line(path, tail_bytes=64 * 1024):
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        if size == 0:
            return None
        fh.seek(max(0, size - tail_bytes))
        chunks = fh.read().split(b"\n")
        while chunks and (not chunks[-1].strip()):
            chunks.pop()
        return chunks[-1].decode("utf-8") if chunks else None


def _count_data_rows(path):
    with open(path, "rb") as fh:
        return max(0, sum((1 for _ in fh)) - 1)


def _make_session():
    """Session with transport-level retry for DNS / connect failures."""
    session = requests.Session()
    retry = Retry(
        total=10,
        connect=10,
        read=0,
        status=5,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=32, pool_maxsize=32)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def download_from_api(
    API_URL,
    base_where,
    file,
    order_col,
    tie_col=None,
    select_cols=None,
    page_size=5000,
    max_retries=50,
    request_timeout=120,
    min_page_size=2000,
):
    """Keyset-paginated downloader for Socrata SODA endpoints.

    Avoids deep $offset (which goes O(offset) on the server and times out beyond
    a few million rows). Resumes by reading the last row of `file` and using
    its (order_col, tie_col) as a cursor.

    `select_cols`, when given, is the value of the SODA `$select` clause (a comma
    separated column list); it MUST include order_col and tie_col so the cursor
    can be read back from the file on resume.

    On repeated ReadTimeouts the page size is halved (down to `min_page_size`)
    so a struggling endpoint can still make progress; it ramps back up after a
    successful page. DNS / connection failures use exponential backoff with
    jitter to ride out resolver flaps.
    """
    cursor_order = None
    cursor_tie = None
    header_cols = None
    header_written = False
    total = 0
    mode = "w"
    if os.path.exists(file) and os.path.getsize(file) > 0:
        with open(file, "r", encoding="utf-8") as fh:
            header_line = fh.readline().rstrip("\n")
        if header_line:
            header_cols = next(csv.reader([header_line]))
            last = _last_nonempty_line(file)
            if last and last != header_line:
                last_row = next(csv.reader([last]))
                row_dict = dict(zip(header_cols, last_row))
                cursor_order = row_dict.get(order_col)
                if tie_col:
                    cursor_tie = row_dict.get(tie_col)
                total = _count_data_rows(file)
                header_written = True
                mode = "a"
                tie_str = f", {tie_col}>'{cursor_tie}'" if cursor_tie else ""
                print(
                    f"Resuming after {order_col}='{cursor_order}'{tie_str} ({total} existing rows)"
                )
    order_clause = f"{order_col} ASC" + (f", {tie_col} ASC" if tie_col else "")
    session = _make_session()
    cur_page_size = page_size
    with open(file, mode=mode, encoding="utf-8") as f:
        while True:
            if cursor_order is None:
                where = base_where
            elif tie_col and cursor_tie is not None:
                where = f"({base_where}) AND ({order_col} > '{cursor_order}' OR ({order_col} = '{cursor_order}' AND {tie_col} > '{cursor_tie}'))"
            else:
                where = f"({base_where}) AND {order_col} > '{cursor_order}'"
            query = {"$where": where, "$order": order_clause, "$limit": cur_page_size}
            if select_cols:
                query["$select"] = select_cols
            response = None
            for attempt in range(1, max_retries + 1):
                try:
                    response = session.get(API_URL, params=query, timeout=request_timeout)
                    break
                except requests.exceptions.RequestException as e:
                    is_timeout = isinstance(
                        e, (requests.exceptions.ReadTimeout, requests.exceptions.ConnectTimeout)
                    )
                    is_conn = isinstance(e, requests.exceptions.ConnectionError)
                    msg = str(e)
                    is_dns = (
                        "Name or service not known" in msg
                        or "Temporary failure in name resolution" in msg
                        or "nodename nor servname" in msg
                        or ("getaddrinfo failed" in msg)
                    )
                    if is_timeout and cur_page_size > min_page_size:
                        cur_page_size = max(min_page_size, cur_page_size // 2)
                        query["$limit"] = cur_page_size
                        print(
                            f"Attempt {attempt} timed out: {e}. Reducing page_size to {cur_page_size} ..."
                        )
                    if attempt == max_retries:
                        print(f"Failed after {max_retries} attempts: {e}")
                        raise
                    if is_dns or is_conn:
                        wait = min(300, 2 ** min(attempt, 8)) + random.uniform(0, 5)
                        kind = "DNS" if is_dns else "connection"
                        print(f"Attempt {attempt} {kind} error: {e}. Retrying in {wait:.1f}s ...")
                    else:
                        wait = min(120, 5 * attempt)
                        print(f"Attempt {attempt} failed: {e}. Retrying in {wait}s ...")
                    time.sleep(wait)
            response.raise_for_status()
            lines = response.text.splitlines()
            if len(lines) <= 1:
                break
            header_resp, data_lines = (lines[0], lines[1:])
            if not header_written:
                f.write(header_resp + "\n")
                header_cols = next(csv.reader([header_resp]))
                header_written = True
            elif header_cols is None:
                header_cols = next(csv.reader([header_resp]))
            for line in data_lines:
                f.write(line + "\n")
            f.flush()
            last_row = next(csv.reader([data_lines[-1]]))
            row_dict = dict(zip(header_cols, last_row))
            cursor_order = row_dict.get(order_col)
            if tie_col:
                cursor_tie = row_dict.get(tie_col)
            n = len(data_lines)
            total += n
            print(f"Downloaded {total} (page_size={cur_page_size}) ...")
            if cur_page_size < page_size:
                cur_page_size = min(page_size, cur_page_size * 2)
            if n < query["$limit"]:
                break
    print(f"\nSaved {total}")
