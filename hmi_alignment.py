"""Timestamp validation shared by the HMI alignment GUI and offline tools."""
import csv
from datetime import datetime, timezone


def load_timestamps(path, frame_count):
    """Require one UTC timestamp per frame, with contiguous one-based indices."""
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != frame_count:
        raise ValueError(f'{path}: {len(rows)} timestamps for {frame_count} data frames')
    timestamps = []
    for index, row in enumerate(rows, 1):
        if int(row['series_index']) != index:
            raise ValueError(f'{path}: expected series_index {index}')
        stamp = datetime.fromisoformat(row['timestamp_utc'].replace('Z', '+00:00'))
        if stamp.tzinfo is None or stamp.utcoffset().total_seconds() != 0:
            raise ValueError(f'{path}: timestamp {index} must explicitly use UTC')
        stamp = stamp.astimezone(timezone.utc)
        if timestamps and stamp <= timestamps[-1]:
            raise ValueError(f'{path}: timestamps must increase strictly')
        timestamps.append(stamp)
    return timestamps


def get_closest(target_datetime, fits_datetimes):
    if not fits_datetimes:
        raise ValueError('No HMI FITS files found')
    return min(fits_datetimes, key=lambda item: (abs(item[0] - target_datetime), item[0], str(item[1])))
