from typing import Optional

import numpy as np


def get_timeline(range_max: int, dtype: Optional[np.dtype] = None) -> np.ndarray:
    """Returns timeline array for requirements.

    Parameters
    ----------
    range_max : int
        Largest value in range. Assume `range(range_max)`. Commonly that's length of the signal.
    dtype : np.dtype
        Minimal definition type. Returned timeline will have dtype that's the same or with higher byte size.

    """
    timeline = np.arange(0, range_max, dtype=dtype)
    if timeline[-1] != range_max - 1:
        inclusive_dtype = smallest_inclusive_dtype(timeline.dtype, range_max)
        timeline = np.arange(0, range_max, dtype=inclusive_dtype)
    return timeline


def smallest_inclusive_dtype(ref_dtype: np.dtype, ref_value) -> np.dtype:
    """Returns a numpy dtype with the same base as reference dtype (ref_dtype)
    but with the range that includes reference value (ref_value).

    Parameters
    ----------
    ref_dtype : dtype
         Reference dtype. Used to select the base, i.e. int or float, for returned type.
    ref_value : value
        A value which needs to be included in returned dtype. Value will be typically int or float.

    """
    # Integer path
    if np.issubdtype(ref_dtype, np.integer):
        for dtype in [np.uint16, np.uint32, np.uint64]:
            if ref_value < np.iinfo(dtype).max:
                return dtype
        max_val = np.iinfo(np.uint32).max
        raise ValueError("Requested too large integer range. Exceeds max( uint64 ) == '{}.".format(max_val))

    # Integer path
    if np.issubdtype(ref_dtype, np.floating):
        for dtype in [np.float16, np.float32, np.float64]:
            if ref_value < np.finfo(dtype).max:
                return dtype
        max_val = np.finfo(np.float64).max
        raise ValueError("Requested too large integer range. Exceeds max( float64 ) == '{}.".format(max_val))

    raise ValueError("Unsupported dtype '{}'. Only intX and floatX are supported.".format(ref_dtype))
