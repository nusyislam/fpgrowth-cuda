"""Python replica of parseMinSupport() in src/main.cpp.

Keep this in sync with the C++ code: the tests use it to predict which
absolute threshold the binary derives from a min_support argument, for both
the absolute-count form ("3") and the fractional form ("0.05").
"""
import math


def parse_min_support(arg: str, num_transactions: int) -> int:
    """Threshold (absolute transaction count, >= 1) the binary will use."""
    if any(c in arg for c in ".eE"):
        f = float(arg)
        if not (0.0 < f <= 1.0):
            raise ValueError(f"fractional min_support must be in (0, 1]: {arg}")
        if num_transactions == 0:
            return 1

        n = float(num_transactions)
        raw = f * n
        k = math.floor(raw + 0.5)  # std::round for non-negative values

        # Match the typed precision: if some exact count k/N rounds to the
        # argument at the same number of decimals, use k; otherwise round up.
        if "e" in arg or "E" in arg:
            tol = 1e-9
        else:
            decimals = len(arg) - arg.find(".") - 1
            tol = 0.5 * 10.0 ** (-decimals) + 1e-12
        if k >= 1 and abs(k / n - f) <= tol:
            return k
        return max(1, math.ceil(raw - 1e-9))

    v = int(arg)
    if v < 0:
        raise ValueError(f"bad min_support: {arg}")
    return max(1, v)
