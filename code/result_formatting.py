"""Display full-precision results without rounding nonzero CI endpoints to zero."""

import math


def ci_pair(lower: str, upper: str) -> str:
    """Format an interval as (lower--upper), or (lower to upper) when a bound is signed."""
    signed = lambda x: x.startswith(("$-$", "$+$", "+", "-"))
    separator = " to " if signed(lower) or signed(upper) else "--"
    return f"({lower}{separator}{upper})"


def number(value, decimals=3):
    value = float(value)
    places = decimals
    # Keep at least two significant digits for small values, including endpoints
    # that would otherwise remain nonzero but lose most of their precision.
    if 0 < abs(value) < 10 ** (1 - decimals):
        places = min(8, max(decimals, 1 - math.floor(math.log10(abs(value)))))
    return f"{value:.{places}f}".replace("-", "$-$")


def difference_bounds(row, metric):
    key = f"Delta_{metric}"
    if f"{key}_CI_lower" in row and f"{key}_CI_upper" in row:
        return float(row[f"{key}_CI_lower"]), float(row[f"{key}_CI_upper"])
    text = str(row[f"{key}_95CI"])
    split = text.index("–", 1)
    return float(text[:split]), float(text[split + 1 :])


def difference(row, metric):
    lower, upper = difference_bounds(row, metric)
    return f"{number(row[f'Delta_{metric}'])} {ci_pair(number(lower), number(upper))}"
