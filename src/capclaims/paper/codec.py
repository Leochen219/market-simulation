"""Exact decimal-prefix interface for paper action domains."""

def legal_prefix_choices(prefix: str, maximum: int) -> tuple[int, ...]:
    if maximum < 0 or maximum > 999 or not prefix.isdigit() and prefix != "":
        raise ValueError("invalid decimal prefix or maximum")
    if len(prefix) >= 3:
        raise ValueError("prefix must contain fewer than three digits")
    return prefix_mask(prefix, 0, maximum)


def decode_prefix(digits: tuple[int, ...] | list[int]) -> int:
    if len(digits) != 3 or any(isinstance(d, bool) or not isinstance(d, int) or not 0 <= d <= 9 for d in digits):
        raise ValueError("a decimal choice requires three digits from 0 to 9")
    return digits[0] * 100 + digits[1] * 10 + digits[2]


def prefix_mask(prefix: str, minimum: int, maximum: int) -> tuple[int, ...]:
    """Return next digits whose at-least-one completion lies in [minimum, maximum]."""
    if not 0 <= minimum <= maximum <= 999 or len(prefix) >= 3 or (prefix and not prefix.isdigit()):
        raise ValueError("invalid prefix bounds")
    place = 10 ** (2 - len(prefix))
    start = int(prefix) * 10 ** (3 - len(prefix)) if prefix else 0
    return tuple(d for d in range(10) if start + d * place <= maximum and start + d * place + place - 1 >= minimum)
