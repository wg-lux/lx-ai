# lx_ai/utils/logging_utils.py
from __future__ import annotations

import os
import sys


def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


_RESET = "\033[0m"
_BOLD = "\033[1m"

_GREEN = "\033[92m"
_CYAN = "\033[96m"
_YELLOW = "\033[93m"
_MAGENTA = "\033[95m"
_BLUE = "\033[94m"
_RED = "\033[91m"
_WHITE = "\033[97m"


def _style(text: str, *codes: str) -> str:
    if not _supports_color():
        return text
    prefix = "".join(codes)
    return f"{prefix}{text}{_RESET}"


def section(title: str, icon: str = "") -> None:
    """
    Strong major section boundary.
    """
    label = f"{icon} {title}".strip()
    line = "═" * 80
    print()
    print(_style(line, _BOLD, _GREEN))
    print(_style(label.center(80), _BOLD, _WHITE))
    print(_style(line, _BOLD, _GREEN))


def subsection(title: str) -> None:
    """
    Standard subsection heading.
    """
    label = f"[{title}]"
    print()
    print(_style(label, _BOLD, _CYAN))


def decision_section(title: str) -> None:
    """
    Highlight decision-making blocks.
    """
    line = "═" * 78
    print()
    print(_style(f"╔{line}╗", _BOLD, _MAGENTA))
    print(_style(f"║ {title:<76} ║", _BOLD, _MAGENTA))
    print(_style(f"╚{line}╝", _BOLD, _MAGENTA))


def decision_subsection(title: str) -> None:
    """
    Inner heading inside a decision block.
    """
    print()
    print(_style(f"▶ {title}", _BOLD, _YELLOW))


def table_header(*cols: str, width: int = 80) -> None:
    """
    Consistent table heading.
    """
    line = "─" * width
    header = "  ".join(f"{c:<10}" for c in cols)
    print(_style(line, _BLUE))
    print(_style(header, _BOLD, _WHITE))
    print(_style(line, _BLUE))


def kv(label: str, value: object, width: int = 22) -> None:
    """
    Key-value line with aligned label.
    """
    label_txt = f"{label:<{width}}"
    print(
        f"{_style(label_txt, _MAGENTA)}: {value}"
        if _supports_color()
        else f"{label_txt}: {value}"
    )


def info(text: str) -> None:
    print(_style(text, _BLUE))


def success(text: str) -> None:
    print(_style(text, _GREEN))


def warning(text: str) -> None:
    print(_style(text, _YELLOW))


def error(text: str) -> None:
    print(_style(text, _RED))


def soft_line(char: str = "─", width: int = 80) -> None:
    print(_style(char * width, _BLUE))
