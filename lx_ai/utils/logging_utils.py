# lx_ai/utils/logging_utils.py
import os
import sys

def section(title: str, icon: str = "") -> None:
    line = "=" * 80
    color = _GREEN if _supports_color() else ""
    reset = _RESET if color else ""

    print(f"\n{color}{line}{reset}")
    print(f"{color}{icon} {title}".strip() + reset)
    print(f"{color}{line}{reset}")

def subsection(title: str) -> None:
    color = _GREEN if _supports_color() else ""
    reset = _RESET if color else ""
    print(f"\n{color}[{title}]{reset}")



def table_header(*cols: str) -> None:
    color = _GREEN if _supports_color() else ""
    reset = _RESET if color else ""

    line = "-" * 80
    header = "  ".join(f"{c:<10}" for c in cols)

    print(f"{color}{line}{reset}")
    print(f"{color}{header}{reset}")
    print(f"{color}{line}{reset}")


def _supports_color() -> bool:
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


_GREEN = "\033[92m"
_RESET = "\033[0m"
