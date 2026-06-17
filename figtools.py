"""Backward-compatible wrapper for the renamed figcrop module."""

from figcrop import *  # noqa: F401,F403
from figcrop import main


if __name__ == "__main__":
    raise SystemExit(main())
