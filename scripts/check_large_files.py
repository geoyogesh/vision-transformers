"""Pre-commit hook: refuse to commit files larger than the limit (checkpoints, datasets)."""

import os
import sys

LIMIT_BYTES = 500_000


def main(paths: list[str]) -> int:
    too_large = [p for p in paths if os.path.isfile(p) and os.path.getsize(p) > LIMIT_BYTES]
    for p in too_large:
        print(f"too large ({os.path.getsize(p) / 1024:.0f} KB > {LIMIT_BYTES // 1000} KB): {p}")
    return 1 if too_large else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
