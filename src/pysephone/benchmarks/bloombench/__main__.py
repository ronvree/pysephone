"""Entry point for ``python -m pysephone.benchmarks.bloombench``."""

import sys

from pysephone.benchmarks.bloombench.cli import main


if __name__ == '__main__':
    sys.exit(main())
