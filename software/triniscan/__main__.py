"""支持 ``python -m triniscan``。"""
import sys

from .core.main import main

if __name__ == "__main__":
    sys.exit(main())
