"""Make `import arles` work no matter which directory pytest is invoked from."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
