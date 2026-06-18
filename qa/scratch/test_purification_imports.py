import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    from config import PURIFY_STRICT_RIGOR
    from core.purification_engine import PurificationEngine
    print(f"Import successful! PURIFY_STRICT_RIGOR={PURIFY_STRICT_RIGOR}")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
