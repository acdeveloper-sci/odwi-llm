"""Make `tests/contract/_fake_adapter.py` importable from this package —
the plan reuses the Stage 1 `FakeAdapter` as the fake `LLMPort` for the
Workflow suite. Same sys.path trick `contract_shape/conftest.py` uses for
`_replay`.
"""

import sys
from pathlib import Path

_CONTRACT = Path(__file__).resolve().parent.parent / "contract"
if str(_CONTRACT) not in sys.path:
    sys.path.insert(0, str(_CONTRACT))
