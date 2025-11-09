"""Pytest configuration and fixtures."""

import os
import sys
from unittest.mock import MagicMock

# Mock config before importing anything that uses it
sys.modules['libs.common.config'] = MagicMock()
mock_config = MagicMock()
mock_config.bin_pack_safety_margin_sec = 300
sys.modules['libs.common.config'].config = mock_config

# Mock database dependencies
sys.modules['libs.common.db'] = MagicMock()
sys.modules['libs.common.logging'] = MagicMock()

