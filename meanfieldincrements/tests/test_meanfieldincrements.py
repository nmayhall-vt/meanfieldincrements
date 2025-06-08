"""
Unit and regression test for the meanfieldincrements package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import meanfieldincrements


def test_meanfieldincrements_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "meanfieldincrements" in sys.modules
