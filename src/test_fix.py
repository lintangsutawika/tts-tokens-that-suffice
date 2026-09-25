import sys
# Make sure we don't import from testbed
sys.path = [p for p in sys.path if '/testbed' not in p]

# Now import from testbed
sys.path.insert(0, '/testbed')

from astropy.io.registry import identify_format
from astropy.table import Table

# This should no longer trigger the error
try:
    result = identify_format("write", Table, "bububu.ecsv", None, [], {})
    print(f"Result: {result}")
except IndexError as e:
    print(f"IndexError occurred: {e}")
except Exception as e:
    print(f"Other error: {type(e).__name__}: {e}")
