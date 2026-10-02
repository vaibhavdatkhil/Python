#!/usr/bin/env python3
"""
Test script for Task 7: verify CSV loading path works end-to-end.
Creates a minimal test CSV, updates config to point to it, and loads it.
"""

import sys
import tempfile
from pathlib import Path

# Create a minimal valid CSV with required columns
csv_content = """timestamp,cpu_util
2024-01-01T00:00:00,0.45
2024-01-01T00:01:00,0.52
2024-01-01T00:02:00,0.48
2024-01-01T00:03:00,0.61
2024-01-01T00:04:00,0.55
"""

# Write to temp file
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
    f.write(csv_content)
    temp_csv_path = f.name

print(f"Created test CSV at: {temp_csv_path}")

# Build a minimal config dict pointing to this CSV
test_config = {
    "data": {
        "source": "csv",
        "csv_path": temp_csv_path,
        "freq": "1min",
    }
}

# Test the loader
sys.path.insert(0, str(Path(__file__).parent))

try:
    from data.loader import load_trace
    
    print("Loading CSV via load_trace()...")
    df = load_trace(cfg=test_config)
    
    print(f"✓ Loaded {len(df)} rows")
    print(f"✓ Columns: {df.columns.tolist()}")
    print(f"✓ Index type: {type(df.index).__name__}")
    print(f"✓ cpu_util range: [{df['cpu_util'].min():.2f}, {df['cpu_util'].max():.2f}]")
    print(f"\nFirst 3 rows:")
    print(df.head(3))
    
    # Verify expectations
    assert len(df) == 5, f"Expected 5 rows, got {len(df)}"
    assert "cpu_util" in df.columns, "Missing cpu_util column"
    assert df.index.name == "timestamp" or isinstance(df.index, pd.core.indexes.datetimes.DatetimeIndex), "Index should be timestamp"
    assert df["cpu_util"].min() >= 0.0 and df["cpu_util"].max() <= 1.0, "cpu_util should be normalized to [0, 1]"
    
    print("\n✓ PASS: CSV loading path works correctly")
    
except Exception as e:
    print(f"\n✗ FAIL: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
finally:
    # Cleanup
    import os
    if os.path.exists(temp_csv_path):
        os.unlink(temp_csv_path)
        print(f"\nCleaned up test file: {temp_csv_path}")
