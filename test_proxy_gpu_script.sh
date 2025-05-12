#!/bin/bash

echo "🧹 Cleaning up proxy test artifacts..."

# Remove test checkpoints
rm -v proxy_early_*.pt proxy_late_*.pt 2>/dev/null

# Remove logs (adjust the pattern as needed)
rm -v logs/proxy_test*.out logs/proxy_test*.err 2>/dev/null

# Optional: Remove any test metric files
rm -v logs/test_metrics.csv 2>/dev/null

echo "✅ Done."