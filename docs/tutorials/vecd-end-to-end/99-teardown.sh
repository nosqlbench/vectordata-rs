#!/usr/bin/env bash
# Step 99 — stop the vecd daemon. Leaves the demo directory in place so you
# can inspect it; prints the one command to remove it when you're done.
#
# Run:  bash 99-teardown.sh
source "$(dirname "$0")/env.sh"

say "stop the vecd daemon"
if vecd status 2>/dev/null | grep -q running; then
  vecd stop || true
else
  echo "  daemon not running"
fi

echo
echo "demo directory left in place:"
echo "  $DEMO_DIR"
echo
echo "to remove it, run (a literal path you can verify):"
echo "  rm -rf $DEMO_DIR"
