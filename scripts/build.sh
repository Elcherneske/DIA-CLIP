# Save current directory
OLD_DIR="$(pwd)"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MSTOOLKIT_DIR="$SCRIPT_DIR/diann/mstoolkit"

cd "$MSTOOLKIT_DIR" || { echo "Failed to cd into $MSTOOLKIT_DIR"; exit 1; }
make

# Return to the original directory
cd "$OLD_DIR" || exit 1