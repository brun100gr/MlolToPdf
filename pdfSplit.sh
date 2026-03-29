#!/bin/bash

# =========================
# Argument check
# =========================
if [ $# -lt 1 ] || [ $# -gt 2 ]; then
    echo "Usage: $0 file.pdf [pages_per_chunk]"
    echo "Example: $0 Impero_Alba.pdf 200"
    exit 1
fi

INPUT="$1"
STEP="${2:-100}"   # default = 100 if not specified

# =========================
# Validations
# =========================
if [ ! -f "$INPUT" ]; then
    echo "Error: file '$INPUT' not found"
    exit 1
fi

if ! [[ "$STEP" =~ ^[0-9]+$ ]] || [ "$STEP" -le 0 ]; then
    echo "Error: pages_per_chunk must be a positive integer"
    exit 1
fi

BASENAME="$(basename "$INPUT" .pdf)"
START=1

# =========================
# Main loop
# =========================
while true; do
    END=$((START + STEP - 1))
    OUTPUT=$(printf "%s_%04d_%04d.pdf" "$BASENAME" "$START" "$END")

    echo "Extracting pages $START-$END → $OUTPUT"

    qpdf "$INPUT" --pages . "$START-$END" -- "$OUTPUT" 2>err.log

    if grep -q "out of range" err.log; then
        rm -f "$OUTPUT"
        FINAL_OUTPUT=$(printf "%s_%04d_end.pdf" "$BASENAME" "$START")
        echo "End of document → $FINAL_OUTPUT"
        qpdf "$INPUT" --pages . "$START-z" -- "$FINAL_OUTPUT"
        break
    fi

    START=$((START + STEP))
done

rm -f err.log
echo "Done ✔"
