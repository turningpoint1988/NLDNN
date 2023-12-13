#!/bin/bash

### NOTE: this script needs to be run once for each species and TF.

### Arguments expected:
ROOT=$1  # the directory for the project
tf=$2  # one of Cell.TF
genome=$3  # one of mm10, hg38
echo "Prepping training datasets for $tf ($genome)."

RAW_DATA_DIR="$ROOT/Human-Mouse/$tf/raw_data/$genome"
DATA_DIR="$ROOT/Human-Mouse/$tf/raw_data/$genome"
allfile="$RAW_DATA_DIR/all.all"

### Check all.all file whether exists in correct dir
if [[ ! -f "$allfile" ]]; then
	echo "File all_intersect.all is missing from $RAW_DATA_DIR. Exiting."
	exit 1
fi

allbed=$allfile

total_windows=`wc -l < "$allbed"`
echo "Total windows: $total_windows"

### Make validation and test sets

vafile="$DATA_DIR/validation.bed"
tefile="$DATA_DIR/test.bed"

grep -F "chr8"$'\t' "$allbed" > "$vafile"
grep -E "chr1"$'\t'"|""chr18"$'\t' "$allbed" > $tefile

te_windows=$(wc -l < "$tefile")
echo "Test set windows: $te_windows"

va_windows=$(wc -l < "$vafile")
echo "Validation set windows: $va_windows"

#
awk '$NF == 1' "$DATA_DIR/test.bed" | shuf > "$DATA_DIR/test_pos_shuf.bed"
awk '$NF == 0' "$DATA_DIR/test.bed" | shuf > "$DATA_DIR/test_neg_shuf.bed"

#
awk '$NF == 1' "$DATA_DIR/validation.bed" | shuf > "$DATA_DIR/validation_pos_shuf.bed"
awk '$NF == 0' "$DATA_DIR/validation.bed" | shuf > "$DATA_DIR/validation_neg_shuf.bed"

### Get training chromosomes, split into bound/unbound examples

grep -Ev "chr1"$'\t'"|""chr18"$'\t'"|""chr8"$'\t' "$allbed" | shuf > "$DATA_DIR/train_shuf.bed"
awk '$NF == 1' "$DATA_DIR/train_shuf.bed" | shuf > "$DATA_DIR/train_pos_shuf.bed"
awk '$NF == 0' "$DATA_DIR/train_shuf.bed" | shuf > "$DATA_DIR/train_neg_shuf.bed"

total_windows=$(wc -l < "$DATA_DIR/train_shuf.bed")
bound_windows=$(wc -l < "$DATA_DIR/train_pos_shuf.bed")
unbound_windows=$(wc -l < "$DATA_DIR/train_neg_shuf.bed")

total=$(( $bound_windows + $unbound_windows ))
if [[ $total != $total_windows ]]; then
	echo "Error: bound + unbound windows does not equal total windows. Exiting."
	exit 1
fi

echo "Bound training windows: $bound_windows"
echo "Unbound training windows: $unbound_windows"

rm "$DATA_DIR/train_shuf.bed"
rm "$DATA_DIR/validation.bed"
rm "$DATA_DIR/test.bed"
echo "Done!"

exit 0





