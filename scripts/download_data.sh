#!/usr/bin/env bash
# Download the benchmark datasets from the UCI Machine Learning Repository into
# data/raw/ and extract the files preprocess.py needs.  Files that already
# exist are skipped, so re-running is cheap.
#
#   Mushroom        https://archive.ics.uci.edu/dataset/73
#   Chess (KR-vs-KP) https://archive.ics.uci.edu/dataset/22
#   Connect-4       https://archive.ics.uci.edu/dataset/26
#   Online Retail   https://archive.ics.uci.edu/dataset/352
set -euo pipefail

RAW="$(cd "$(dirname "$0")/.." && pwd)/data/raw"
BASE="https://archive.ics.uci.edu/static/public"
mkdir -p "$RAW"

# fetch <url> <zip name>
fetch() {
    if [[ -f "$RAW/$2" ]]; then
        echo "skip     $2 (exists)"
    else
        echo "download $2"
        curl -fsSL --retry 3 -o "$RAW/$2.part" "$1"
        mv "$RAW/$2.part" "$RAW/$2"
    fi
}

# extract <zip name> <member>   (python's zipfile: no dependency on unzip)
extract() {
    if [[ -f "$RAW/$2" ]]; then
        echo "skip     $2 (exists)"
    else
        echo "extract  $2"
        python3 -I - "$RAW/$1" "$2" "$RAW" <<'EOF'
import sys, zipfile
zipfile.ZipFile(sys.argv[1]).extract(sys.argv[2], sys.argv[3])
EOF
    fi
}

fetch "$BASE/73/mushroom.zip"                         mushroom.zip
fetch "$BASE/22/chess+king+rook+vs+king+pawn.zip"     chess.zip
fetch "$BASE/26/connect+4.zip"                        connect4.zip
fetch "$BASE/352/online+retail.zip"                   online_retail.zip

extract mushroom.zip      agaricus-lepiota.data
extract mushroom.zip      agaricus-lepiota.names
extract chess.zip         kr-vs-kp.data
extract chess.zip         kr-vs-kp.names
extract connect4.zip      connect-4.data.Z
extract connect4.zip      connect-4.names
extract online_retail.zip "Online Retail.xlsx"

# connect-4.data is Unix-compress'd (.Z); gzip can decode that format.
if [[ -f "$RAW/connect-4.data" ]]; then
    echo "skip     connect-4.data (exists)"
else
    echo "extract  connect-4.data"
    gzip -dc "$RAW/connect-4.data.Z" > "$RAW/connect-4.data.part"
    mv "$RAW/connect-4.data.part" "$RAW/connect-4.data"
fi

echo "raw data in $RAW"
