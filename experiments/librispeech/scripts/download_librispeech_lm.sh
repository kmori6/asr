#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <out_dir>" >&2
  exit 1
fi

readonly out_dir="$1"
readonly filename="librispeech-lm-norm.txt.gz"
readonly url="https://www.openslr.org/resources/11/${filename}"

mkdir -p "$out_dir"

echo "Downloading ${filename}..."
wget --continue --output-document "${out_dir}/${filename}" "$url"

echo "LibriSpeech language-modeling text download complete."
