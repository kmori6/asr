#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <out_dir>" >&2
  exit 1
fi

readonly out_dir="$1"
readonly base_url="https://www.openslr.org/resources/12"
readonly subsets=(
  "train-clean-100"
  "train-clean-360"
  "train-other-500"
  "dev-clean"
  "dev-other"
  "test-clean"
  "test-other"
)

mkdir -p "$out_dir"

for subset in "${subsets[@]}"; do
  archive="${subset}.tar.gz"
  archive_path="${out_dir}/${archive}"
  extracted_dir="${out_dir}/LibriSpeech/${subset}"

  if [[ ! -f "$archive_path" ]]; then
    echo "Downloading ${archive}..."
    wget --continue --output-document "$archive_path" "${base_url}/${archive}"
  fi

  if [[ -d "$extracted_dir" ]]; then
    echo "Skipping ${archive}: ${extracted_dir} already exists."
    continue
  fi

  echo "Extracting ${archive}..."
  tar -xzf "$archive_path" -C "$out_dir"
done

echo "LibriSpeech download complete."
