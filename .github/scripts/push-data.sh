#!/usr/bin/env bash
# Copy a directory into the `data` branch and push it, retrying from a fresh
# clone if another workflow pushed in the meantime. Creates the branch on the
# first run. A `cache` directory inside the source is never copied.
#
#   push-data.sh <source dir> <destination dir in the data branch> <commit message>
#
# Needs GITHUB_TOKEN (with contents: write), GITHUB_REPOSITORY and RUNNER_TEMP.
set -euo pipefail

src=$(cd "$1" && pwd)
dest=$2
msg=$3
url=${DATA_REMOTE_URL:-"https://x-access-token:${GITHUB_TOKEN}@github.com/${GITHUB_REPOSITORY}.git"}
store="$RUNNER_TEMP/data-store"

for attempt in 1 2 3 4 5; do
  rm -rf "$store"
  if ! git clone --quiet --depth 1 --branch data "$url" "$store" 2>/dev/null; then
    if git ls-remote --exit-code --heads "$url" data >/dev/null 2>&1; then
      echo "clone failed but the branch exists; retrying"
      sleep $((attempt * 5))
      continue
    fi
    mkdir -p "$store"
    git -C "$store" init --quiet
    git -C "$store" checkout --quiet --orphan data
    git -C "$store" remote add origin "$url"
  fi
  git -C "$store" config user.name "github-actions[bot]"
  git -C "$store" config user.email "41898282+github-actions[bot]@users.noreply.github.com"
  mkdir -p "$store/$dest"
  (cd "$src" && tar --exclude=./cache -cf - .) | tar -xf - -C "$store/$dest"
  git -C "$store" add -A "$dest"
  if git -C "$store" diff --cached --quiet; then
    echo "nothing new to save"
    exit 0
  fi
  git -C "$store" commit --quiet -m "$msg"
  if git -C "$store" push --quiet origin data; then
    echo "saved to data:$dest"
    exit 0
  fi
  echo "push rejected (attempt $attempt); retrying from a fresh clone"
  sleep $((attempt * 5))
done
echo "could not push to the data branch" >&2
exit 1
