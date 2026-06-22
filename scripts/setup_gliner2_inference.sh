#!/usr/bin/env bash
# Fetch the upstream gliner2_inference engine at the pinned tag and apply our local
# patches. The upstream source is NOT committed to this repo — only patches/ and this
# script are. This populates third_party/gliner2_inference (gitignored), which
# gliner2_binding/Cargo.toml depends on via `path`.
#
# Local patches (see patches/):
#   - classifier input dtype fix (fp32 models no longer fail with a float16 error)
#   - structured/JSON extraction support
set -euo pipefail

REPO="https://github.com/SemplificaAI/gliner2-rs"
TAG="v0.5.1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DEST="$ROOT/third_party/gliner2_inference"
PATCH_DIR="$ROOT/patches"

echo "==> Fetching $REPO @ $TAG"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone --quiet --depth 1 --branch "$TAG" "$REPO" "$TMP/src"

# The crate lives in the repo's rust_component/ subdirectory.
rm -rf "$DEST"
mkdir -p "$(dirname "$DEST")"
cp -r "$TMP/src/rust_component" "$DEST"

echo "==> Applying patches"
shopt -s nullglob
patches=("$PATCH_DIR"/*.patch)
if [ ${#patches[@]} -eq 0 ]; then
  echo "    (no patches found in $PATCH_DIR)"
else
  for p in "${patches[@]}"; do
    echo "    applying $(basename "$p")"
    # Patch paths are crate-root-relative (a/src/..., b/src/...); -p1 strips the a//b/.
    patch -p1 -d "$DEST" < "$p"
  done
fi

echo "==> gliner2_inference ready at $DEST"
