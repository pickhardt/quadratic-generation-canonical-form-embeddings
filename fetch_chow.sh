#!/bin/sh
# Fetch Schock's Chow-ring presentation of \tilde M(3,6), the external input to
# x36_chow_degree.py and x36_c2_blowup.py.  Pinned to the exact commit used for
# the certificates in this bundle, and checked by SHA-256.
set -e
COMMIT=282e0f3feee68b695f0a6cb255ef3b47425ae5fd
SHA=613a38de9b93c85b3bf61166782c6937ab101af13a22f03ea8da3a8d0a0f2885
URL="https://raw.githubusercontent.com/NSchock/IntersectionM36/$COMMIT/ChowM36.sage"
curl -fsSL "$URL" -o ChowM36.sage
echo "$SHA  ChowM36.sage" | shasum -a 256 -c -
echo "ChowM36.sage fetched and verified."
