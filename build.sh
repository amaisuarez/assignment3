#!/usr/bin/env bash
# Compiles the Java implementations with the fixed settings used in the study.
# The Vector API (optional SIMD extension) is an incubator module in JDK 21, so it
# must be added explicitly both here and when running (see config.json, java_opts).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p java/build
echo "[build] javac $(javac -version 2>&1 | cut -d' ' -f2) --add-modules jdk.incubator.vector"
javac -Xlint:all --add-modules jdk.incubator.vector -d java/build java/*.java
echo "[build] done"
