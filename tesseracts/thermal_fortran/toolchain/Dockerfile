# Copyright 2026 Coldplate contributors.
# SPDX-License-Identifier: Apache-2.0
#
# Toolchain image for the Fortran + Enzyme Tesseract:
#
#   Fortran --flang--> LLVM IR --Enzyme pass--> shared library
#
# Split out of the Tesseract build on purpose. It pulls ~200 MB of compiler
# across the network, and building it separately means that cost is paid once
# and cached, instead of on every rebuild of the component.
#
# Uses LLVM's flang rather than LFortran (which the upstream Enzyme example
# uses, and which describes itself as alpha). flang ships in the same apt repo
# as the LLVM tools and the Enzyme plugin is built against, so the whole
# toolchain comes from one source and one version.
#
# Build with:  docker build -t coldplate-enzyme-toolchain:1.0 .

FROM debian:bookworm-slim

ARG LLVM_VERSION=19
# The official LLVM-19 nightly asset is vendored because Enzyme's nightly
# workflow deletes and recreates its release assets. The digest is still
# checked in-image so repository corruption fails closed before compilation.
ARG ENZYME_SHA256=5b43014ab23fdf212b5c0852e5ae1d2e9d3062bf0aa2323bbbf63b33369ef031

# Retries matter here: large apt downloads on some networks arrive corrupted,
# which surfaces as a hash mismatch rather than a connection error.
RUN set -eux; \
    for i in 1 2 3 4 5; do \
        apt-get update -qq && \
        apt-get install -y --no-install-recommends \
            wget gnupg ca-certificates build-essential && break || \
        { echo "retry $i"; rm -rf /var/lib/apt/lists/*; sleep 10; }; \
    done; \
    rm -rf /var/lib/apt/lists/*

RUN set -eux; \
    mkdir -p /etc/apt/keyrings; \
    wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/llvm.gpg; \
    echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg] http://apt.llvm.org/bookworm/ llvm-toolchain-bookworm-${LLVM_VERSION} main" \
        > /etc/apt/sources.list.d/llvm.list; \
    for i in 1 2 3 4 5 6; do \
        apt-get update -qq && \
        apt-get install -y --no-install-recommends \
            llvm-${LLVM_VERSION} clang-${LLVM_VERSION} flang-${LLVM_VERSION} && break || \
        { echo "retry $i"; apt-get clean; rm -rf /var/lib/apt/lists/*; sleep 15; }; \
    done; \
    rm -rf /var/lib/apt/lists/*; \
    for tool in opt llvm-as llvm-link llvm-dis llc clang clang++ flang flang-new; do \
        ln -sf /usr/bin/${tool}-${LLVM_VERSION} /usr/local/bin/${tool} 2>/dev/null || true; \
    done; \
    ln -sf /usr/bin/flang-new-${LLVM_VERSION} /usr/local/bin/flang-new 2>/dev/null || true

# Enzyme's LLVM plugin, matching the toolchain version. Keep the upstream
# Apache-2.0-with-LLVM-exceptions licence next to the redistributed object.
COPY vendor/LLVMEnzyme-19.so /usr/local/lib/LLVMEnzyme-19.so
COPY vendor/LICENSE.Enzyme.txt /usr/local/share/licenses/enzyme/LICENSE
RUN set -eux; \
    test -s /usr/local/lib/LLVMEnzyme-${LLVM_VERSION}.so; \
    echo "${ENZYME_SHA256}  /usr/local/lib/LLVMEnzyme-${LLVM_VERSION}.so" | sha256sum -c -

ENV ENZYME_LIB=/usr/local/lib/LLVMEnzyme-19.so

# Pre-install what the Tesseract runtime stage would otherwise fetch itself.
# That stage runs an unconditional `apt-get update && apt-get install`, and on
# a flaky link the index fetch is the most likely thing to fail; having the
# packages already present makes the install a no-op. The LLVM source list is
# dropped afterwards so the update only has to talk to the Debian mirror.
RUN set -eux; \
    for i in 1 2 3 4 5; do \
        apt-get update -qq && \
        apt-get install -y --no-install-recommends \
            libnss-wrapper python3 python3-venv python3-dev && break || \
        { echo "retry $i"; apt-get clean; rm -rf /var/lib/apt/lists/*; sleep 15; }; \
    done; \
    rm -f /etc/apt/sources.list.d/llvm.list; \
    rm -rf /var/lib/apt/lists/*

# Fail the build here rather than inside a Tesseract build if anything is missing.
RUN set -eux; \
    opt --version | head -2; \
    clang --version | head -1; \
    (flang-new --version || flang --version) | head -1; \
    ls -la "${ENZYME_LIB}"
