#!/bin/sh

CARGO_PROFILE_RELEASE_LTO=fat
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
RUSTFLAGS="-C debuginfo=0 -C target-feature=avx"
cargo build --release --target x86_64-unknown-linux-musl

rm -rf lazycop.zip bin/
mkdir bin/
cp /usr/bin/vampire bin/
strip target/x86_64-unknown-linux-musl/release/lazycop
cp target/x86_64-unknown-linux-musl/release/lazycop bin/
cp etc/starexec_run_default bin/
zip lazycop.zip -r bin/
rm -rf bin/
