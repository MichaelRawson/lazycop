#!/bin/sh

CARGO_PROFILE_RELEASE_LTO=fat
CARGO_PROFILE_RELEASE_CODEGEN_UNITS=1
CARGO_PROFILE_RELEASE_DEBUG=0
CARGO_PROFILE_RELEASE_PANIC="abort"
cargo build --release --target x86_64-unknown-linux-musl

rm -rf lazycop.zip bin/
mkdir bin/
strip target/x86_64-unknown-linux-musl/release/lazycop
cp target/x86_64-unknown-linux-musl/release/lazycop bin/
cp etc/starexec_run_default bin/
zip lazycop.zip -r bin/
rm -rf bin/
