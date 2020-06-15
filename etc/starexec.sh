#!/bin/sh

rm -rf lazycop.zip bin/
mkdir bin/
RUSTFLAGS="-C codegen-units=1 -C lto -C debuginfo=0 -C target-feature=avx" cargo build --release --target x86_64-unknown-linux-musl
cp /usr/bin/vampire bin/
strip target/x86_64-unknown-linux-musl/release/lazycop
cp target/x86_64-unknown-linux-musl/release/lazycop bin/
cp etc/starexec_run_default bin/
zip lazycop.zip -r bin/
rm -rf bin/
