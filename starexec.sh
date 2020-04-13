#!/bin/sh

rm -rf lazycop.zip bin/
mkdir bin/
cargo build --release --target x86_64-unknown-linux-musl
cp /usr/bin/vampire bin/
cp target/x86_64-unknown-linux-musl/release/lazycop bin/
cp starexec_run_default bin/
zip lazycop.zip -r bin/
rm -rf bin/
