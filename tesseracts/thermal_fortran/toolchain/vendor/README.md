# Vendored Enzyme LLVM-19 plugin

`LLVMEnzyme-19.so` is the official Enzyme nightly LLVM-19 plugin retrieved from
<https://github.com/EnzymeAD/Enzyme/releases/download/nightly/LLVMEnzyme-19.so>
on 2026-08-14.

- Size: 8,050,632 bytes
- SHA-256: `5b43014ab23fdf212b5c0852e5ae1d2e9d3062bf0aa2323bbbf63b33369ef031`
- Upstream: <https://github.com/EnzymeAD/Enzyme>
- Licence: Apache-2.0 with LLVM exceptions; see `LICENSE.Enzyme.txt`
- Licence snapshot commit: `25933271ffb8a475d66766e0c75929e81d4df432`
- Licence SHA-256: `f2db94d30c9657f2556732f3e80973d49fc4d093eede0a54ffda88152296f695`

The object is stored in the repository because the upstream `nightly` release
asset is deleted and recreated. The Docker build independently verifies the
digest before using the plugin.
