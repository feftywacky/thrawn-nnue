# Running Tests

Run the full test suite from the repository root:

```bash
python3.11 -m unittest discover -s tests -v
```

If you want to force a clean rebuild of the native `.binpack` bridge first:

```bash
rm -rf build/native_binpack
python3.11 -m unittest discover -s tests -v
```

## What the test suite covers

- HalfKP feature indexing and training-time `P` factorization
- Incremental accumulator updates versus own-king refreshes
- Native `.binpack` fixture generation, parsing, and HalfKP batch loading
- Position-budgeted training loop behavior and CP/WDL validation metrics
- `.nnue` version-5 export header and tensor layout round-trip
- `verify-export` material sanity reporting
- Checkpoint metadata round-trip when `torch` is installed

## Notes

- Some tests require the native library to compile successfully through CMake.
- Tests that exercise training are skipped automatically if `torch` is not installed.
- The checkpoint metadata test is skipped automatically if `torch` is not installed.
