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

- A-768 feature indexing and dual-perspective transforms
- Incremental accumulator updates versus full refresh
- Native `.binpack` fixture generation, parsing, and batch loading
- Position-budgeted training loop behavior and validation cadence
- Validation metrics, including WDL accuracy and teacher/result disagreement reporting
- `.nnue` export header and tensor layout round-trip
- Checkpoint metadata round-trip when `torch` is installed

## Notes

- Some tests require the native library to compile successfully through CMake.
- Tests that exercise training are skipped automatically if `torch` is not installed.
- The checkpoint metadata test is skipped automatically if `torch` is not installed.
