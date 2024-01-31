# surya-rs

[![Build](https://github.com/Jimexist/surya-rs/actions/workflows/builld.yaml/badge.svg)](https://github.com/Jimexist/surya-rs/actions/workflows/builld.yaml)
[![Crates.io Version](https://img.shields.io/crates/v/surya)](https://crates.io/crates/surya)

Rust implementation of [surya][surya], a multilingual document OCR toolkit.
The implementation is based on a modified version of Segformer and [OpenCV][opencv].

Please refer to the original project for more details on licensing of the weights.

## Roadmap

This project is still in development, feel free to star and check back.

- [x] model structure, segformer (for inference only)
- [x] weights loading
- [x] image input pre-processing
- [x] heatmap and affinity map
- [x] bboxes
- [x] image splitting and stitching
- [ ] text recognition
- [ ] benchmark
- [ ] quantifications

## How to build and install

Setup rust toolchain if you haven't yet:

```bash
# visit https://rustup.rs/ for more detailed information
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Build and install the binary:

```bash
# run this first on Mac if you have a M1 chip
export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select --print-path)/usr/lib/"
# run this first on other Mac
export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select --print-path)/Toolchains/XcodeDefault.xctoolchain/"
# run this if you have a mac with Metal support
cargo install --path . --features=cli,metal --bin surya
# run this on other architectures
cargo install --path . --features=cli --bin surya
```

The binary when built does _not_ include the weights file itself, and will instead download via the HuggingFace Hub API. Once downloaded, the weights file will be cached in the HuggingFace cache directory.

Check `-h` for help:

```text
Surya is a multilingual document OCR toolkit, original implementation in Python and PyTorch

Usage: surya [OPTIONS] <IMAGE>

Arguments:
  <IMAGE>  path to image

Options:
      --model-repo <MODEL_REPO>
          model's hugging face repo [default: vikp/line_detector]
      --weights-file-name <WEIGHTS_FILE_NAME>
          model's weights file name [default: model.safetensors]
      --config-file-name <CONFIG_FILE_NAME>
          model's config file name [default: config.json]
      --generate-bbox-image
          whether to generate bbox image
      --generate-heatmap
          whether to generate heatmap
      --generate-affinity-map
          whether to generate affinity map
      --output-dir <OUTPUT_DIR>
          output directory, each file will be generating a subdirectory under this directory [default: ./surya_output]
      --device-type <DEVICE_TYPE>
          [default: cpu] [possible values: cpu, gpu, metal]
  -h, --help
          Print help
  -V, --version
          Print version
```

You can use this to control logging level:

```bash
export RUST_LOG=info # or debug, warn, etc.
```

## Library

This lib is also published as a trait for other rust projects to use.

[surya]: https://github.com/VikParuchuri/surya
[opencv]: https://crates.io/crates/opencv
