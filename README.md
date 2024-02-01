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

Install `llvm` and `opencv` (example on Mac):

```bash
brew install llvm opencv
```

Build and install the binary:

```bash
# run this first on Mac if you have a M1 chip
export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select --print-path)/usr/lib/"
# run this first on other Mac
export DYLD_FALLBACK_LIBRARY_PATH="$(xcode-select --print-path)/Toolchains/XcodeDefault.xctoolchain/"
# optionally you can include features like accelerate, metal, mkl, etc.
cargo install --path . --features=cli
```

The binary when built does _not_ include the weights file itself, and will instead download via the HuggingFace Hub API. Once downloaded, the weights file will be cached in the HuggingFace cache directory.

Check `-h` for help:

```text
Surya is a multilingual document OCR toolkit, original implementation in Python and PyTorch

Usage: surya [OPTIONS] <IMAGE>

Arguments:
  <IMAGE>  path to image

Options:
      --batch-size <BATCH_SIZE>
          detection batch size, if not supplied defaults to 2 on CPU and 16 on GPU
      --model-repo <MODEL_REPO>
          detection model's hugging face repo [default: vikp/line_detector]
      --weights-file-name <WEIGHTS_FILE_NAME>
          detection model's weights file name [default: model.safetensors]
      --config-file-name <CONFIG_FILE_NAME>
          detection model's config file name [default: config.json]
      --non-max-suppression-threshold <NON_MAX_SUPPRESSION_THRESHOLD>
          a value between 0.0 and 1.0 to filter low density part of heatmap [default: 0.35]
      --extract-text-threshold <EXTRACT_TEXT_THRESHOLD>
          a value between 0.0 and 1.0 to filter out bbox with low heatmap density [default: 0.6]
      --bbox-area-threshold <BBOX_AREA_THRESHOLD>
          a pixel threshold to filter out small area bbox [default: 10]
      --polygons
          whether to output polygons json file
      --image
          whether to generate bbox image
      --heatmap
          whether to generate heatmap
      --affinity-map
          whether to generate affinity map
      --output-dir <OUTPUT_DIR>
          output directory, under which the input image will be generating a subdirectory [default: ./surya_output]
      --device <DEVICE_TYPE>
          device type, if not specified will try to use GPU or Metal [possible values: cpu, gpu, metal]
      --verbose
          whether to enable verbose mode
  -h, --help
          Print help
  -V, --version
          Print version
```

You can also use this to control logging level:

```bash
export SURYA_LOG=warn # or debug, warn, etc.
```

## Library

This lib is also published as a trait for other rust projects to use.

[surya]: https://github.com/VikParuchuri/surya
[opencv]: https://crates.io/crates/opencv
