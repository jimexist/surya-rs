# surya-rs

[![Build](https://github.com/Jimexist/surya-rs/actions/workflows/builld.yaml/badge.svg)](https://github.com/Jimexist/surya-rs/actions/workflows/builld.yaml)
![Crates.io Version](https://img.shields.io/crates/v/surya)

Rust implementation of [surya][surya], a multilingual document OCR toolkit. The implementation is based on a modified version of Segformer.

## Roadmap

This project is still in development, feel free to star and check back.

- [x] model structure, segformer (for inference only)
- [x] weights loading
- [ ] image input pre-processing
- [ ] heatmap and bboxes
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
# run this if you have a mac with M1/2/3 chip
cargo install --path . --features=cli,metal --bin surya
# run this on other architectures
cargo install --path . --features=cli --bin surya
```

The binary when built does _not_ include the weights file itself, and will instead download via the HuggingFace Hub API. Once downloaded, the weights file will be cached in the HuggingFace cache directory.

Check `-h` for help:

```text
❯ surya --help
Surya is a multilingual document OCR toolkit, original implementation in Python and PyTorch

Usage: surya [OPTIONS] --image <IMAGE>

Options:
      --image <IMAGE>                path to image
      --model-repo <MODEL_REPO>      model's hugging face repo [default: vikp/line_detector]
      --weights-name <WEIGHTS_NAME>  model's weights name [default: model.safetensors]
      --device-type <DEVICE_TYPE>    [default: cpu] [possible values: cpu, gpu, metal]
  -h, --help                         Print help
  -V, --version                      Print version
```

You can use this to control logging level:

```bash
export RUST_LOG=info # or debug, warn, etc.
```

## Library

This lib is also published as a trait for other rust projects to use.

[surya]: https://github.com/VikParuchuri/surya
