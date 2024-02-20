//! MBart with MOE
use std::collections::HashMap;

use candle_core::{Module, Result, Tensor};
use candle_nn::{Activation, VarBuilder};

// TODO this is a placeholder

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct MBartConfig {
    activation_function: Activation,
    id2label: HashMap<String, String>,
    langs: HashMap<String, usize>,
    vocab_size: usize,
    moe_layers: Vec<usize>,
    d_model: usize,
    d_expert: usize,
    decoder_attention_heads: usize,
    decoder_ffn_dim: usize,
    decoder_layers: usize,
    kv_heads: usize,
    max_position_embeddings: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct MBart {}

impl MBart {
    pub(crate) fn new(_config: &MBartConfig, _vb: VarBuilder) -> Result<Self> {
        Ok(Self {})
    }
}

impl Module for MBart {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        Ok(input.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mbart_config() {
        let raw_json = r#"{
            "_name_or_path": "",
            "activation_dropout": 0.0,
            "activation_function": "gelu",
            "add_cross_attention": true,
            "add_final_layer_norm": true,
            "architectures": [
              "MBartForCausalLM"
            ],
            "attention_dropout": 0.0,
            "bad_words_ids": null,
            "begin_suppress_tokens": null,
            "bos_token_id": 0,
            "chunk_size_feed_forward": 0,
            "classifier_dropout": 0.0,
            "cross_attention_hidden_size": null,
            "d_expert": 1024,
            "d_model": 1024,
            "decoder_attention_heads": 16,
            "decoder_ffn_dim": 4096,
            "decoder_layerdrop": 0.0,
            "decoder_layers": 7,
            "decoder_start_token_id": null,
            "diversity_penalty": 0.0,
            "do_sample": false,
            "dropout": 0.1,
            "early_stopping": false,
            "encoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "encoder_layerdrop": 0.0,
            "encoder_layers": 12,
            "encoder_no_repeat_ngram_size": 0,
            "eos_token_id": 2,
            "exponential_decay_length_penalty": null,
            "finetuning_task": null,
            "forced_bos_token_id": null,
            "forced_eos_token_id": 2,
            "id2label": {
              "0": "LABEL_0",
              "1": "LABEL_1"
            },
            "init_std": 0.02,
            "is_decoder": true,
            "is_encoder_decoder": false,
            "kv_heads": 4,
            "label2id": {
              "LABEL_0": 0,
              "LABEL_1": 1
            },
            "langs": {
              "af": 65539,
              "am": 65540,
              "ar": 65541,
              "as": 65542,
              "az": 65543,
              "be": 65544,
              "bg": 65545,
              "bn": 65546,
              "br": 65547,
              "bs": 65548,
              "ca": 65549,
              "cs": 65550,
              "cy": 65551,
              "da": 65552,
              "de": 65553,
              "el": 65554,
              "en": 65555,
              "eo": 65556,
              "es": 65557,
              "et": 65558,
              "eu": 65559,
              "fa": 65560,
              "fi": 65561,
              "fr": 65562,
              "fy": 65563,
              "ga": 65564,
              "gd": 65565,
              "gl": 65566,
              "gu": 65567,
              "ha": 65568,
              "he": 65569,
              "hi": 65570,
              "hr": 65571,
              "hu": 65572,
              "hy": 65573,
              "id": 65574,
              "is": 65575,
              "it": 65576,
              "ja": 65577,
              "jv": 65578,
              "ka": 65579,
              "kk": 65580,
              "km": 65581,
              "kn": 65582,
              "ko": 65583,
              "ku": 65584,
              "ky": 65585,
              "la": 65586,
              "lo": 65587,
              "lt": 65588,
              "lv": 65589,
              "mg": 65590,
              "mk": 65591,
              "ml": 65592,
              "mn": 65593,
              "mr": 65594,
              "ms": 65595,
              "my": 65596,
              "ne": 65597,
              "nl": 65598,
              "no": 65599,
              "om": 65600,
              "or": 65601,
              "pa": 65602,
              "pl": 65603,
              "ps": 65604,
              "pt": 65605,
              "ro": 65606,
              "ru": 65607,
              "sa": 65608,
              "sd": 65609,
              "si": 65610,
              "sk": 65611,
              "sl": 65612,
              "so": 65613,
              "sq": 65614,
              "sr": 65615,
              "su": 65616,
              "sv": 65617,
              "sw": 65618,
              "ta": 65619,
              "te": 65620,
              "th": 65621,
              "tl": 65622,
              "tr": 65623,
              "ug": 65624,
              "uk": 65625,
              "ur": 65626,
              "uz": 65627,
              "vi": 65628,
              "xh": 65629,
              "yi": 65630,
              "zh": 65631
            },
            "length_penalty": 1.0,
            "max_length": 256,
            "max_position_embeddings": 1536,
            "min_length": 0,
            "model_type": "mbart",
            "moe_layers": [
              3
            ],
            "no_repeat_ngram_size": 0,
            "num_beam_groups": 1,
            "num_beams": 1,
            "num_decoder_layers": 6,
            "num_hidden_layers": 12,
            "num_return_sequences": 1,
            "output_attentions": false,
            "output_hidden_states": false,
            "output_scores": false,
            "pad_token_id": 1,
            "prefix": null,
            "problem_type": null,
            "pruned_heads": {},
            "remove_invalid_values": false,
            "repetition_penalty": 1.0,
            "return_dict": true,
            "return_dict_in_generate": false,
            "scale_embedding": true,
            "sep_token_id": null,
            "suppress_tokens": null,
            "task_specific_params": null,
            "temperature": 1.0,
            "tf_legacy_loss": false,
            "tie_encoder_decoder": false,
            "tie_word_embeddings": true,
            "tokenizer_class": null,
            "top_k": 50,
            "top_p": 1.0,
            "torch_dtype": "float32",
            "torchscript": false,
            "typical_p": 1.0,
            "use_bfloat16": false,
            "use_cache": true,
            "use_moe": true,
            "vocab_size": 65792
          }"#;
        let deserialized: MBartConfig = serde_json::from_str(raw_json).unwrap();
        assert_eq!(deserialized.langs.len(), 93);
        assert_eq!(deserialized.vocab_size, 65792);
        assert_eq!(deserialized.moe_layers, vec![3]);
        assert_eq!(deserialized.d_model, 1024);
        assert_eq!(deserialized.d_expert, 1024);
        assert_eq!(deserialized.decoder_attention_heads, 16);
        assert_eq!(deserialized.decoder_ffn_dim, 4096);
        assert_eq!(deserialized.decoder_layers, 7);
        assert_eq!(deserialized.kv_heads, 4);
        assert_eq!(deserialized.max_position_embeddings, 1536);
    }
}
