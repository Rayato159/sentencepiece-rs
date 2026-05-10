use crate::darts::DoubleArray;
use crate::proto::{NormalizerSpec, TrainerSpec};
use crate::util::{SPACE_SYMBOL, first_char_len};
use crate::{Error, Result};

/// SentencePiece-compatible normalizer.
///
/// It supports the model flags used by the runtime and the precompiled
/// normalization trie embedded in standard SentencePiece model files.
#[derive(Clone, Debug)]
pub struct Normalizer {
    spec: NormalizerSpec,
    treat_whitespace_as_suffix: bool,
    charsmap: Option<PrecompiledCharsMap>,
    user_symbols: Vec<String>,
}

#[derive(Clone, Debug)]
struct PrecompiledCharsMap {
    trie: DoubleArray,
    normalized: Vec<u8>,
}

impl Normalizer {
    pub(crate) fn new(spec: NormalizerSpec, trainer_spec: &TrainerSpec) -> Result<Self> {
        let charsmap = if spec.precompiled_charsmap.is_empty() {
            None
        } else {
            Some(PrecompiledCharsMap::decode(&spec.precompiled_charsmap)?)
        };

        Ok(Self {
            spec,
            treat_whitespace_as_suffix: trainer_spec.treat_whitespace_as_suffix,
            charsmap,
            user_symbols: Vec::new(),
        })
    }

    pub(crate) fn new_denormalizer(spec: NormalizerSpec) -> Result<Self> {
        let charsmap = if spec.precompiled_charsmap.is_empty() {
            None
        } else {
            Some(PrecompiledCharsMap::decode(&spec.precompiled_charsmap)?)
        };

        Ok(Self {
            spec,
            treat_whitespace_as_suffix: false,
            charsmap,
            user_symbols: Vec::new(),
        })
    }

    pub(crate) fn set_user_symbols(&mut self, mut user_symbols: Vec<String>) {
        user_symbols.sort_by_key(|symbol| std::cmp::Reverse(symbol.len()));
        self.user_symbols = user_symbols;
    }

    /// Normalizes a UTF-8 string with the model's SentencePiece rules.
    pub fn normalize(&self, input: &str) -> Result<String> {
        if input.is_empty() {
            return Ok(String::new());
        }

        let mut cursor = 0;
        let bytes = input.as_bytes();

        if self.spec.remove_extra_whitespaces {
            while cursor < bytes.len() {
                let (normalized, consumed) = self.normalize_prefix(&input[cursor..]);
                if normalized.as_slice() != b" " {
                    break;
                }
                cursor += consumed;
            }
        }

        if cursor == bytes.len() {
            return Ok(String::new());
        }

        let mut output = Vec::with_capacity((bytes.len() - cursor) * 3);
        let add_ws = |output: &mut Vec<u8>| {
            if self.spec.escape_whitespaces {
                output.extend_from_slice(SPACE_SYMBOL.as_bytes());
            } else {
                output.push(b' ');
            }
        };

        if !self.treat_whitespace_as_suffix && self.spec.add_dummy_prefix {
            add_ws(&mut output);
        }

        let mut is_prev_space = self.spec.remove_extra_whitespaces;
        while cursor < bytes.len() {
            let (mut normalized, consumed) = self.normalize_prefix(&input[cursor..]);

            while is_prev_space && normalized.first() == Some(&b' ') {
                normalized.remove(0);
            }

            if !normalized.is_empty() {
                for byte in normalized.iter().copied() {
                    if self.spec.escape_whitespaces && byte == b' ' {
                        output.extend_from_slice(SPACE_SYMBOL.as_bytes());
                    } else {
                        output.push(byte);
                    }
                }
                is_prev_space = normalized.last() == Some(&b' ');
            }

            cursor += consumed;
            if !self.spec.remove_extra_whitespaces {
                is_prev_space = false;
            }
        }

        if self.spec.remove_extra_whitespaces {
            let suffix = if self.spec.escape_whitespaces {
                SPACE_SYMBOL.as_bytes()
            } else {
                b" "
            };
            while output.ends_with(suffix) {
                let new_len = output.len() - suffix.len();
                output.truncate(new_len);
            }
        }

        if self.treat_whitespace_as_suffix && self.spec.add_dummy_prefix {
            add_ws(&mut output);
        }

        String::from_utf8(output)
            .map_err(|_| Error::model_parse("normalization produced invalid UTF-8"))
    }

    pub(crate) fn add_dummy_prefix(&self) -> bool {
        self.spec.add_dummy_prefix
    }

    pub(crate) fn remove_extra_whitespaces(&self) -> bool {
        self.spec.remove_extra_whitespaces
    }

    fn normalize_prefix(&self, input: &str) -> (Vec<u8>, usize) {
        if input.is_empty() {
            return (Vec::new(), 0);
        }

        if let Some(symbol) = self
            .user_symbols
            .iter()
            .find(|symbol| input.as_bytes().starts_with(symbol.as_bytes()))
        {
            return (symbol.as_bytes().to_vec(), symbol.len());
        }

        if let Some(charsmap) = &self.charsmap
            && let Some((offset, length)) = charsmap.longest_match(input.as_bytes())
            && let Some(normalized) = charsmap.normalized_at(offset)
        {
            return (normalized.to_vec(), length);
        }

        let len = first_char_len(input);
        (input.as_bytes()[..len].to_vec(), len)
    }
}

impl PrecompiledCharsMap {
    fn decode(blob: &[u8]) -> Result<Self> {
        if blob.len() <= 4 {
            return Err(Error::model_parse("normalization rule blob is broken"));
        }

        let trie_blob_size = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]) as usize;
        if trie_blob_size >= blob.len() {
            return Err(Error::model_parse(
                "normalization trie data exceeds the input blob size",
            ));
        }
        if trie_blob_size < 1024 || (trie_blob_size & 0x3ff) != 0 {
            return Err(Error::model_parse(
                "normalization trie data size is not divisible by 1024",
            ));
        }

        let trie_start = 4;
        let trie_end = trie_start + trie_blob_size;
        let normalized = blob[trie_end..].to_vec();
        if normalized.is_empty() || normalized.last() != Some(&0) {
            return Err(Error::model_parse(
                "normalization data block must be null-terminated",
            ));
        }

        Ok(Self {
            trie: DoubleArray::from_le_blob(&blob[trie_start..trie_end])?,
            normalized,
        })
    }

    fn longest_match(&self, input: &[u8]) -> Option<(usize, usize)> {
        self.trie
            .common_prefix_search(input)
            .into_iter()
            .max_by_key(|(_, length)| *length)
            .map(|(offset, length)| (offset as usize, length))
    }

    fn normalized_at(&self, offset: usize) -> Option<&[u8]> {
        if offset >= self.normalized.len() {
            return None;
        }
        let tail = &self.normalized[offset..];
        let end = tail.iter().position(|byte| *byte == 0)?;
        Some(&tail[..end])
    }
}
