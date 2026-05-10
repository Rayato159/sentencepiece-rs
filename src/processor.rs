use std::fs;
use std::path::Path;

use crate::model::{SentencePieceModel, Token, merge_or_byte_fallback_tokens};
use crate::normalizer::Normalizer;
use crate::util::{SPACE_SYMBOL, piece_to_byte, replace_space_symbol};
use crate::{Error, Result};

/// Main API for loading a SentencePiece model and tokenizing text.
#[derive(Clone, Debug)]
pub struct SentencePieceProcessor {
    model: SentencePieceModel,
    normalizer: Normalizer,
    denormalizer: Option<Normalizer>,
    encode_extra_options: Vec<ExtraOption>,
    decode_extra_options: Vec<ExtraOption>,
}

#[derive(Clone, Copy, Debug)]
enum ExtraOption {
    Bos,
    Eos,
    Reverse,
    UnkPiece,
}

impl SentencePieceProcessor {
    /// Loads a SentencePiece `.model` / `.spm` file from disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_serialized_model(&bytes)
    }

    /// Loads a SentencePiece model from serialized protobuf bytes.
    pub fn from_serialized_model(bytes: &[u8]) -> Result<Self> {
        Self::from_model(SentencePieceModel::from_slice(bytes)?)
    }

    /// Creates a processor from an already loaded model.
    pub fn from_model(model: SentencePieceModel) -> Result<Self> {
        let mut normalizer = Normalizer::new(model.normalizer_spec(), model.trainer_spec())?;
        let user_symbols = model.user_symbols();
        crate::model::validate_no_duplicate_user_symbols(&user_symbols)?;
        normalizer.set_user_symbols(user_symbols);

        let denormalizer = model
            .denormalizer_spec()
            .filter(|spec| !spec.precompiled_charsmap.is_empty())
            .map(Normalizer::new_denormalizer)
            .transpose()?;

        Ok(Self {
            model,
            normalizer,
            denormalizer,
            encode_extra_options: Vec::new(),
            decode_extra_options: Vec::new(),
        })
    }

    /// Returns the loaded model metadata.
    pub fn model(&self) -> &SentencePieceModel {
        &self.model
    }

    /// Normalizes text with the model's normalizer.
    pub fn normalize(&self, input: &str) -> Result<String> {
        self.normalizer.normalize(input)
    }

    /// Encodes text into SentencePiece token strings.
    pub fn encode(&self, input: &str) -> Result<Vec<String>> {
        Ok(self
            .encode_tokens(input)?
            .into_iter()
            .map(|token| token.piece)
            .collect())
    }

    /// Encodes text into vocabulary ids.
    pub fn encode_to_ids(&self, input: &str) -> Result<Vec<usize>> {
        Ok(self
            .encode_tokens(input)?
            .into_iter()
            .map(|token| token.id)
            .collect())
    }

    /// Decodes SentencePiece token strings into text.
    pub fn decode<T: AsRef<str>>(&self, pieces: &[T]) -> Result<String> {
        self.decode_pieces(pieces)
    }

    /// Decodes SentencePiece token strings into text.
    pub fn decode_pieces<T: AsRef<str>>(&self, pieces: &[T]) -> Result<String> {
        let mut tokens = pieces
            .iter()
            .map(|piece| {
                let piece = piece.as_ref().to_owned();
                let id = self.model.piece_to_id(&piece);
                Token { piece, id }
            })
            .collect::<Vec<_>>();

        self.apply_extra_options(&self.decode_extra_options, &mut tokens);
        self.decode_tokens(&tokens)
    }

    /// Decodes vocabulary ids into text.
    pub fn decode_ids(&self, ids: &[usize]) -> Result<String> {
        let mut tokens = Vec::with_capacity(ids.len());
        for id in ids {
            let piece = self.model.id_to_piece(*id)?.to_owned();
            tokens.push(Token { piece, id: *id });
        }
        self.apply_extra_options(&self.decode_extra_options, &mut tokens);
        self.decode_tokens(&tokens)
    }

    /// Configures encode options like `bos`, `eos`, `reverse`, and `unk_piece`.
    ///
    /// Use a colon-separated string, e.g. `bos:eos`.
    pub fn set_encode_extra_options(&mut self, options: &str) -> Result<()> {
        self.encode_extra_options = self.parse_extra_options(options)?;
        Ok(())
    }

    /// Configures decode options like `bos`, `eos`, and `reverse`.
    ///
    /// Use a colon-separated string, e.g. `reverse`.
    pub fn set_decode_extra_options(&mut self, options: &str) -> Result<()> {
        self.decode_extra_options = self.parse_extra_options(options)?;
        Ok(())
    }

    /// Unknown token id.
    pub fn unk_id(&self) -> usize {
        self.model.unk_id()
    }

    /// BOS token id, if defined.
    pub fn bos_id(&self) -> Option<usize> {
        self.model.bos_id()
    }

    /// EOS token id, if defined.
    pub fn eos_id(&self) -> Option<usize> {
        self.model.eos_id()
    }

    /// PAD token id, if defined.
    pub fn pad_id(&self) -> Option<usize> {
        self.model.pad_id()
    }

    /// Looks up a piece id, falling back to `<unk>`.
    pub fn piece_to_id(&self, piece: &str) -> usize {
        self.model.piece_to_id(piece)
    }

    /// Looks up a piece string by id.
    pub fn id_to_piece(&self, id: usize) -> Result<&str> {
        self.model.id_to_piece(id)
    }

    fn encode_tokens(&self, input: &str) -> Result<Vec<Token>> {
        let normalized = self.normalizer.normalize(input)?;
        let tokens = self.model.encode_normalized(&normalized)?;
        let mut tokens = merge_or_byte_fallback_tokens(&self.model, tokens)?;
        self.apply_extra_options(&self.encode_extra_options, &mut tokens);
        Ok(tokens)
    }

    fn decode_tokens(&self, tokens: &[Token]) -> Result<String> {
        let mut text = String::new();
        let mut byte_buffer = Vec::new();
        let mut is_bos_ws = true;
        let mut bos_ws_seen = false;

        for token in tokens {
            if self.model.is_byte(token.id) {
                if let Some(byte) = piece_to_byte(&token.piece) {
                    byte_buffer.push(byte);
                    continue;
                }
                return Err(Error::model_parse(format!(
                    "invalid byte piece {}",
                    token.piece
                )));
            }

            flush_byte_buffer(&mut byte_buffer, &mut text);

            if bos_ws_seen || !text.is_empty() {
                is_bos_ws = false;
            }

            let (decoded, consumed_bos_ws) = self.decode_sentence_piece(token, is_bos_ws)?;
            bos_ws_seen = consumed_bos_ws;
            text.push_str(&decoded);
        }

        flush_byte_buffer(&mut byte_buffer, &mut text);

        if let Some(denormalizer) = &self.denormalizer {
            denormalizer.normalize(&text)
        } else {
            Ok(text)
        }
    }

    fn decode_sentence_piece(&self, token: &Token, is_bos_ws: bool) -> Result<(String, bool)> {
        if self.model.is_control(token.id) {
            return Ok((String::new(), false));
        }

        if self.model.is_unknown(token.id) {
            let unk_piece = self.model.id_to_piece(token.id)?;
            if unk_piece == token.piece {
                return Ok((self.model.unk_surface().to_owned(), false));
            }
            return Ok((token.piece.clone(), false));
        }

        let mut piece = token.piece.as_str();
        let mut has_bos_ws = false;
        if is_bos_ws
            && (self.normalizer.add_dummy_prefix() || self.normalizer.remove_extra_whitespaces())
            && piece.starts_with(SPACE_SYMBOL)
        {
            piece = &piece[SPACE_SYMBOL.len()..];
            has_bos_ws = true;
            if self.normalizer.remove_extra_whitespaces() {
                has_bos_ws = false;
            }
        }

        Ok((replace_space_symbol(piece), has_bos_ws))
    }

    fn parse_extra_options(&self, options: &str) -> Result<Vec<ExtraOption>> {
        if options.is_empty() {
            return Ok(Vec::new());
        }

        options
            .split(':')
            .map(|option| match option {
                "bos" => {
                    self.ensure_special_defined(self.model.bos_piece())?;
                    Ok(ExtraOption::Bos)
                }
                "eos" => {
                    self.ensure_special_defined(self.model.eos_piece())?;
                    Ok(ExtraOption::Eos)
                }
                "reverse" => Ok(ExtraOption::Reverse),
                "unk" | "unk_piece" => Ok(ExtraOption::UnkPiece),
                other => Err(Error::invalid_input(format!(
                    "extra option {other:?} is not available"
                ))),
            })
            .collect()
    }

    fn ensure_special_defined(&self, piece: &str) -> Result<()> {
        let id = self.model.piece_to_id(piece);
        if self.model.is_unknown(id) {
            Err(Error::invalid_input(format!(
                "id for special piece {piece:?} is not defined"
            )))
        } else {
            Ok(())
        }
    }

    fn apply_extra_options(&self, options: &[ExtraOption], tokens: &mut Vec<Token>) {
        for option in options {
            match option {
                ExtraOption::Reverse => tokens.reverse(),
                ExtraOption::Eos => {
                    let id = self.model.piece_to_id(self.model.eos_piece());
                    tokens.push(Token {
                        piece: self.model.eos_piece().to_owned(),
                        id,
                    });
                }
                ExtraOption::Bos => {
                    let id = self.model.piece_to_id(self.model.bos_piece());
                    tokens.insert(
                        0,
                        Token {
                            piece: self.model.bos_piece().to_owned(),
                            id,
                        },
                    );
                }
                ExtraOption::UnkPiece => {
                    for token in tokens.iter_mut() {
                        if self.model.is_unknown(token.id) {
                            token.piece = self.model.unk_piece().to_owned();
                        }
                    }
                }
            }
        }
    }
}

fn flush_byte_buffer(byte_buffer: &mut Vec<u8>, text: &mut String) {
    if byte_buffer.is_empty() {
        return;
    }
    match std::str::from_utf8(byte_buffer) {
        Ok(valid) => text.push_str(valid),
        Err(_) => text.push_str(&String::from_utf8_lossy(byte_buffer)),
    }
    byte_buffer.clear();
}
