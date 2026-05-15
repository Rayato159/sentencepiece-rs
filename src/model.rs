use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use crate::proto::{ModelProto, SentencePiece as ProtoPiece, TrainerSpec};
pub use crate::proto::{ModelType, PieceType};
use crate::util::{SPACE_SYMBOL, byte_to_piece, char_len_at, piece_to_byte};
use crate::{Error, Result};

const UNK_PENALTY: f32 = 10.0;

/// A vocabulary entry from a SentencePiece model.
#[derive(Clone, Debug)]
pub struct Piece {
    /// Token text, e.g. `▁hello`.
    pub piece: String,
    /// Log-probability or merge score stored by the model.
    pub score: f32,
    /// SentencePiece token kind.
    pub kind: PieceType,
}

#[derive(Clone, Debug)]
pub(crate) struct Token {
    pub(crate) piece: String,
    pub(crate) id: usize,
}

/// Loaded SentencePiece model and vocabulary metadata.
#[derive(Clone, Debug)]
pub struct SentencePieceModel {
    proto: ModelProto,
    pieces: Vec<Piece>,
    piece_to_id: HashMap<String, usize>,
    regular_piece_to_id: HashMap<String, usize>,
    by_first_byte: HashMap<u8, Vec<usize>>,
    user_symbols: Vec<String>,
    min_score: f32,
    unk_id: usize,
    byte_ids: [Option<usize>; 256],
}

#[derive(Clone, Debug)]
struct BestPathNode {
    id: usize,
    best_path_score: f32,
    starts_at: usize,
}

#[derive(Clone, Debug)]
struct BpeSymbol {
    piece: String,
    freeze: bool,
}

impl SentencePieceModel {
    /// Loads a serialized SentencePiece model from disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let bytes = fs::read(path)?;
        Self::from_slice(&bytes)
    }

    /// Loads a serialized SentencePiece model from memory.
    pub fn from_slice(bytes: &[u8]) -> Result<Self> {
        Self::from_proto(ModelProto::decode(bytes)?)
    }

    pub(crate) fn from_proto(proto: ModelProto) -> Result<Self> {
        let _special_id_hints = (
            proto.trainer_spec.unk_id,
            proto.trainer_spec.bos_id,
            proto.trainer_spec.eos_id,
            proto.trainer_spec.pad_id,
        );
        for sample in &proto.self_test_data.samples {
            let _sample_fields = (&sample.input, &sample.expected);
        }

        let mut pieces = Vec::with_capacity(proto.pieces.len());
        let mut piece_to_id = HashMap::with_capacity(proto.pieces.len());
        let mut regular_piece_to_id = HashMap::new();
        let mut by_first_byte: HashMap<u8, Vec<usize>> = HashMap::new();
        let mut user_symbols = Vec::new();
        let mut byte_ids = [None; 256];
        let mut unk_id = None;
        let mut byte_found = [false; 256];
        let mut min_score = f32::MAX;

        for (id, piece) in proto.pieces.iter().enumerate() {
            validate_piece(piece)?;
            if piece_to_id.insert(piece.piece.clone(), id).is_some() {
                return Err(Error::model_parse(format!(
                    "piece {:?} is already defined",
                    piece.piece
                )));
            }

            match piece.kind {
                PieceType::Normal | PieceType::UserDefined | PieceType::Unused => {
                    regular_piece_to_id.insert(piece.piece.clone(), id);
                    if let Some(first) = piece.piece.as_bytes().first().copied() {
                        by_first_byte.entry(first).or_default().push(id);
                    }
                    if piece.kind == PieceType::Normal {
                        min_score = min_score.min(piece.score);
                    }
                    if piece.kind == PieceType::UserDefined {
                        user_symbols.push(piece.piece.clone());
                    }
                }
                PieceType::Unknown => {
                    if unk_id.replace(id).is_some() {
                        return Err(Error::model_parse("unk is already defined"));
                    }
                }
                PieceType::Byte => {
                    if !proto.trainer_spec.byte_fallback {
                        return Err(Error::model_parse(format!(
                            "byte piece {} is present but byte_fallback is false",
                            piece.piece
                        )));
                    }
                    let byte = piece_to_byte(&piece.piece).ok_or_else(|| {
                        Error::model_parse(format!("byte piece {} is invalid", piece.piece))
                    })?;
                    byte_found[byte as usize] = true;
                    byte_ids[byte as usize] = Some(id);
                }
                PieceType::Control => {}
            }

            pieces.push(Piece {
                piece: piece.piece.clone(),
                score: piece.score,
                kind: piece.kind,
            });
        }

        let unk_id = unk_id.ok_or_else(|| Error::model_parse("unk is not defined"))?;

        if proto.trainer_spec.byte_fallback && byte_found.iter().any(|found| !found) {
            return Err(Error::model_parse(
                "there are not 256 byte pieces although byte_fallback is true",
            ));
        }

        if min_score == f32::MAX {
            min_score = 0.0;
        }

        Ok(Self {
            proto,
            pieces,
            piece_to_id,
            regular_piece_to_id,
            by_first_byte,
            user_symbols,
            min_score,
            unk_id,
            byte_ids,
        })
    }

    /// Number of pieces in the vocabulary.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// Model type stored in the trainer spec.
    pub fn model_type(&self) -> ModelType {
        self.proto.trainer_spec.model_type
    }

    /// Returns all pieces in vocabulary order.
    pub fn pieces(&self) -> &[Piece] {
        &self.pieces
    }

    /// Looks up a piece and returns its id, falling back to `<unk>`.
    pub fn piece_to_id(&self, piece: &str) -> usize {
        self.piece_to_id.get(piece).copied().unwrap_or(self.unk_id)
    }

    /// Looks up a piece and returns `None` if it is not in the vocabulary.
    pub fn try_piece_to_id(&self, piece: &str) -> Option<usize> {
        self.piece_to_id.get(piece).copied()
    }

    /// Returns the piece string for a vocabulary id.
    pub fn id_to_piece(&self, id: usize) -> Result<&str> {
        self.pieces
            .get(id)
            .map(|piece| piece.piece.as_str())
            .ok_or_else(|| Error::invalid_input(format!("invalid piece id: {id}")))
    }

    /// Returns true if the id is the unknown token.
    pub fn is_unknown(&self, id: usize) -> bool {
        self.pieces
            .get(id)
            .is_some_and(|piece| piece.kind == PieceType::Unknown)
    }

    /// Returns true if the id is a control token like `<s>` or `</s>`.
    pub fn is_control(&self, id: usize) -> bool {
        self.pieces
            .get(id)
            .is_some_and(|piece| piece.kind == PieceType::Control)
    }

    /// Returns true if the id is marked unused.
    pub fn is_unused(&self, id: usize) -> bool {
        self.pieces
            .get(id)
            .is_some_and(|piece| piece.kind == PieceType::Unused)
    }

    /// Returns true if the id is a byte-fallback piece.
    pub fn is_byte(&self, id: usize) -> bool {
        self.pieces
            .get(id)
            .is_some_and(|piece| piece.kind == PieceType::Byte)
    }

    /// Unknown token id.
    pub fn unk_id(&self) -> usize {
        self.unk_id
    }

    /// BOS token id if the model defines it as a control token.
    pub fn bos_id(&self) -> Option<usize> {
        self.special_control_id(&self.proto.trainer_spec.bos_piece)
    }

    /// EOS token id if the model defines it as a control token.
    pub fn eos_id(&self) -> Option<usize> {
        self.special_control_id(&self.proto.trainer_spec.eos_piece)
    }

    /// PAD token id if the model defines it as a control token.
    pub fn pad_id(&self) -> Option<usize> {
        self.special_control_id(&self.proto.trainer_spec.pad_piece)
    }

    pub(crate) fn trainer_spec(&self) -> &TrainerSpec {
        &self.proto.trainer_spec
    }

    pub(crate) fn normalizer_spec(&self) -> crate::proto::NormalizerSpec {
        self.proto.normalizer_spec.clone()
    }

    pub(crate) fn denormalizer_spec(&self) -> Option<crate::proto::NormalizerSpec> {
        self.proto.denormalizer_spec.clone()
    }

    pub(crate) fn user_symbols(&self) -> Vec<String> {
        self.user_symbols.clone()
    }

    pub(crate) fn byte_fallback_enabled(&self) -> bool {
        self.proto.trainer_spec.byte_fallback
    }

    pub(crate) fn unk_piece(&self) -> &str {
        &self.proto.trainer_spec.unk_piece
    }

    pub(crate) fn bos_piece(&self) -> &str {
        &self.proto.trainer_spec.bos_piece
    }

    pub(crate) fn eos_piece(&self) -> &str {
        &self.proto.trainer_spec.eos_piece
    }

    pub(crate) fn unk_surface(&self) -> &str {
        &self.proto.trainer_spec.unk_surface
    }

    pub(crate) fn byte_id(&self, byte: u8) -> Option<usize> {
        self.byte_ids[byte as usize]
    }

    pub(crate) fn encode_normalized(&self, normalized: &str) -> Result<Vec<Token>> {
        match self.proto.trainer_spec.model_type {
            ModelType::Unigram => self.encode_unigram(normalized),
            ModelType::Bpe => Ok(self.encode_bpe(normalized)),
            ModelType::Word => Ok(self.encode_word(normalized)),
            ModelType::Char => Ok(self.encode_char(normalized)),
        }
    }

    fn special_control_id(&self, piece: &str) -> Option<usize> {
        let id = self.try_piece_to_id(piece)?;
        self.is_control(id).then_some(id)
    }

    fn encode_unigram(&self, normalized: &str) -> Result<Vec<Token>> {
        if normalized.is_empty() {
            return Ok(Vec::new());
        }

        let len = normalized.len();
        let unk_score = self.min_score - UNK_PENALTY;
        let mut best_path_ends_at = vec![None::<BestPathNode>; len + 1];
        best_path_ends_at[0] = Some(BestPathNode {
            id: self.unk_id,
            best_path_score: 0.0,
            starts_at: 0,
        });

        for (starts_at, _ch) in normalized.char_indices() {
            let Some(best_here) = best_path_ends_at[starts_at].as_ref() else {
                continue;
            };
            let best_here_score = best_here.best_path_score;

            let mblen = char_len_at(normalized, starts_at);
            let mut has_single_node = false;
            let suffix = &normalized[starts_at..];
            if let Some(first) = suffix.as_bytes().first().copied()
                && let Some(ids) = self.by_first_byte.get(&first)
            {
                for &id in ids {
                    if self.is_unused(id) {
                        continue;
                    }
                    let piece = &self.pieces[id].piece;
                    if !suffix.starts_with(piece) {
                        continue;
                    }

                    let end = starts_at + piece.len();
                    let score = self.score_for(id, piece.len());
                    let candidate_score = best_here_score + score;
                    let target = &mut best_path_ends_at[end];
                    if target
                        .as_ref()
                        .is_none_or(|c| candidate_score > c.best_path_score)
                    {
                        *target = Some(BestPathNode {
                            id,
                            best_path_score: candidate_score,
                            starts_at,
                        });
                    }

                    if piece.len() == mblen {
                        has_single_node = true;
                    }
                }
            }

            if !has_single_node {
                let end = starts_at + mblen;
                let candidate_score = best_here_score + unk_score;
                let target = &mut best_path_ends_at[end];
                if target
                    .as_ref()
                    .is_none_or(|c| candidate_score > c.best_path_score)
                {
                    *target = Some(BestPathNode {
                        id: self.unk_id,
                        best_path_score: candidate_score,
                        starts_at,
                    });
                }
            }
        }

        let mut ends_at = len;
        let mut output = Vec::new();
        while ends_at > 0 {
            let node = best_path_ends_at[ends_at].as_ref().ok_or_else(|| {
                Error::model_parse("failed to find a valid unigram tokenization path")
            })?;
            output.push(Token {
                piece: normalized[node.starts_at..ends_at].to_owned(),
                id: node.id,
            });
            ends_at = node.starts_at;
        }
        output.reverse();
        Ok(output)
    }

    /// Optimized BPE encoding:
    /// - Uses a reusable String for merge key lookups (avoids format!() alloc per pair)
    /// - Marks dead slots instead of Vec::remove() (avoids O(n) shift per merge)
    fn encode_bpe(&self, normalized: &str) -> Vec<Token> {
        if normalized.is_empty() {
            return Vec::new();
        }

        let mut symbols = self.split_chars_or_user_symbols(normalized);
        let mut reverse_merge: HashMap<String, (String, String)> = HashMap::new();

        // Reusable string for building merge keys — avoids format!() allocation per pair.
        let mut merge_key = String::with_capacity(normalized.len());

        loop {
            let mut best: Option<(usize, usize, f32)> = None;
            for left in 0..symbols.len().saturating_sub(1) {
                let sym_l = &symbols[left];
                if sym_l.piece.is_empty() || sym_l.freeze {
                    continue;
                }
                let sym_r = &symbols[left + 1];
                if sym_r.piece.is_empty() || sym_r.freeze {
                    continue;
                }

                // Build merge key in reusable buffer instead of format!().
                merge_key.clear();
                merge_key.push_str(&sym_l.piece);
                merge_key.push_str(&sym_r.piece);

                let Some(id) = self.regular_piece_to_id.get(&merge_key).copied() else {
                    continue;
                };
                let score = self.pieces[id].score;
                let replace = best.is_none_or(|(best_left, _, best_score)| {
                    score > best_score || (score == best_score && left < best_left)
                });
                if replace {
                    best = Some((left, id, score));
                }
            }

            let Some((left, id, _)) = best else {
                break;
            };

            let right = left + 1;

            if self.is_unused(id) {
                // Rare path: save pre-merge pieces for resegment_bpe to split later.
                reverse_merge.insert(
                    merge_key.clone(),
                    (symbols[left].piece.clone(), symbols[right].piece.clone()),
                );
            }

            // Merge: append right piece into left's allocation.
            let right_piece = symbols[right].piece.clone();
            symbols[left].piece.push_str(&right_piece);
            symbols[left].freeze = false;
            symbols[right].piece.clear(); // mark slot as dead
        }

        let mut output = Vec::new();
        for symbol in &symbols {
            if !symbol.piece.is_empty() {
                self.resegment_bpe(&symbol.piece, &reverse_merge, &mut output);
            }
        }
        output
    }

    fn encode_char(&self, normalized: &str) -> Vec<Token> {
        self.split_chars_or_user_symbols(normalized)
            .into_iter()
            .map(|symbol| {
                let id = self.piece_to_id(&symbol.piece);
                Token {
                    piece: symbol.piece,
                    id,
                }
            })
            .collect()
    }

    fn encode_word(&self, normalized: &str) -> Vec<Token> {
        split_into_words(
            normalized,
            self.proto.trainer_spec.treat_whitespace_as_suffix,
            self.proto.trainer_spec.allow_whitespace_only_pieces,
        )
        .into_iter()
        .map(|piece| {
            let id = self.piece_to_id(&piece);
            Token { piece, id }
        })
        .collect()
    }

    fn split_chars_or_user_symbols(&self, normalized: &str) -> Vec<BpeSymbol> {
        let mut output = Vec::new();
        let mut cursor = 0;
        while cursor < normalized.len() {
            if let Some(symbol) = self
                .user_symbols
                .iter()
                .filter(|symbol| normalized[cursor..].starts_with(symbol.as_str()))
                .max_by_key(|symbol| symbol.len())
            {
                output.push(BpeSymbol {
                    piece: symbol.clone(),
                    freeze: true,
                });
                cursor += symbol.len();
            } else {
                let len = char_len_at(normalized, cursor);
                output.push(BpeSymbol {
                    piece: normalized[cursor..cursor + len].to_owned(),
                    freeze: false,
                });
                cursor += len;
            }
        }
        output
    }

    fn resegment_bpe(
        &self,
        piece: &str,
        reverse_merge: &HashMap<String, (String, String)>,
        output: &mut Vec<Token>,
    ) {
        let id = self.piece_to_id(piece);
        if !self.is_unused(id) {
            output.push(Token {
                piece: piece.to_owned(),
                id,
            });
            return;
        }

        if let Some((left, right)) = reverse_merge.get(piece) {
            self.resegment_bpe(left, reverse_merge, output);
            self.resegment_bpe(right, reverse_merge, output);
        } else {
            output.push(Token {
                piece: piece.to_owned(),
                id,
            });
        }
    }

    fn score_for(&self, id: usize, byte_len: usize) -> f32 {
        if self.pieces[id].kind == PieceType::UserDefined {
            0.1 * (byte_len.saturating_sub(1) as f32)
        } else {
            self.pieces[id].score
        }
    }
}

fn validate_piece(piece: &ProtoPiece) -> Result<()> {
    if piece.piece.is_empty() {
        return Err(Error::model_parse("piece must not be empty"));
    }
    if piece.piece.as_bytes().contains(&0) {
        return Err(Error::model_parse("piece must not include a null byte"));
    }
    Ok(())
}

fn split_into_words(
    text: &str,
    treat_whitespace_as_suffix: bool,
    allow_whitespace_only_pieces: bool,
) -> Vec<String> {
    if text.is_empty() {
        return Vec::new();
    }

    let mut ranges = Vec::<(usize, usize)>::new();
    let mut in_ws_sequence = false;
    let mut positions = text
        .char_indices()
        .map(|(index, ch)| (index, index + ch.len_utf8()))
        .collect::<Vec<_>>();
    positions.push((text.len(), text.len()));

    if treat_whitespace_as_suffix {
        ranges.push((0, 0));
        for window in positions.windows(2) {
            let (begin, end) = window[0];
            if begin == end {
                continue;
            }
            let is_ws = &text[begin..end] == SPACE_SYMBOL;
            if is_ws {
                in_ws_sequence = true;
            } else if in_ws_sequence {
                if allow_whitespace_only_pieces {
                    ranges.push((begin, begin));
                }
                in_ws_sequence = false;
            }

            if let Some(last) = ranges.last_mut() {
                last.1 = end;
            }

            let next_begin = window[1].0;
            if next_begin < text.len() && is_ws && !allow_whitespace_only_pieces {
                ranges.push((next_begin, next_begin));
            }
        }
    } else {
        for window in positions.windows(2) {
            let (begin, end) = window[0];
            if begin == end {
                continue;
            }
            let is_ws = &text[begin..end] == SPACE_SYMBOL;
            if begin == 0 || (is_ws && (!in_ws_sequence || !allow_whitespace_only_pieces)) {
                ranges.push((begin, begin));
                in_ws_sequence = true;
            }
            if in_ws_sequence && !is_ws {
                in_ws_sequence = false;
            }
            if let Some(last) = ranges.last_mut() {
                last.1 = end;
            }
        }
    }

    ranges
        .into_iter()
        .filter(|(begin, end)| begin < end)
        .map(|(begin, end)| text[begin..end].to_owned())
        .collect()
}

pub(crate) fn byte_pieces_for_unknown(
    model: &SentencePieceModel,
    piece: &str,
) -> Result<Vec<Token>> {
    let mut output = Vec::with_capacity(piece.len());
    for byte in piece.as_bytes().iter().copied() {
        let byte_piece = byte_to_piece(byte);
        let id = model.byte_id(byte).ok_or_else(|| {
            Error::model_parse(format!("byte fallback piece {byte_piece} is missing"))
        })?;
        output.push(Token {
            piece: byte_piece,
            id,
        });
    }
    Ok(output)
}

pub(crate) fn merge_or_byte_fallback_tokens(
    model: &SentencePieceModel,
    tokens: Vec<Token>,
) -> Result<Vec<Token>> {
    let mut output: Vec<Token> = Vec::new();
    for token in tokens {
        let is_unk = model.is_unknown(token.id);
        if is_unk && model.byte_fallback_enabled() {
            output.extend(byte_pieces_for_unknown(model, &token.piece)?);
            continue;
        }

        if is_unk && output.last().is_some_and(|prev| model.is_unknown(prev.id)) {
            if let Some(prev) = output.last_mut() {
                prev.piece.push_str(&token.piece);
            }
        } else {
            output.push(token);
        }
    }
    Ok(output)
}

pub(crate) fn validate_no_duplicate_user_symbols(symbols: &[String]) -> Result<()> {
    let mut seen = HashSet::with_capacity(symbols.len());
    for symbol in symbols {
        if !seen.insert(symbol) {
            return Err(Error::model_parse(format!(
                "user-defined symbol {symbol:?} is duplicated"
            )));
        }
    }
    Ok(())
}
