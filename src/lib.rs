//! SentencePiece runtime in Rust.
//!
//! This crate loads existing SentencePiece `.model` / `.spm` files and exposes
//! a small processor API for normalization, encoding, and decoding.

mod darts;
mod error;
mod model;
mod normalizer;
mod processor;
mod proto;
mod util;

pub use crate::error::{Error, Result};
pub use crate::model::{ModelType, Piece, PieceType, SentencePieceModel};
pub use crate::normalizer::Normalizer;
pub use crate::processor::SentencePieceProcessor;
pub use crate::util::{DEFAULT_UNKNOWN_SURFACE, REPLACEMENT_CHARACTER, SPACE_SYMBOL};
