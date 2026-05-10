/// SentencePiece's visible whitespace marker, U+2581 LOWER ONE EIGHT BLOCK.
pub const SPACE_SYMBOL: &str = "\u{2581}";

/// Default decoded surface for the `<unk>` piece.
pub const DEFAULT_UNKNOWN_SURFACE: &str = " \u{2047} ";

/// Unicode replacement character, U+FFFD.
pub const REPLACEMENT_CHARACTER: &str = "\u{FFFD}";

pub(crate) fn byte_to_piece(byte: u8) -> String {
    format!("<0x{byte:02X}>")
}

pub(crate) fn piece_to_byte(piece: &str) -> Option<u8> {
    let hex = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if hex.len() != 2 {
        return None;
    }
    u8::from_str_radix(hex, 16).ok()
}

pub(crate) fn first_char_len(input: &str) -> usize {
    input.chars().next().map(char::len_utf8).unwrap_or(0)
}

pub(crate) fn char_len_at(input: &str, byte_index: usize) -> usize {
    input[byte_index..]
        .chars()
        .next()
        .map(char::len_utf8)
        .unwrap_or(0)
}

pub(crate) fn replace_space_symbol(input: &str) -> String {
    input.replace(SPACE_SYMBOL, " ")
}
