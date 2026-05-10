use sentencepiece_rs::{ModelType, SPACE_SYMBOL, SentencePieceProcessor};

const NORMAL: i32 = 1;
const UNKNOWN: i32 = 2;
const CONTROL: i32 = 3;
const USER_DEFINED: i32 = 4;
const BYTE: i32 = 6;

#[test]
fn loads_model_metadata() {
    let processor = test_processor();

    assert_eq!(processor.model().vocab_size(), 9);
    assert_eq!(processor.model().model_type(), ModelType::Unigram);
    assert_eq!(processor.unk_id(), 0);
    assert_eq!(processor.bos_id(), Some(1));
    assert_eq!(processor.eos_id(), Some(2));
    assert_eq!(processor.piece_to_id("missing"), 0);
    assert_eq!(processor.id_to_piece(4).unwrap(), "▁hello");
}

#[test]
fn normalizes_whitespace_without_charsmap() {
    let processor = test_processor();

    assert_eq!(
        processor.normalize("  hello   world  ").unwrap(),
        "▁hello▁world"
    );
    assert_eq!(processor.normalize("").unwrap(), "");
    assert_eq!(processor.normalize("     ").unwrap(), "");
}

#[test]
fn encodes_and_decodes_unigram() {
    let processor = test_processor();

    let pieces = processor.encode("hello world").unwrap();
    assert_eq!(pieces, ["▁hello", "▁world"]);

    let ids = processor.encode_to_ids("hello world").unwrap();
    assert_eq!(ids, [4, 5]);

    assert_eq!(processor.decode(&pieces).unwrap(), "hello world");
    assert_eq!(processor.decode_ids(&ids).unwrap(), "hello world");
}

#[test]
fn handles_unknown_pieces_and_surfaces() {
    let processor = test_processor();

    let pieces = processor.encode("hello z").unwrap();
    assert_eq!(pieces, ["▁hello", SPACE_SYMBOL, "z"]);
    assert_eq!(processor.encode_to_ids("hello z").unwrap(), [4, 3, 0]);
    assert_eq!(processor.decode(&pieces).unwrap(), "hello z");

    let explicit_unk = ["<unk>"];
    assert_eq!(processor.decode(&explicit_unk).unwrap(), " ⁇ ");
}

#[test]
fn handles_unicode_text() {
    let processor = test_processor();

    let pieces = processor.encode("こんにちは 世界").unwrap();
    assert_eq!(pieces, ["▁こんにちは", "▁世界"]);
    assert_eq!(processor.decode(&pieces).unwrap(), "こんにちは 世界");
}

#[test]
fn keeps_user_defined_symbols_intact() {
    let processor = test_processor();

    let pieces = processor.encode("hello <USER>").unwrap();
    assert_eq!(pieces, ["▁hello", SPACE_SYMBOL, "<USER>"]);
    assert_eq!(processor.decode(&pieces).unwrap(), "hello <USER>");
}

#[test]
fn byte_fallback_round_trips_unicode_unknowns() {
    let processor = byte_fallback_processor();

    let pieces = processor.encode("hi 🚀").unwrap();
    assert_eq!(
        pieces,
        ["▁hi", SPACE_SYMBOL, "<0xF0>", "<0x9F>", "<0x9A>", "<0x80>"]
    );
    assert_eq!(processor.decode(&pieces).unwrap(), "hi 🚀");
}

#[test]
fn bpe_model_merges_best_pairs() {
    let processor = bpe_processor();

    let pieces = processor.encode("abc").unwrap();
    assert_eq!(pieces, [SPACE_SYMBOL, "ab", "c"]);
    assert_eq!(processor.decode(&pieces).unwrap(), "abc");
}

#[test]
fn extra_options_work_in_order() {
    let mut processor = test_processor();
    processor.set_encode_extra_options("bos:eos").unwrap();

    assert_eq!(
        processor.encode("hello world").unwrap(),
        ["<s>", "▁hello", "▁world", "</s>"]
    );

    processor.set_encode_extra_options("reverse").unwrap();
    assert_eq!(
        processor.encode("hello world").unwrap(),
        ["▁world", "▁hello"]
    );
}

fn test_processor() -> SentencePieceProcessor {
    let pieces = vec![
        piece("<unk>", 0.0, UNKNOWN),
        piece("<s>", 0.0, CONTROL),
        piece("</s>", 0.0, CONTROL),
        piece(SPACE_SYMBOL, -5.0, NORMAL),
        piece("▁hello", 0.0, NORMAL),
        piece("▁world", 0.0, NORMAL),
        piece("▁こんにちは", 0.0, NORMAL),
        piece("▁世界", 0.0, NORMAL),
        piece("<USER>", 0.0, USER_DEFINED),
    ];
    SentencePieceProcessor::from_serialized_model(&model(pieces, Vec::new())).unwrap()
}

fn byte_fallback_processor() -> SentencePieceProcessor {
    let mut pieces = vec![
        piece("<unk>", 0.0, UNKNOWN),
        piece("<s>", 0.0, CONTROL),
        piece("</s>", 0.0, CONTROL),
        piece(SPACE_SYMBOL, -5.0, NORMAL),
        piece("▁hi", 0.0, NORMAL),
    ];
    for byte in 0u8..=255 {
        pieces.push(piece(&format!("<0x{byte:02X}>"), -20.0, BYTE));
    }

    let mut trainer = Vec::new();
    bool_field(&mut trainer, 35, true);
    SentencePieceProcessor::from_serialized_model(&model(pieces, trainer)).unwrap()
}

fn bpe_processor() -> SentencePieceProcessor {
    let pieces = vec![
        piece("<unk>", 0.0, UNKNOWN),
        piece("<s>", 0.0, CONTROL),
        piece("</s>", 0.0, CONTROL),
        piece(SPACE_SYMBOL, 0.0, NORMAL),
        piece("a", 0.0, NORMAL),
        piece("b", 0.0, NORMAL),
        piece("c", 0.0, NORMAL),
        piece("ab", 10.0, NORMAL),
    ];
    let mut trainer = Vec::new();
    varint_field(&mut trainer, 3, 2);
    SentencePieceProcessor::from_serialized_model(&model(pieces, trainer)).unwrap()
}

fn model(pieces: Vec<Vec<u8>>, trainer: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::new();
    for piece in pieces {
        message_field(&mut out, 1, &piece);
    }
    if !trainer.is_empty() {
        message_field(&mut out, 2, &trainer);
    }
    out
}

fn piece(text: &str, score: f32, kind: i32) -> Vec<u8> {
    let mut out = Vec::new();
    string_field(&mut out, 1, text);
    fixed32_field(&mut out, 2, score.to_bits());
    varint_field(&mut out, 3, kind as u64);
    out
}

fn message_field(out: &mut Vec<u8>, field: u32, bytes: &[u8]) {
    key(out, field, 2);
    varint(out, bytes.len() as u64);
    out.extend_from_slice(bytes);
}

fn string_field(out: &mut Vec<u8>, field: u32, value: &str) {
    message_field(out, field, value.as_bytes());
}

fn fixed32_field(out: &mut Vec<u8>, field: u32, value: u32) {
    key(out, field, 5);
    out.extend_from_slice(&value.to_le_bytes());
}

fn bool_field(out: &mut Vec<u8>, field: u32, value: bool) {
    varint_field(out, field, u64::from(value));
}

fn varint_field(out: &mut Vec<u8>, field: u32, value: u64) {
    key(out, field, 0);
    varint(out, value);
}

fn key(out: &mut Vec<u8>, field: u32, wire: u8) {
    varint(out, ((field as u64) << 3) | u64::from(wire));
}

fn varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}
