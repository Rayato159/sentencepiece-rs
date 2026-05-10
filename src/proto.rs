use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

impl PieceType {
    pub(crate) fn from_i32(value: i32) -> Self {
        match value {
            2 => Self::Unknown,
            3 => Self::Control,
            4 => Self::UserDefined,
            5 => Self::Unused,
            6 => Self::Byte,
            _ => Self::Normal,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ModelType {
    Unigram,
    Bpe,
    Word,
    Char,
}

impl ModelType {
    pub(crate) fn from_i32(value: i32) -> Self {
        match value {
            2 => Self::Bpe,
            3 => Self::Word,
            4 => Self::Char,
            _ => Self::Unigram,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub(crate) struct ModelProto {
    pub(crate) pieces: Vec<SentencePiece>,
    pub(crate) trainer_spec: TrainerSpec,
    pub(crate) normalizer_spec: NormalizerSpec,
    pub(crate) denormalizer_spec: Option<NormalizerSpec>,
    pub(crate) self_test_data: SelfTestData,
}

#[derive(Clone, Debug)]
pub(crate) struct SentencePiece {
    pub(crate) piece: String,
    pub(crate) score: f32,
    pub(crate) kind: PieceType,
}

#[derive(Clone, Debug)]
pub(crate) struct TrainerSpec {
    pub(crate) model_type: ModelType,
    pub(crate) treat_whitespace_as_suffix: bool,
    pub(crate) allow_whitespace_only_pieces: bool,
    pub(crate) byte_fallback: bool,
    pub(crate) unk_id: i32,
    pub(crate) bos_id: i32,
    pub(crate) eos_id: i32,
    pub(crate) pad_id: i32,
    pub(crate) unk_surface: String,
    pub(crate) unk_piece: String,
    pub(crate) bos_piece: String,
    pub(crate) eos_piece: String,
    pub(crate) pad_piece: String,
}

#[derive(Clone, Debug)]
pub(crate) struct NormalizerSpec {
    pub(crate) name: String,
    pub(crate) precompiled_charsmap: Vec<u8>,
    pub(crate) add_dummy_prefix: bool,
    pub(crate) remove_extra_whitespaces: bool,
    pub(crate) escape_whitespaces: bool,
}

#[derive(Clone, Debug, Default)]
pub(crate) struct SelfTestData {
    pub(crate) samples: Vec<SelfTestSample>,
}

#[derive(Clone, Debug)]
pub(crate) struct SelfTestSample {
    pub(crate) input: String,
    pub(crate) expected: String,
}

impl Default for TrainerSpec {
    fn default() -> Self {
        Self {
            model_type: ModelType::Unigram,
            treat_whitespace_as_suffix: false,
            allow_whitespace_only_pieces: false,
            byte_fallback: false,
            unk_id: 0,
            bos_id: 1,
            eos_id: 2,
            pad_id: -1,
            unk_surface: crate::util::DEFAULT_UNKNOWN_SURFACE.to_owned(),
            unk_piece: "<unk>".to_owned(),
            bos_piece: "<s>".to_owned(),
            eos_piece: "</s>".to_owned(),
            pad_piece: "<pad>".to_owned(),
        }
    }
}

impl Default for NormalizerSpec {
    fn default() -> Self {
        Self {
            name: String::new(),
            precompiled_charsmap: Vec::new(),
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

impl ModelProto {
    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        let mut proto = Self::default();
        let mut reader = ProtoReader::new(bytes);
        while let Some((field, wire)) = reader.read_key()? {
            match field {
                1 if wire == WireType::LengthDelimited => {
                    let bytes = reader.read_len()?;
                    proto.pieces.push(decode_sentence_piece(bytes)?);
                }
                2 if wire == WireType::LengthDelimited => {
                    proto.trainer_spec = decode_trainer_spec(reader.read_len()?)?;
                }
                3 if wire == WireType::LengthDelimited => {
                    proto.normalizer_spec = decode_normalizer_spec(reader.read_len()?)?;
                }
                4 if wire == WireType::LengthDelimited => {
                    proto.self_test_data = decode_self_test_data(reader.read_len()?)?;
                }
                5 if wire == WireType::LengthDelimited => {
                    proto.denormalizer_spec = Some(decode_normalizer_spec(reader.read_len()?)?);
                }
                _ => reader.skip(wire)?,
            }
        }
        Ok(proto)
    }
}

fn decode_sentence_piece(bytes: &[u8]) -> Result<SentencePiece> {
    let mut piece = SentencePiece {
        piece: String::new(),
        score: 0.0,
        kind: PieceType::Normal,
    };
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, wire)) = reader.read_key()? {
        match field {
            1 if wire == WireType::LengthDelimited => {
                piece.piece = reader.read_string()?;
            }
            2 if wire == WireType::ThirtyTwoBit => {
                piece.score = reader.read_f32()?;
            }
            3 if wire == WireType::Varint => {
                piece.kind = PieceType::from_i32(reader.read_varint()? as i32);
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(piece)
}

fn decode_trainer_spec(bytes: &[u8]) -> Result<TrainerSpec> {
    let mut spec = TrainerSpec::default();
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, wire)) = reader.read_key()? {
        match field {
            3 if wire == WireType::Varint => {
                spec.model_type = ModelType::from_i32(reader.read_varint()? as i32);
            }
            24 if wire == WireType::Varint => {
                spec.treat_whitespace_as_suffix = reader.read_bool()?;
            }
            26 if wire == WireType::Varint => {
                spec.allow_whitespace_only_pieces = reader.read_bool()?;
            }
            35 if wire == WireType::Varint => {
                spec.byte_fallback = reader.read_bool()?;
            }
            40 if wire == WireType::Varint => {
                spec.unk_id = reader.read_varint()? as i32;
            }
            41 if wire == WireType::Varint => {
                spec.bos_id = reader.read_varint()? as i32;
            }
            42 if wire == WireType::Varint => {
                spec.eos_id = reader.read_varint()? as i32;
            }
            43 if wire == WireType::Varint => {
                spec.pad_id = reader.read_varint()? as i32;
            }
            44 if wire == WireType::LengthDelimited => {
                spec.unk_surface = reader.read_string()?;
            }
            45 if wire == WireType::LengthDelimited => {
                spec.unk_piece = reader.read_string()?;
            }
            46 if wire == WireType::LengthDelimited => {
                spec.bos_piece = reader.read_string()?;
            }
            47 if wire == WireType::LengthDelimited => {
                spec.eos_piece = reader.read_string()?;
            }
            48 if wire == WireType::LengthDelimited => {
                spec.pad_piece = reader.read_string()?;
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(spec)
}

fn decode_normalizer_spec(bytes: &[u8]) -> Result<NormalizerSpec> {
    let mut spec = NormalizerSpec::default();
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, wire)) = reader.read_key()? {
        match field {
            1 if wire == WireType::LengthDelimited => {
                spec.name = reader.read_string()?;
            }
            2 if wire == WireType::LengthDelimited => {
                spec.precompiled_charsmap = reader.read_len()?.to_vec();
            }
            3 if wire == WireType::Varint => {
                spec.add_dummy_prefix = reader.read_bool()?;
            }
            4 if wire == WireType::Varint => {
                spec.remove_extra_whitespaces = reader.read_bool()?;
            }
            5 if wire == WireType::Varint => {
                spec.escape_whitespaces = reader.read_bool()?;
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(spec)
}

fn decode_self_test_data(bytes: &[u8]) -> Result<SelfTestData> {
    let mut data = SelfTestData::default();
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, wire)) = reader.read_key()? {
        match field {
            1 if wire == WireType::LengthDelimited => {
                data.samples
                    .push(decode_self_test_sample(reader.read_len()?)?);
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(data)
}

fn decode_self_test_sample(bytes: &[u8]) -> Result<SelfTestSample> {
    let mut sample = SelfTestSample {
        input: String::new(),
        expected: String::new(),
    };
    let mut reader = ProtoReader::new(bytes);
    while let Some((field, wire)) = reader.read_key()? {
        match field {
            1 if wire == WireType::LengthDelimited => {
                sample.input = reader.read_string()?;
            }
            2 if wire == WireType::LengthDelimited => {
                sample.expected = reader.read_string()?;
            }
            _ => reader.skip(wire)?,
        }
    }
    Ok(sample)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WireType {
    Varint,
    SixtyFourBit,
    LengthDelimited,
    ThirtyTwoBit,
}

impl WireType {
    fn from_key(value: u64) -> Result<Self> {
        match value & 0b111 {
            0 => Ok(Self::Varint),
            1 => Ok(Self::SixtyFourBit),
            2 => Ok(Self::LengthDelimited),
            5 => Ok(Self::ThirtyTwoBit),
            wire => Err(Error::model_parse(format!(
                "unsupported protobuf wire type {wire}"
            ))),
        }
    }
}

struct ProtoReader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> ProtoReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    fn read_key(&mut self) -> Result<Option<(u32, WireType)>> {
        if self.position == self.bytes.len() {
            return Ok(None);
        }

        let key = self.read_varint()?;
        let field = (key >> 3) as u32;
        if field == 0 {
            return Err(Error::model_parse("protobuf field number 0 is invalid"));
        }
        Ok(Some((field, WireType::from_key(key)?)))
    }

    fn read_varint(&mut self) -> Result<u64> {
        let mut value = 0u64;
        for shift in (0..64).step_by(7) {
            let byte = *self
                .bytes
                .get(self.position)
                .ok_or_else(|| Error::model_parse("unexpected end of protobuf varint"))?;
            self.position += 1;
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(Error::model_parse("protobuf varint is too long"))
    }

    fn read_bool(&mut self) -> Result<bool> {
        Ok(self.read_varint()? != 0)
    }

    fn read_len(&mut self) -> Result<&'a [u8]> {
        let len = self.read_varint()? as usize;
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| Error::model_parse("protobuf length overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::model_parse(
                "unexpected end of protobuf length field",
            ));
        }
        let out = &self.bytes[self.position..end];
        self.position = end;
        Ok(out)
    }

    fn read_string(&mut self) -> Result<String> {
        let bytes = self.read_len()?;
        String::from_utf8(bytes.to_vec())
            .map_err(|_| Error::model_parse("protobuf string is not valid UTF-8"))
    }

    fn read_f32(&mut self) -> Result<f32> {
        let end = self
            .position
            .checked_add(4)
            .ok_or_else(|| Error::model_parse("protobuf fixed32 overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::model_parse("unexpected end of protobuf fixed32"));
        }
        let bytes = [
            self.bytes[self.position],
            self.bytes[self.position + 1],
            self.bytes[self.position + 2],
            self.bytes[self.position + 3],
        ];
        self.position = end;
        Ok(f32::from_le_bytes(bytes))
    }

    fn skip(&mut self, wire: WireType) -> Result<()> {
        match wire {
            WireType::Varint => {
                self.read_varint()?;
            }
            WireType::SixtyFourBit => {
                self.skip_bytes(8)?;
            }
            WireType::LengthDelimited => {
                let len = self.read_varint()? as usize;
                self.skip_bytes(len)?;
            }
            WireType::ThirtyTwoBit => {
                self.skip_bytes(4)?;
            }
        }
        Ok(())
    }

    fn skip_bytes(&mut self, len: usize) -> Result<()> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| Error::model_parse("protobuf skip overflow"))?;
        if end > self.bytes.len() {
            return Err(Error::model_parse(
                "unexpected end while skipping protobuf field",
            ));
        }
        self.position = end;
        Ok(())
    }
}
