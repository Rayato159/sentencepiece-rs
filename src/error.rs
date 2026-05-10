use std::fmt;

/// Crate-wide result type.
pub type Result<T> = std::result::Result<T, Error>;

/// Errors returned by model loading, normalization, encoding, and decoding.
#[derive(Debug)]
pub enum Error {
    /// File I/O failed.
    Io(std::io::Error),
    /// A serialized SentencePiece model could not be parsed or validated.
    ModelParse(String),
    /// The caller provided invalid input.
    InvalidInput(String),
    /// The model uses a feature this runtime does not support yet.
    Unsupported(String),
}

impl Error {
    pub(crate) fn model_parse(message: impl Into<String>) -> Self {
        Self::ModelParse(message.into())
    }

    pub(crate) fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => write!(f, "I/O error: {error}"),
            Self::ModelParse(message) => write!(f, "model parse error: {message}"),
            Self::InvalidInput(message) => write!(f, "invalid input: {message}"),
            Self::Unsupported(message) => write!(f, "unsupported feature: {message}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::ModelParse(_) | Self::InvalidInput(_) | Self::Unsupported(_) => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}
