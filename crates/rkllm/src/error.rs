//! Errors returned by this crate.

use std::error::Error as StdError;
use std::ffi::NulError;
use std::fmt;
use std::os::raw::c_int;

/// A `Result` with this crate's [`Error`].
pub type Result<T, E = Error> = std::result::Result<T, E>;

/// Something the RKLLM runtime, or the conversion into it, refused.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// A `librkllmrt` function returned a non-zero status.
    ///
    /// The runtime does not document distinct codes, so the raw value is
    /// carried through alongside the name of the call that produced it.
    Status {
        /// Name of the C function that failed, such as `rkllm_run`.
        call: &'static str,
        /// The status it returned.
        code: c_int,
    },
    /// `rkllm_init` reported success but left the handle null.
    NullHandle,
    /// A string destined for C contained an interior NUL byte.
    InteriorNul {
        /// Which argument the string was for.
        field: &'static str,
        /// The offset of the offending byte.
        position: usize,
    },
    /// Loading `librkllmrt.so` failed.
    #[cfg(feature = "libloading")]
    Library(rkllm_sys::libloading::Error),
    /// The blocking task running the inference panicked or was cancelled.
    #[cfg(feature = "tokio")]
    Task(tokio::task::JoinError),
}

impl Error {
    pub(crate) fn status(call: &'static str, code: c_int) -> Self {
        Error::Status { call, code }
    }

    /// Returns `Ok(())` if `code` is zero, and a [`Error::Status`] otherwise.
    pub(crate) fn check(call: &'static str, code: c_int) -> Result<()> {
        if code == 0 {
            Ok(())
        } else {
            Err(Error::status(call, code))
        }
    }

    pub(crate) fn nul(field: &'static str, err: &NulError) -> Self {
        Error::InteriorNul {
            field,
            position: err.nul_position(),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Status { call, code } => write!(f, "{call} returned status {code}"),
            Error::NullHandle => f.write_str("rkllm_init succeeded but produced a null handle"),
            Error::InteriorNul { field, position } => {
                write!(f, "{field} contains a NUL byte at offset {position}")
            }
            #[cfg(feature = "libloading")]
            Error::Library(e) => write!(f, "could not load librkllmrt: {e}"),
            #[cfg(feature = "tokio")]
            Error::Task(e) => write!(f, "the inference task did not finish: {e}"),
        }
    }
}

impl StdError for Error {
    fn source(&self) -> Option<&(dyn StdError + 'static)> {
        match self {
            #[cfg(feature = "libloading")]
            Error::Library(e) => Some(e),
            #[cfg(feature = "tokio")]
            Error::Task(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "libloading")]
impl From<rkllm_sys::libloading::Error> for Error {
    fn from(e: rkllm_sys::libloading::Error) -> Self {
        Error::Library(e)
    }
}

#[cfg(feature = "tokio")]
impl From<tokio::task::JoinError> for Error {
    fn from(e: tokio::task::JoinError) -> Self {
        Error::Task(e)
    }
}
