//! Locating `librkllmrt.so` on disk.

use std::collections::HashSet;
use std::env;
use std::ffi::OsString;
use std::path::{Path, PathBuf};

use crate::LIBRARY_NAME;

/// Directories searched after the environment and the executable, in order.
const SYSTEM_DIRS: &[&str] = &[
    "/usr/lib",
    "/usr/local/lib",
    "/usr/lib64",
    "/lib",
    "/opt/rkllm/lib",
];

/// Every copy of the RKLLM runtime this machine appears to have, best first.
///
/// Only paths that exist are yielded, and each underlying file is yielded once
/// even when several candidates resolve to it. The search order is most
/// specific to least:
///
/// 1. `RKLLM_LIB`, taken as the full path to the shared object.
/// 2. `RKLLM_LIB_DIR`, the directory holding it. The build script reads the
///    same variable for the `link` feature.
/// 3. Each entry of `LD_LIBRARY_PATH`.
/// 4. Beside the running executable, then `lib/` under it and beside it.
/// 5. The working directory.
/// 6. The usual system library directories, including the multiarch one for
///    this target.
///
/// Nothing here consults the dynamic loader's cache, so a library installed
/// somewhere unusual and registered with `ldconfig` will not turn up. Passing
/// [`LIBRARY_NAME`] straight to the loader still finds that one.
///
/// ```no_run
/// # use rkllm_sys::{find_library_path, LIBRARY_NAME};
/// let library = find_library_path()
///     .next()
///     .unwrap_or_else(|| LIBRARY_NAME.into());
/// ```
pub fn find_library_path() -> impl Iterator<Item = PathBuf> {
    existing(candidates(|key| env::var_os(key)))
}

/// Builds the candidate list, reading the environment through `var`.
fn candidates(var: impl Fn(&str) -> Option<OsString>) -> Vec<PathBuf> {
    let mut out = Vec::new();

    if let Some(path) = var("RKLLM_LIB") {
        out.push(PathBuf::from(path));
    }
    if let Some(dir) = var("RKLLM_LIB_DIR") {
        out.push(Path::new(&dir).join(LIBRARY_NAME));
    }
    if let Some(paths) = var("LD_LIBRARY_PATH") {
        out.extend(env::split_paths(&paths).map(|dir| dir.join(LIBRARY_NAME)));
    }

    if let Ok(exe) = env::current_exe()
        && let Some(dir) = exe.parent()
    {
        out.push(dir.join(LIBRARY_NAME));
        out.push(dir.join("lib").join(LIBRARY_NAME));
        if let Some(parent) = dir.parent() {
            out.push(parent.join("lib").join(LIBRARY_NAME));
        }
    }

    out.push(PathBuf::from(LIBRARY_NAME));

    for dir in SYSTEM_DIRS {
        out.push(Path::new(dir).join(LIBRARY_NAME));
    }
    for triple in multiarch_triples() {
        out.push(Path::new("/usr/lib").join(triple).join(LIBRARY_NAME));
    }

    out
}

/// The multiarch directory names plausible for the target being built for.
fn multiarch_triples() -> Vec<String> {
    let arch = env::consts::ARCH;
    let mut triples = vec![format!("{arch}-linux-gnu")];
    if arch == "arm" {
        triples.push("arm-linux-gnueabihf".to_owned());
    }
    triples
}

/// Keeps the candidates that exist, without repeating a file reached two ways.
fn existing(candidates: Vec<PathBuf>) -> impl Iterator<Item = PathBuf> {
    let mut seen: HashSet<PathBuf> = HashSet::new();
    candidates.into_iter().filter(move |path| {
        if !path.is_file() {
            return false;
        }
        // Fall back to the path itself when it cannot be canonicalized, so an
        // unreadable parent directory hides nothing.
        let identity = path.canonicalize().unwrap_or_else(|_| path.clone());
        seen.insert(identity)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> + use<> {
        let owned: Vec<(String, OsString)> = pairs
            .iter()
            .map(|(k, v)| ((*k).to_owned(), OsString::from(*v)))
            .collect();
        move |key| owned.iter().find(|(k, _)| k == key).map(|(_, v)| v.clone())
    }

    #[test]
    fn the_explicit_variables_come_first() {
        let found = candidates(fake_env(&[
            ("RKLLM_LIB", "/custom/librkllmrt.so"),
            ("RKLLM_LIB_DIR", "/custom/dir"),
        ]));
        assert_eq!(found[0], Path::new("/custom/librkllmrt.so"));
        assert_eq!(found[1], Path::new("/custom/dir").join(LIBRARY_NAME));
    }

    #[test]
    fn every_ld_library_path_entry_is_searched() {
        let found = candidates(fake_env(&[("LD_LIBRARY_PATH", "/one:/two")]));
        assert!(found.contains(&Path::new("/one").join(LIBRARY_NAME)));
        assert!(found.contains(&Path::new("/two").join(LIBRARY_NAME)));
    }

    #[test]
    fn the_system_directories_are_always_searched() {
        let found = candidates(fake_env(&[]));
        assert!(found.contains(&Path::new("/usr/lib").join(LIBRARY_NAME)));
        assert!(found.contains(&Path::new("/usr/local/lib").join(LIBRARY_NAME)));
        let multiarch = Path::new("/usr/lib")
            .join(format!("{}-linux-gnu", env::consts::ARCH))
            .join(LIBRARY_NAME);
        assert!(found.contains(&multiarch));
    }

    #[test]
    fn missing_paths_are_left_out() {
        let found: Vec<_> = existing(vec![
            PathBuf::from("/definitely/not/here/librkllmrt.so"),
            PathBuf::from("/also/not/here"),
        ])
        .collect();
        assert!(found.is_empty());
    }

    #[test]
    fn a_file_reached_two_ways_is_yielded_once() {
        let dir = env::temp_dir().join(format!("rkllm-find-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("temp dir");

        let real = dir.join(LIBRARY_NAME);
        std::fs::write(&real, b"not really a shared object").expect("write");
        let link = dir.join("alias.so");
        std::os::unix::fs::symlink(&real, &link).expect("symlink");

        // The same file by two names, plus a repeat of the first.
        let found: Vec<_> = existing(vec![real.clone(), link, real.clone()]).collect();

        let _ = std::fs::remove_dir_all(&dir);
        assert_eq!(found, vec![real], "the symlink resolves to the same file");
    }

    #[test]
    fn directories_are_not_mistaken_for_the_library() {
        let found: Vec<_> = existing(vec![PathBuf::from("/usr/lib")]).collect();
        assert!(found.is_empty(), "a directory is not the shared object");
    }
}
