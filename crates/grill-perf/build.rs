use std::env;
use std::path::Path;
use std::process::Command;

// Everything compiled into grill-perf, relative to this package directory; src embeds tools/ files.
const INPUTS: [&str; 5] = [
    "../../Cargo.toml",
    "../../Cargo.lock",
    ".",
    "../grill-sse",
    "../../tools",
];

fn main() {
    println!("cargo:rerun-if-env-changed=GRILL_PERF_SOURCE_COMMIT");
    for input in INPUTS {
        println!("cargo:rerun-if-changed={input}");
    }
    let head = head();
    let source = match env::var_os("GRILL_PERF_SOURCE_COMMIT") {
        Some(commit) => {
            let commit = commit
                .into_string()
                .ok()
                .filter(|commit| is_commit(commit))
                .unwrap_or_else(|| {
                    panic!(
                        "GRILL_PERF_SOURCE_COMMIT must be a full 40-character lowercase hex commit"
                    )
                });
            if let Ok((head, clean)) = head {
                assert!(
                    head == commit && clean,
                    "GRILL_PERF_SOURCE_COMMIT {commit} must equal HEAD {head} with clean build inputs"
                );
            }
            commit
        }
        None => match head {
            Ok((head, true)) => head,
            Ok((_, false)) => unrecorded("build inputs differ from HEAD"),
            Err(reason) => unrecorded(&reason),
        },
    };
    println!("cargo:rustc-env=GRILL_PERF_SOURCE={source}");
    println!(
        "cargo:rustc-env=GRILL_PERF_TARGET={}",
        env::var("TARGET").expect("cargo sets TARGET for build scripts")
    );
}

fn unrecorded(reason: &str) -> String {
    println!("cargo:warning=grill-perf source unrecorded: {reason}");
    "unrecorded".to_owned()
}

fn is_commit(commit: &str) -> bool {
    commit.len() == 40
        && commit
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// HEAD and whether the build inputs match it, when this workspace is the repository root.
fn head() -> Result<(String, bool), String> {
    let resolved = git(&[
        "rev-parse",
        "--show-toplevel",
        "--git-path",
        "HEAD",
        "--git-path",
        "logs/HEAD",
        "--git-path",
        "packed-refs",
        "--git-path",
        "index",
        "HEAD",
        "--symbolic-full-name",
        "HEAD",
    ])?;
    let [top, head_file, reflog, packed, index, head, reference] =
        resolved.lines().collect::<Vec<_>>()[..]
    else {
        return Err(format!("unexpected git rev-parse output {resolved:?}"));
    };
    // An enclosing checkout (such as one holding an extracted archive) is not this source's history.
    if Path::new(top).canonicalize().ok() != Path::new("../..").canonicalize().ok() {
        return Err(format!("workspace is not the git repository root {top}"));
    }
    let reference = git(&["rev-parse", "--git-path", reference])?;
    // A missing path would rerun every build; the reflog also catches a packed ref's first loose write.
    for path in [head_file, reflog, reference.trim_end(), packed, index] {
        if Path::new(path).exists() {
            println!("cargo:rerun-if-changed={path}");
        }
    }
    if !is_commit(head) {
        return Err(format!("HEAD {head} is not a 40-character SHA-1 commit"));
    }
    let mut status = vec![
        "--no-optional-locks",
        "status",
        "--porcelain",
        "--untracked-files=all",
        "--",
    ];
    status.extend(INPUTS);
    // Skip-worktree and assume-unchanged entries are invisible to status, so require plain tracking.
    let mut tracked = vec!["--no-optional-locks", "ls-files", "-v", "--"];
    tracked.extend(INPUTS);
    let clean =
        git(&status)?.is_empty() && git(&tracked)?.lines().all(|line| line.starts_with("H "));
    Ok((head.to_owned(), clean))
}

fn git(args: &[&str]) -> Result<String, String> {
    let output = Command::new("git")
        .args(args)
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !output.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args.join(" "),
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout).map_err(|e| e.to_string())
}
