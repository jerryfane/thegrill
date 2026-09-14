# Performance release notes

## Unreleased — extended serving evidence

- Add explicit policy2 first-generated, first-answer and client-completion
  latency gates on homogeneous flat workloads, plus complete-wave worst-lane
  first-answer and max/min ratio gates. Correlated peers are observations,
  not extra independent trials.
- Share checked rational envelope arithmetic without changing policy1
  serialization, tolerance boundaries, reason precedence or exit meanings.
- Preserve all A/B/A2, warmup, output amount and cache eligibility requirements.
  Actual-hit latency never fabricates cold-prefill throughput.
- Add finite workload4 solo/mixed schedules, fixed offsets and first-generated
  triggers, named controls, actual overlap evidence and settle-before-publication
  cancellation. Preserve never-dispatched and failed peers.
- Add workload5 streamed-tool fragments and first-delta/fully-validated-call
  timing with exact retained history and replay. Tool fragments are not text.
- Add explicit bounded metrics2 provider accounting and prospective required
  telemetry gates. Missing sources, gaps and resets are not zero; shared counters
  are not attributed to one request, and an isolation declaration is not proof.
  Source/position selectors cannot bypass contradictory engine accounting;
  unrelated engine identities remain independent.
- Add small GLM/DeepSeek routine and mixed profiles from the existing prompts,
  with explicit controls and traffic ceilings. Native CPU captures and replay
  were exercised; no real-backend qualification follows from fixture success.
- Named mixed-schedule policy gates, repeated conversations, larger histories,
  tails and resource/startup/kernel domains remain separate integration gates.
  Independent review and live qualification are still required. Released v0.1.0
  artifacts and frozen workload examples are unchanged.

## Version 0.1.0 — published prerelease

[v0.1.0 is published](https://github.com/plotarmordev/thegrill/releases/tag/v0.1.0)
from explicitly approved source `4ff02a780680a8475e83db25d187173a21d51f33`.
The tag, archive receipts and native installed-smoke summaries bind that source.
The version matches the root and `grill-perf` packages; the quality CLI remains
unpackaged.

### Published artifact verification

[Publication run 34765220280](https://github.com/plotarmordev/thegrill/actions/runs/34765220280)
passed exact-source authorization, both native build/smoke paths, source/output
scope and secret gates, and upload/tag identity checks before publishing.

| Native target | Published archive SHA-256 |
|---|---|
| `aarch64-unknown-linux-gnu` | `f0174d3b3f1687eaf365f5519d1c44eb6750856bcdadb714674dd0dca5a454cf` |
| `x86_64-unknown-linux-gnu` | `6f036b78f87fa01b47d263be2848ffd202e75e5a8bcce01d7644e1735cb40572` |

All eight published assets were downloaded and checked against GitHub's asset
digests, checksum sidecars and receipts. Both archives and executables matched
the preapproval staging bytes; both native smoke summaries report success
without Rust or a source checkout. The published ARM64 binary was also unpacked
and run outside a checkout: all four GLM/DeepSeek candidate decisions reproduced
the retained gates and role identities with zero network syscalls. Its evaluator
hash is distinct from the live collector hash; this is offline replay of the
[bounded live studies](SHARED-RECIPES.md#completed-glm-live-scope), not another
live acquisition or a claim of model-quality validation.

The public HTTPS archive download was separately exercised. Released archives
remain unchanged, including their pre-publication documentation snapshot; the
current [installation guide](INSTALL.md) supplies the published download URL.

### User-visible scope

- Stage a standalone `grill-perf` binary with the existing workloads, selection
  manifests, recipe bundle, installation guide and license notices.
- Use the existing CLI outside a source checkout. `--version`, `--help`, offline
  selection/bundle verification and baseline/control/offline comparison use the
  same implementation as a source build.
- Keep stored evidence and workload identities unchanged. The release number is
  not a workload/schema version or a measurement-contract revision.
- The quality CLI remains work in progress and is not packaged.
- Invalid capture loads no longer substitute the built-in C1/exact400 scope for
  an unverified selected capture. Scope and human acquisition counts are explicitly
  unavailable; timing-window errors name the acquisition and both duration values.
  Accounting is not promoted, and timing checks, report schemas and exit codes
  remain unchanged. Offline replay does not rewrite historical reports.
- Add offline `preflight` using the same local admission as `run`, including
  optional deployment, policy, metrics and credential inputs. It returns workload
  pins and bounded budgets without network traffic or capture creation. Deployment
  validation now names the offending field and observed UTF-8 byte count; the
  4,096-byte, nonempty and control-free limits remain unchanged.
- Add flat workload v3 with an optional typed warmup output budget, separate from
  measured output and conversation v2. Phase controls bind request/evidence
  identity, weighted ceilings, usage checks and failure reporting; incompatible
  phase controls cannot be compared as matched workloads. Old examples, recipe
  hashes and absent-override behavior remain unchanged. A 32/400 declaration
  does not establish sparkDash protocol or live-backend equivalence.

### Target and compatibility boundary

Published targets are `x86_64-unknown-linux-gnu` and
`aarch64-unknown-linux-gnu`, built natively on Ubuntu 24.04 with glibc 2.39
and Rust 1.98.0. The supported installed-runtime baseline is native Ubuntu
24.04 with glibc 2.39; older libc, musl, other operating systems and emulation
are not qualified by this release mechanism. Other Linux distributions need
separate validation or a source build.
A readable system CA trust store (Ubuntu `ca-certificates`) is required for client
initialization, including loopback HTTP. The clean-runtime smoke records the
read-only native CA bundle digest; no CA contents are packaged and TLS checks
are never disabled.

Existing evidence compatibility checks remain authoritative. A newer binary
must not reinterpret or rewrite old receipts in place. Retain the exact binary,
workload bytes and source pin needed to inspect historical evidence. Selected
concurrency, conversation and historical recipe-bundle workflows remain
separate contracts; packaging does not make them mutually comparable.

### Executed staged qualification

Preparation inspected package versions, asset paths, CI definitions, pinned
upstream license/action metadata and the approved Gitleaks release checksums.
The release workflow now gates the committed source and extracted public
payload/sidecars for scope and secrets before artifact upload. The complete
redistribution ledger and required notice payload are present; staging verifies
their exact input set and hashes rather than carrying an unresolved notice
placeholder.

Both targets completed native build, existing workspace checks, clean-runtime
installed CLI smoke, and source/output scope and secret gates at source
`ef4f9e29bb081f997e795b3dded80c6965f729da` in
[staging run 34677697749](https://github.com/plotarmordev/thegrill/actions/runs/34677697749).
The independent push and pull-request CI runs at that source also passed.
The complete local workspace gate passed 206 tests (two opt-in ignores); the
15 CPU publication-boundary tests passed without making real publication writes.

| Native target | Staged archive SHA-256 |
|---|---|
| `aarch64-unknown-linux-gnu` | `fd4beba564a940dd18a5a4005514cf47937925fdc96503f6ac265193f90b93d1` |
| `x86_64-unknown-linux-gnu` | `94078e602a3528e94be0aa406f10d02d62846bb9554c03206acc3674d1953d90` |

Each public build receipt and smoke summary binds that exact source, version,
target, archive, executable and runtime/CA-store identity. Both fixtures used the
same installed binary without Rust or a source checkout: default C1 plus an
explicitly selected/authenticated control mapping; baseline, unchanged control,
candidate and network-disabled offline replay; representative input, pin,
authentication, backend-rejection and budget failures. Raw receipts stay private.
The C1 control was inconclusive on ARM and measured slower on x86; both results
are retained, not rerun until favorable or misrepresented as a server-change
effect. Selected comparisons completed with `DESCRIPTIVE`, not a performance PASS.

These hashes identify the named staged artifacts, not every later build of
version 0.1.0. A later source revision requires new exact-source staging and
receipts before approval; these historical qualification records are not rewritten.

#### Qualification after preflight and phase-output integration

Source `dd771ac36963290cf8b1079b4c2a38be40cec949` completed both native
installed-runtime workflows and source/output gates in
[staging run 34740266773](https://github.com/plotarmordev/thegrill/actions/runs/34740266773).
The local workspace gate passed 221 Rust tests (two existing opt-in ignores)
and 15 publication-boundary tests, plus build, formatting and Clippy. The merged
source tree is identical; [post-merge CI passed](https://github.com/plotarmordev/thegrill/actions/runs/34740590731).

| Native target | Staged archive SHA-256 |
|---|---|
| `aarch64-unknown-linux-gnu` | `a9d73cfbe7e3b49e29c63112c0b5d98c450fca01e8695a972a1ace056d93cfe5` |
| `x86_64-unknown-linux-gnu` | `b5a724bcd851a8fc78389a04b1b1642f63e2825cc696f690f4abc90d56d06078` |

Both summaries report the same two neutral scenarios: C1 baseline ready,
unchanged control `INCONCLUSIVE`, candidate `IMPROVED`; selected baseline ready,
control and candidate `DESCRIPTIVE`. These are controlled CPU fixtures, not
live performance claims. Both runtimes had neither Rust nor a source checkout.
Checks covered inherited inputs, network-disabled replay, corrupt downloads,
asset hashes, input/pin/authentication/backend failures and budget exhaustion.
The downloaded archives and every payload digest were independently checked
against their checksum sidecars and build receipts.

These are historical exact-source staging records, not the published release
assets and not replacements for the earlier hashes. Published assets are pinned
separately above; historical qualification records remain unchanged.

### Remaining limits

Historical Actions artifacts require access to their retained workflow run.
Use the approved release assets linked above for published downloads; do not
substitute an arbitrary archive with the same version number.

Installation and loopback fixtures are not live-backend performance qualification,
model/template qualification, GPU measurements or causal evidence. No serving,
model or GPU calls are part of release preparation. No bit-for-bit reproducible
build claim, signature or independent execution attestation is made. Checksums
bind downloaded bytes; obtain the expected digest from the reviewed release,
not an untrusted mirror alone.

See [release procedure](../RELEASES.md) and [installation](INSTALL.md).
