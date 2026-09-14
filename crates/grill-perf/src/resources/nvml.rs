//! Opt-in NVML reads, never CUDA, discovery, index selection or a subprocess.
//! ABI: NVIDIA nvml_dev v12.9.40/nvml.h, SHA256 below. See RESOURCES.md.
// Copyright 1993-2025 NVIDIA Corporation. All rights reserved.
// NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
// CODE FOR ANY PURPOSE. IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
// IMPLIED WARRANTY OF ANY KIND. NVIDIA DISCLAIMS ALL WARRANTIES WITH
// REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
// MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
// IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
// OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
// OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
// OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
// OR PERFORMANCE OF THIS SOURCE CODE.
// U.S. Government End Users. This source code is a "commercial item" as
// that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting of
// "commercial computer software" and "commercial computer software
// documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995)
// and is provided to the U.S. Government only as a commercial end item.
// Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through
// 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the
// source code with only those rights set forth herein.

use super::{Failure, Metric, Raw, Reading, Target, Unit, reading};
use libc::{c_char, c_double, c_int, c_longlong, c_uint, c_ulong, c_ulonglong, c_ushort, c_void};
use serde::{Deserialize, Serialize};
use std::ffi::{CStr, CString};

pub(super) const ADAPTER: &str = "linux-nvml-resource-v1";
pub(super) const ABI_SHA256: &str =
    "5a9ed049520b02354b74280f3a1e83b27292cf9f7958beb227461c0bc8676b99";
const LIBRARY: &CStr = c"libnvidia-ml.so.1";
// Maximum encoded trace is below this reservation: eight bounded calls, two
// 80-byte version buffers, two 96-byte UUID buffers, fixed-size records and pins.
const TRACE_CAP: usize = 8192;
const POWER_INSTANT: c_uint = 186;
const GPU_SCOPE: c_uint = 0;
const INIT_FLAGS: c_uint = 0;
type Device = *mut c_void;
type Init = unsafe extern "C" fn(c_uint) -> c_int;
type Shutdown = unsafe extern "C" fn() -> c_int;
type Text = unsafe extern "C" fn(*mut c_char, c_uint) -> c_int;
type Handle = unsafe extern "C" fn(*const c_char, *mut Device) -> c_int;
type Uuid = unsafe extern "C" fn(Device, *mut c_char, c_uint) -> c_int;
type Memory = unsafe extern "C" fn(Device, *mut MemoryInfo) -> c_int;
type Power = unsafe extern "C" fn(Device, c_int, *mut FieldValue) -> c_int;

#[repr(C)]
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct MemoryInfo {
    total: c_ulonglong,
    free: c_ulonglong,
    used: c_ulonglong,
}
#[repr(C)]
union NvmlValue {
    d: c_double,
    ui: c_uint,
    ul: c_ulong,
    ull: c_ulonglong,
    sll: c_longlong,
    si: c_int,
    us: c_ushort,
}
#[repr(C)]
struct FieldValue {
    field_id: c_uint,
    scope_id: c_uint,
    timestamp: c_longlong,
    latency_us: c_longlong,
    value_type: c_int,
    code: c_int,
    value: NvmlValue,
}

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Api {
    Init,
    Version,
    Driver,
    Handle,
    UuidBefore,
    Memory,
    Power,
    UuidAfter,
    Shutdown,
}
impl Api {
    fn symbol(self) -> &'static CStr {
        match self {
            Self::Init => c"nvmlInitWithFlags",
            Self::Version => c"nvmlSystemGetNVMLVersion",
            Self::Driver => c"nvmlSystemGetDriverVersion",
            Self::Handle => c"nvmlDeviceGetHandleByUUID",
            Self::UuidBefore | Self::UuidAfter => c"nvmlDeviceGetUUID",
            Self::Memory => c"nvmlDeviceGetMemoryInfo",
            Self::Power => c"nvmlDeviceGetFieldValues",
            Self::Shutdown => c"nvmlShutdown",
        }
    }
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Scalar {
    DoubleBits(u64),
    UnsignedInt(u32),
    UnsignedLong(u64),
    UnsignedLongLong(u64),
    SignedLongLong(i64),
    SignedInt(i32),
    UnsignedShort(u16),
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct PowerValue {
    field_id: u32,
    scope_id: u32,
    timestamp_us: i64,
    latency_us: i64,
    value_type: i32,
    code: i32,
    value: Option<Scalar>,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(
    tag = "kind",
    content = "value",
    rename_all = "snake_case",
    deny_unknown_fields
)]
enum Payload {
    Text(Vec<u8>),
    Handle(bool),
    Memory(MemoryInfo),
    Power(PowerValue),
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(tag = "state", rename_all = "snake_case", deny_unknown_fields)]
enum Returned {
    MissingSymbol,
    Called { code: i32, value: Option<Payload> },
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Call {
    api: Api,
    result: Returned,
}
#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum LoadFailure {
    UnsupportedHost,
    LibraryUnavailable,
    ShutdownSymbolMissing,
}
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
struct Trace {
    version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    init_flags: Option<u32>,
    abi_sha256: String,
    library: String,
    // This is the adapter ABI contract, not a fabricated observed driver version.
    abi: String,
    load_failure: Option<LoadFailure>,
    calls: Vec<Call>,
}
impl Trace {
    fn new() -> Self {
        Self {
            version: 2,
            init_flags: Some(INIT_FLAGS),
            abi_sha256: ABI_SHA256.into(),
            library: LIBRARY.to_string_lossy().into_owned(),
            abi: "linux-lp64-v1".into(),
            load_failure: None,
            calls: Vec::with_capacity(9),
        }
    }
    fn push(&mut self, api: Api, code: i32, value: Option<Payload>) -> bool {
        self.calls.push(Call {
            api,
            result: Returned::Called {
                code,
                value: if code == 0 { value } else { None },
            },
        });
        code == 0
    }
}

struct Library {
    handle: *mut c_void,
    shutdown: Shutdown,
}
// SAFETY: this owns one dlopen reference and never exposes it. Linux dlopen/
// dlclose and the pinned NVML API are thread-safe (nvml.h, introductory contract).
// Device handles and initialized sessions stay inside one synchronous collect
// call; moving the owner between sampler tasks cannot move an in-flight call.
// Do not infer Sync: the observer retains exclusive ownership of its library.
unsafe impl Send for Library {}
impl Library {
    fn open(name: &CStr) -> std::result::Result<Self, LoadFailure> {
        if !cfg!(all(
            target_os = "linux",
            target_pointer_width = "64",
            any(target_arch = "aarch64", target_arch = "x86_64")
        )) {
            return Err(LoadFailure::UnsupportedHost);
        }
        // SAFETY: fixed soname, platform ABI checked above, no persistent driver link.
        let handle = unsafe { libc::dlopen(name.as_ptr(), libc::RTLD_LOCAL | libc::RTLD_NOW) };
        if handle.is_null() {
            return Err(LoadFailure::LibraryUnavailable);
        }
        let shutdown = unsafe { libc::dlsym(handle, Api::Shutdown.symbol().as_ptr()) };
        if shutdown.is_null() {
            unsafe {
                libc::dlclose(handle);
            }
            return Err(LoadFailure::ShutdownSymbolMissing);
        }
        // SAFETY: exact documented symbol and C function prototype; library stays live.
        Ok(Self {
            handle,
            shutdown: unsafe { std::mem::transmute::<*mut c_void, Shutdown>(shutdown) },
        })
    }
    fn symbol(&self, api: Api, trace: &mut Trace) -> Option<*mut c_void> {
        let symbol = unsafe { libc::dlsym(self.handle, api.symbol().as_ptr()) };
        if symbol.is_null() {
            trace.calls.push(Call {
                api,
                result: Returned::MissingSymbol,
            });
            None
        } else {
            Some(symbol)
        }
    }
    fn collect(&self, uuid: &str, memory: bool, trace: &mut Trace) {
        let Some(symbol) = self.symbol(Api::Init, trace) else {
            return;
        };
        let init = unsafe { std::mem::transmute::<*mut c_void, Init>(symbol) };
        // Lazy v2 initialization permits the selected UUID lookup; NO_ATTACH can
        // make a present device unresolvable. Never fall back to legacy nvmlInit.
        if !trace.push(Api::Init, unsafe { init(INIT_FLAGS) }, None) {
            return;
        }
        self.collect_initialized(uuid, memory, trace);
        trace.push(Api::Shutdown, unsafe { (self.shutdown)() }, None);
    }
    fn collect_initialized(&self, uuid: &str, memory: bool, trace: &mut Trace) {
        for api in [Api::Version, Api::Driver] {
            let Some(symbol) = self.symbol(api, trace) else {
                return;
            };
            let get = unsafe { std::mem::transmute::<*mut c_void, Text>(symbol) };
            let mut bytes = vec![0u8; 80];
            let code = unsafe { get(bytes.as_mut_ptr().cast(), bytes.len() as c_uint) };
            if !trace.push(api, code, Some(Payload::Text(bytes))) {
                return;
            }
        }
        let Some(symbol) = self.symbol(Api::Handle, trace) else {
            return;
        };
        let get = unsafe { std::mem::transmute::<*mut c_void, Handle>(symbol) };
        // Admission already requires a full ASCII UUID; no evidence-selected symbol/path.
        let Ok(uuid) = CString::new(uuid) else {
            return;
        };
        let mut device = std::ptr::null_mut();
        let code = unsafe { get(uuid.as_ptr(), &mut device) };
        if !trace.push(Api::Handle, code, Some(Payload::Handle(!device.is_null())))
            || device.is_null()
        {
            return;
        }
        if !self.uuid(device, Api::UuidBefore, trace) {
            return;
        }
        let api = if memory { Api::Memory } else { Api::Power };
        let Some(symbol) = self.symbol(api, trace) else {
            return;
        };
        let (code, payload) = if memory {
            let get = unsafe { std::mem::transmute::<*mut c_void, Memory>(symbol) };
            let mut value = MemoryInfo {
                total: 0,
                free: 0,
                used: 0,
            };
            let code = unsafe { get(device, &mut value) };
            (code, Payload::Memory(value))
        } else {
            let get = unsafe { std::mem::transmute::<*mut c_void, Power>(symbol) };
            let mut field = FieldValue {
                field_id: POWER_INSTANT,
                scope_id: GPU_SCOPE,
                timestamp: 0,
                latency_us: 0,
                value_type: -1,
                code: -1,
                value: NvmlValue { ull: 0 },
            };
            let code = unsafe { get(device, 1, &mut field) };
            // Only read the active union member after BOTH return codes succeed.
            let value = if code == 0 && field.code == 0 {
                unsafe {
                    match field.value_type {
                        0 => Some(Scalar::DoubleBits(field.value.d.to_bits())),
                        1 => Some(Scalar::UnsignedInt(field.value.ui)),
                        2 => Some(Scalar::UnsignedLong(field.value.ul)),
                        3 => Some(Scalar::UnsignedLongLong(field.value.ull)),
                        4 => Some(Scalar::SignedLongLong(field.value.sll)),
                        5 => Some(Scalar::SignedInt(field.value.si)),
                        6 => Some(Scalar::UnsignedShort(field.value.us)),
                        _ => None,
                    }
                }
            } else {
                None
            };
            (
                code,
                Payload::Power(PowerValue {
                    field_id: field.field_id,
                    scope_id: field.scope_id,
                    timestamp_us: field.timestamp,
                    latency_us: field.latency_us,
                    value_type: field.value_type,
                    code: field.code,
                    value,
                }),
            )
        };
        if !trace.push(api, code, Some(payload)) {
            return;
        }
        self.uuid(device, Api::UuidAfter, trace);
    }
    fn uuid(&self, device: Device, api: Api, trace: &mut Trace) -> bool {
        let Some(symbol) = self.symbol(api, trace) else {
            return false;
        };
        let get = unsafe { std::mem::transmute::<*mut c_void, Uuid>(symbol) };
        let mut bytes = vec![0u8; 96];
        let code = unsafe { get(device, bytes.as_mut_ptr().cast(), bytes.len() as c_uint) };
        trace.push(api, code, Some(Payload::Text(bytes)))
    }
}
impl Drop for Library {
    fn drop(&mut self) {
        unsafe {
            libc::dlclose(self.handle);
        }
    }
}

/// One observer-owned lazy library; host-only observations never construct it.
/// Each selected read balances init/shutdown and retains both codes. Initialization
/// and all NVML read overhead belong to the snapshot, not an uncharged setup phase.
#[derive(Default)]
pub(super) struct NvmlSource {
    library: Option<std::result::Result<Library, LoadFailure>>,
}
impl NvmlSource {
    pub(super) fn read(&mut self, uuid: &str, memory: bool, cap: usize) -> Raw {
        let mut raw = Raw {
            name: "nvml.json".into(),
            bytes: Vec::new(),
            error: None,
        };
        // Do not call a driver when the complete finite trace cannot be retained.
        if cap < TRACE_CAP {
            raw.error = Some(Failure::ByteBudget);
            return raw;
        }
        let library = self.library.get_or_insert_with(|| Library::open(LIBRARY));
        let mut trace = Trace::new();
        match library {
            Ok(library) => library.collect(uuid, memory, &mut trace),
            Err(error) => trace.load_failure = Some(*error),
        }
        match serde_json::to_vec(&trace) {
            Ok(bytes) if bytes.len() <= TRACE_CAP => raw.bytes = bytes,
            _ => raw.error = Some(Failure::ByteBudget),
        }
        raw
    }
}

pub(super) fn selected(target: &Target) -> Option<(&str, bool)> {
    match target {
        Target::NvidiaMemory { uuid, .. } => Some((uuid, true)),
        Target::NvidiaPower { uuid, .. } => Some((uuid, false)),
        _ => None,
    }
}
pub(super) fn valid_uuid(uuid: &str) -> bool {
    let Some(value) = uuid.strip_prefix("GPU-") else {
        return false;
    };
    value.len() == 36
        && value.bytes().enumerate().all(|(i, b)| {
            if [8, 13, 18, 23].contains(&i) {
                b == b'-'
            } else {
                b.is_ascii_hexdigit()
            }
        })
}
fn return_failure(code: i32) -> Failure {
    match code {
        3 | 13 | 25 | 26 => Failure::UnsupportedMetric,
        4 | 17 => Failure::Permission,
        6 | 9 | 12 | 21 | 28 => Failure::Missing,
        10 => Failure::Deadline,
        15 | 16 => Failure::SourceChanged,
        2 | 7 => Failure::Malformed,
        _ => Failure::Io,
    }
}
fn text(payload: &Payload, len: usize) -> std::result::Result<&str, Failure> {
    let Payload::Text(bytes) = payload else {
        return Err(Failure::Malformed);
    };
    if bytes.len() != len {
        return Err(Failure::Malformed);
    }
    let end = bytes
        .iter()
        .position(|b| *b == 0)
        .ok_or(Failure::Malformed)?;
    let value = std::str::from_utf8(&bytes[..end]).map_err(|_| Failure::Malformed)?;
    if value.is_empty() || !value.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(Failure::Malformed);
    }
    Ok(value)
}
fn power(value: &PowerValue) -> std::result::Result<u64, Failure> {
    if value.field_id != POWER_INSTANT || value.scope_id != GPU_SCOPE {
        return Err(Failure::Malformed);
    }
    if value.code != 0 {
        return if value.value.is_some() {
            Err(Failure::Malformed)
        } else {
            Err(return_failure(value.code))
        };
    }
    let milliwatts = match (&value.value, value.value_type) {
        (Some(Scalar::UnsignedInt(n)), 1) => u64::from(*n),
        (Some(Scalar::UnsignedLong(n)), 2) | (Some(Scalar::UnsignedLongLong(n)), 3) => *n,
        (Some(Scalar::DoubleBits(_)), 0)
        | (Some(Scalar::SignedLongLong(_)), 4)
        | (Some(Scalar::SignedInt(_)), 5)
        | (Some(Scalar::UnsignedShort(_)), 6)
        | (None, 7..)
        | (None, i32::MIN..=-1) => return Err(Failure::UnsupportedMetric),
        _ => return Err(Failure::Malformed),
    };
    if value.timestamp_us <= 0 || value.latency_us < 0 {
        return Err(Failure::Malformed);
    }
    milliwatts.checked_mul(1000).ok_or(Failure::Overflow)
}

/// Pure replay: no dlopen, API calls, device handles, or clock mixing.
pub(super) fn parse(
    bytes: &[u8],
    uuid: &str,
    memory: bool,
) -> std::result::Result<(Option<String>, Vec<Reading>), Failure> {
    if bytes.len() > TRACE_CAP {
        return Err(Failure::ByteBudget);
    }
    let trace: Trace = serde_json::from_slice(bytes).map_err(|_| Failure::Malformed)?;
    // Version 1 implicitly used NO_ATTACH. Preserve its replay, not its collection.
    if !matches!((trace.version, trace.init_flags), (1, None) | (2, Some(0)))
        || trace.abi_sha256 != ABI_SHA256
        || trace.library != "libnvidia-ml.so.1"
        || trace.abi != "linux-lp64-v1"
        || trace.calls.len() > 9
    {
        return Err(Failure::Malformed);
    }
    if let Some(failure) = trace.load_failure {
        if !trace.calls.is_empty() {
            return Err(Failure::Malformed);
        }
        return Err(match failure {
            LoadFailure::LibraryUnavailable => Failure::Missing,
            LoadFailure::UnsupportedHost | LoadFailure::ShutdownSymbolMissing => {
                Failure::UnsupportedMetric
            }
        });
    }
    let order = [
        Api::Init,
        Api::Version,
        Api::Driver,
        Api::Handle,
        Api::UuidBefore,
        if memory { Api::Memory } else { Api::Power },
        Api::UuidAfter,
    ];
    let mut initialized = false;
    let mut stopped = false;
    let mut first_failure = None;
    let mut position = 0;
    let mut version = None;
    let mut driver = None;
    let mut readings = Vec::new();
    let mut shutdown = false;
    for call in &trace.calls {
        if call.api == Api::Shutdown {
            if !initialized || shutdown || (!stopped && position != order.len()) {
                return Err(Failure::Malformed);
            }
            shutdown = true;
            match call.result {
                Returned::Called { code, value: None } => {
                    if code != 0 {
                        first_failure.get_or_insert(return_failure(code));
                    }
                }
                _ => return Err(Failure::Malformed),
            }
            continue;
        }
        if shutdown || stopped || order.get(position) != Some(&call.api) {
            return Err(Failure::Malformed);
        }
        position += 1;
        let payload = match &call.result {
            Returned::MissingSymbol => {
                stopped = true;
                first_failure.get_or_insert(Failure::UnsupportedMetric);
                continue;
            }
            Returned::Called { code, value } if *code != 0 => {
                if value.is_some() {
                    return Err(Failure::Malformed);
                }
                stopped = true;
                first_failure.get_or_insert(return_failure(*code));
                continue;
            }
            Returned::Called { value, .. } => value,
        };
        if call.api == Api::Init {
            if payload.is_some() {
                return Err(Failure::Malformed);
            }
            initialized = true;
            continue;
        }
        let payload = payload.as_ref().ok_or(Failure::Malformed)?;
        let parsed = match call.api {
            Api::Version => text(payload, 80).map(|s| version = Some(s)),
            Api::Driver => text(payload, 80).map(|s| driver = Some(s)),
            Api::Handle => match payload {
                Payload::Handle(true) => Ok(()),
                Payload::Handle(false) => {
                    stopped = true;
                    Err(Failure::Malformed)
                }
                _ => Err(Failure::Malformed),
            },
            Api::UuidBefore | Api::UuidAfter => text(payload, 96).and_then(|s| {
                if s == uuid && valid_uuid(s) {
                    Ok(())
                } else {
                    Err(Failure::SourceChanged)
                }
            }),
            Api::Memory => match payload {
                Payload::Memory(m)
                    if m.total > 0
                        && m.total != u64::MAX
                        && m.free.checked_add(m.used) == Some(m.total) =>
                {
                    readings = vec![
                        reading(Metric::MemoryUsed, Unit::Bytes, Ok(m.used)),
                        reading(Metric::MemoryFree, Unit::Bytes, Ok(m.free)),
                        reading(Metric::MemoryLimit, Unit::Bytes, Ok(m.total)),
                    ];
                    Ok(())
                }
                _ => Err(Failure::Malformed),
            },
            Api::Power => match payload {
                Payload::Power(p) => power(p)
                    .map(|n| readings = vec![reading(Metric::Power, Unit::Microwatts, Ok(n))]),
                _ => Err(Failure::Malformed),
            },
            _ => Err(Failure::Malformed),
        };
        if let Err(failure) = parsed {
            first_failure.get_or_insert(failure);
        }
    }
    if initialized != shutdown || (!stopped && position != order.len()) || position == 0 {
        return Err(Failure::Malformed);
    }
    if let Some(failure) = first_failure {
        return Err(failure);
    }
    // Compact hash binds observed UUID/runtime versions/API pin. Rank remains a
    // separate operator declaration in Target, not an observed device property.
    let identity =
        serde_json::to_vec(&(ABI_SHA256, uuid, version, driver)).map_err(|_| Failure::Malformed)?;
    Ok((
        Some(format!("nvml:{}", super::evidence::digest(&identity))),
        readings,
    ))
}

#[cfg(test)]
#[path = "../../tests/support/resources_nvml.rs"]
mod tests;
