/* Synthetic CPU fixture, NOT NVIDIA NVML and never linked to a device library.
 * ABI subset from NVIDIA nvml_dev v12.9.40/nvml.h; see resources/nvml.rs
 * and docs/performance/RESOURCES.md for the source pin and required notices.
 * This fixture deliberately omits the memory symbol while exposing power.
 */
#include <stdint.h>
#include <string.h>

typedef struct device_st *device_t;
union value { double d; int si; unsigned int ui; unsigned long ul;
    unsigned long long ull; long long sll; unsigned short us; };
struct field { unsigned int field_id, scope_id; long long timestamp, latency;
    int value_type, code; union value value; };
static int initialized;
static unsigned int init_flags;
static const char uuid[] = "GPU-00000000-0000-0000-0000-000000000001";
int nvmlInitWithFlags(unsigned int flags) {
    if ((flags != 0 && flags != 2) || initialized) return 2;
    initialized = 1; init_flags = flags; return 0;
}
int nvmlShutdown(void) {
    if (!initialized) return 1;
    initialized = 0; return 0;
}
static int text(char *out, unsigned int cap, const char *s) {
    if (!initialized) return 1;
    if (cap < strlen(s) + 1) return 7;
    memcpy(out, s, strlen(s) + 1); return 0;
}
int nvmlSystemGetNVMLVersion(char *out, unsigned int cap) {
    return text(out, cap, "fixture-nvml");
}
int nvmlSystemGetDriverVersion(char *out, unsigned int cap) {
    return text(out, cap, "fixture-driver");
}
int nvmlDeviceGetHandleByUUID(const char *requested, device_t *device) {
    if (!initialized) return 1;
    /* Reproduce the observed driver behavior without loading NVIDIA libraries. */
    if (init_flags == 2 || strcmp(requested, uuid)) return 6;
    *device = (device_t)(uintptr_t)1; return 0;
}
int nvmlDeviceGetUUID(device_t device, char *out, unsigned int cap) {
    if (device != (device_t)(uintptr_t)1) return 2;
    return text(out, cap, uuid);
}
int nvmlDeviceGetFieldValues(device_t device, int count, struct field *out) {
    if (!initialized) return 1;
    if (device != (device_t)(uintptr_t)1 || count != 1 || out->field_id != 186 || out->scope_id != 0) return 2;
    out->timestamp = 123456789; out->latency = 17;
    out->value_type = 1; out->code = 0; out->value.ui = 125123;
    return 0;
}
