"""Bounded metadata snapshots for the separately versioned KV registration contract.

No backend imports, tensor reads, device work, process control or journal I/O.
Callers must observe successful registration at a pinned source boundary. These
snapshots establish registered topology/layout, never contents or completion.
"""

import math

SCHEMA = "thegrill.kv.registered-state.v2"
SHAPE_FIELDS = ("kv_size", "nl", "nb", "bs", "nh", "hs", "element_size", "block_stride_elems")
DTYPE_BYTES = {"torch.float16": 2, "torch.float32": 4, "torch.bfloat16": 2, "torch.uint8": 1}
FORMATS = (
    "NB_NL_TWO_BS_NH_HS", "NL_X_TWO_NB_BS_NH_HS", "NL_X_NB_TWO_BS_NH_HS",
    "NL_X_NB_BS_HS", "TWO_X_NL_X_NBBS_NH_HS", "NL_X_NBBS_ONE_HS",
    "NL_X_TWO_NB_NH_BS_HS", "NL_X_NB_TWO_NH_BS_HS", "NB_NL_TWO_NH_BS_HS",
    "TWO_X_NL_X_NB_BS_NH_HS", "NL_X_NB_NH_BS_TWO_HS", "NL_X_NB_BS_NH_TWO_HS",
    "NL_X_NB_NH_BS_CS", "NL_X_NB_BS_NH_CS", "NL_X_NB_BSV_BSS",
    "NL_X_TWO_NB_NH_ONE_BS_HS",
)


def integer(value, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError("registered integer outside contract")
    return value


def boolean(value):
    if type(value) is not bool:
        raise ValueError("registered flag is not boolean")
    return value


def text(value):
    if not isinstance(value, str) or not 0 < len(value.encode("utf-8")) <= 256:
        raise ValueError("registered name outside contract")
    return value


def window(value):
    if value == -1 and type(value) is int:
        return value
    return integer(value, 1, 1 << 20)


def hints(value):
    if type(value) is not dict or set(value) - {"kv_layout"}:
        raise ValueError("unknown registered layout hint")
    layout = value.get("kv_layout")
    if layout not in (None, "NHD", "HND"):
        raise ValueError("unsupported registered layout hint")
    return {"kv_layout": layout}


def indices(values, count):
    if not 1 <= len(values) <= count:
        raise ValueError("registered layer index count outside contract")
    result = [integer(i, 0, count - 1) for i in values]
    if not result or result != sorted(set(result)):
        raise ValueError("registered layer indices are empty or unordered")
    return result


def wire_groups(infos, tensor_count):
    if not 1 <= len(infos) <= 64:
        raise ValueError("registered group count outside contract")
    result = []
    used = set()
    for index, info in enumerate(infos):
        if (type(info.recurrent_state) is not bool or info.recurrent_state
                or type(info.extra_object_group_tag) is not int or info.extra_object_group_tag != 0):
            raise ValueError("auxiliary and recurrent groups are outside the attention-only contract")
        layers = indices(info.layer_indices, tensor_count)
        if used.intersection(layers):
            raise ValueError("registered layer belongs to multiple wire groups")
        used.update(layers)
        result.append({
            "wire_group_index": index,
            "engine_group_id": integer(info.engine_group_id, 0, 63),
            "layer_indices": layers,
            "tokens_per_block": integer(info.tokens_per_block, 1, 1 << 20),
            "sw_size_tokens": window(info.sw_size_tokens),
        })
    spaces = {group["engine_group_id"] for group in result}
    if spaces != set(range(len(spaces))):
        raise ValueError("registered engine group IDs are not dense")
    return result, used


def worker_registration(worker, vllm_config, layout_hints):
    """Snapshot the final worker registration after the actual wire ACK.

    layout_hints must be the argument sent by that registration, not a second
    runtime query. The configuration must belong to this worker's connector.
    """
    strategy = worker.parallel_strategy
    pc = vllm_config.parallel_config
    mc = vllm_config.model_config
    if integer(strategy.dcp_size, 1, 64) != 1:
        raise ValueError("DCP token sharding is outside the selected observer contract")
    if integer(pc.decode_context_parallel_size, 1, 64) != 1:
        raise ValueError("connector DCP configuration is outside the selected observer contract")
    physical = {
        "vllm_world_size": integer(strategy.vllm_world_size, 1, 64),
        "vllm_worker_id": integer(strategy.vllm_worker_id, 0, 63),
        "tp_size": integer(strategy.tp_size, 1, 64),
        "pp_size": integer(strategy.pp_size, 1, 64),
        "dp_size": integer(pc.data_parallel_size, 1, 64),
        "n_servers": integer(strategy.n_servers, 1, 64),
    }
    world = physical["vllm_world_size"]
    rank = physical["vllm_worker_id"]
    tp, pp, servers = physical["tp_size"], physical["pp_size"], physical["n_servers"]
    if (world != tp * pp or rank >= world or world % servers or tp % servers
            or (servers > 1 and (pp != 1 or physical["dp_size"] != 1))):
        raise ValueError("unsupported registered physical topology")
    if (integer(pc.world_size, 1, 64) != world or integer(pc.rank, 0, 63) != rank
            or integer(pc.tensor_parallel_size, 1, 64) != tp
            or integer(pc.pipeline_parallel_size, 1, 64) != pp):
        raise ValueError("connector and worker physical topology disagree")
    flags = {"mla_enabled": boolean(mc.use_mla), "is_hybrid": boolean(mc.is_hybrid),
             "mla_only": boolean(strategy.mla_only)}
    if flags["mla_only"] != (flags["mla_enabled"] and not flags["is_hybrid"]):
        raise ValueError("registered MLA projection disagrees with model flags")
    partition = {
        "world_size": integer(worker.world_size, 1, 64),
        "rank": integer(worker.worker_id, 0, 63),
        "kv_tp_size": integer(strategy.kv_tp_size, 1, 64),
        "is_writer": boolean(worker.is_kv_writer),
    }
    expected = {
        "world_size": world // tp if flags["mla_only"] else world // servers,
        "rank": rank // tp if flags["mla_only"] else rank % (world // servers),
        "kv_tp_size": tp // servers,
        "is_writer": rank % (tp // servers) == 0 if flags["mla_only"] else True,
    }
    if partition != expected:
        raise ValueError("registered cache projection disagrees with physical topology")
    readers = partition["kv_tp_size"] if flags["mla_only"] else 1
    if integer(strategy.num_kv_readers, 1, 64) != readers:
        raise ValueError("registered reader count disagrees with physical topology")
    count = integer(len(worker.kv_caches), 1, 4096)
    layers = [{"tensor_index": i, "name": text(name)}
              for i, name in enumerate(worker.kv_caches)]
    if len({layer["name"] for layer in layers}) != count:
        raise ValueError("duplicate registered layer name")
    groups, _ = wire_groups(worker.engine_group_infos, count)
    return {
        "kind": "worker_registration", "schema": SCHEMA,
        "engine_instance": integer(worker.instance_id, 0, (1 << 63) - 1),
        "model": text(worker.model_name), "engine_type": "VLLM",
        "physical": physical, "model_flags": flags, "cache_partition": partition,
        "server_slot": rank // (world // servers), "layout_hints": hints(layout_hints),
        "registered_layers": layers, "wire_engine_groups": groups,
    }


def cache_registration(values):
    """Snapshot locals after insertion of a new cache ContextEntry.

    A no-op re-registration must never call this function. Do not retain the
    input locals or any tensor/context reference in the journal.
    """
    owner = values["self"]
    context = values["cache_context"]
    manager = context.kv_layer_groups_manager
    engine_type = values["engine_type"]
    if engine_type.name != "VLLM":
        raise ValueError("unsupported registration engine")
    count = integer(context.num_layers, 1, 4096)
    if len(values["kv_caches"]) != count:
        raise ValueError("registered tensor count changed during context construction")
    groups, used = wire_groups(values["engine_group_infos"], count)
    chunk = integer(owner._ctx.chunk_size, 1, 1 << 20)
    separate = boolean(owner._ctx.separate_object_groups)
    full = boolean(owner._ctx.full_sw_kv)
    if full:
        raise ValueError("full sliding-window blend mode is outside the VLLM contract")
    kernels = manager.kernel_groups
    objects = manager.object_groups
    if len(kernels) != len(groups) or not 1 <= len(objects) <= 64:
        raise ValueError("registered wire/kernel/object group count mismatch")
    memberships = {}
    object_rows = []
    attention = manager.get_attn_desc()
    if (len(attention.group_kinds) != len(objects)
            or any(kind != "attention" for kind in attention.group_kinds)):
        raise ValueError("non-attention object groups are outside the observer contract")
    if len(attention.num_chunks_in_sw) != len(objects):
        raise ValueError("registered attention/object group count mismatch")
    for oid, obj in enumerate(objects):
        members = indices(obj.kernel_group_indices, len(kernels))
        for kid in members:
            if kid in memberships:
                raise ValueError("kernel belongs to multiple registered object groups")
            memberships[kid] = oid
        cross_window = window(obj.sw_size_chunks)
        if cross_window != -1:
            integer(cross_window, 1, 1024)
        if attention.num_chunks_in_sw[oid] != cross_window:
            raise ValueError("registered object window differs from transfer policy")
        object_rows.append({"object_group_id": oid, "kernel_group_ids": members,
                            "num_chunks_in_sw": cross_window})
    if set(memberships) != set(range(len(kernels))):
        raise ValueError("registered kernel group has no object owner")
    kernel_rows = []
    expected_classes = {}
    for kid, group in enumerate(kernels):
        wire = groups[kid]
        shape = {name: integer(getattr(group.shape_desc, name),
                               0 if name == "block_stride_elems" else 1,
                               (1 << 63) - 1 if name == "block_stride_elems" else (1 << 31) - 1)
                 for name in SHAPE_FIELDS}
        dtype = str(group.dtype)
        if dtype not in DTYPE_BYTES or shape["element_size"] != DTYPE_BYTES[dtype]:
            raise ValueError("registered dtype does not match its element size")
        if group.engine_kv_format is None:
            raise ValueError("registered kernel has no resolved engine format")
        format_code = integer(int(group.engine_kv_format), 0, len(FORMATS) - 1)
        format_name = group.engine_kv_format.name
        if FORMATS[format_code] != format_name:
            raise ValueError("registered format code/name mismatch")
        tokens = integer(group.tokens_per_block, 1, 1 << 20)
        sw = window(group.sw_size_tokens)
        layers = indices(group.layer_indices, count)
        if (wire["engine_group_id"] != integer(group.engine_group_idx, 0, 63)
                or wire["layer_indices"] != layers or wire["tokens_per_block"] != tokens
                or wire["sw_size_tokens"] != sw or shape["nl"] != len(layers)):
            raise ValueError("registered wire/kernel identity mismatch")
        if tokens % shape["bs"] or chunk % tokens or (0 < sw < chunk and sw % tokens):
            raise ValueError("registered logical/physical chunk alignment mismatch")
        retained = chunk if sw == -1 or sw >= chunk else sw
        expected_shape = [shape["kv_size"], shape["nl"], retained * shape["bs"] // tokens,
                          shape["nh"] * shape["hs"]]
        actual_shape, actual_dtype = context.get_kernel_group_shape_dtype(chunk, kid)
        if len(actual_shape) != 4:
            raise ValueError("registered object allocation must have four dimensions")
        object_shape = [integer(dimension, 1, (1 << 31) - 1) for dimension in actual_shape]
        if object_shape != expected_shape or str(actual_dtype) != dtype:
            raise ValueError("registered object allocation differs from logical layout")
        size = integer(math.prod(object_shape) * DTYPE_BYTES[dtype], 1, (1 << 63) - 1)
        kernel_rows.append({
            "kernel_group_id": kid, "engine_group_id": wire["engine_group_id"],
            "object_group_id": memberships[kid], "layer_indices": layers, "shape": shape,
            "dtype": dtype, "engine_kv_format_code": format_code,
            "engine_kv_format": format_name, "tokens_per_block": tokens,
            "sw_size_tokens": sw, "object_shape": object_shape, "allocation_bytes": size,
        })
        classification = -1 if not separate or sw == -1 else (sw + chunk - 1) // chunk
        expected_classes.setdefault(classification, []).append(kid)
    expected_objects = [{"object_group_id": oid, "kernel_group_ids": members,
                         "num_chunks_in_sw": classification}
                        for oid, (classification, members) in enumerate(expected_classes.items())]
    if object_rows != expected_objects:
        raise ValueError("registered object partition differs from window policy")
    layouts = values["group_layout_descs"]
    if set(layouts) != set(range(len(objects))):
        raise ValueError("registered allocation descriptor roster mismatch")
    for oid, obj in enumerate(object_rows):
        layout = layouts[oid]
        members = [kernel_rows[kid] for kid in obj["kernel_group_ids"]]
        if ([list(shape) for shape in layout.shapes] != [row["object_shape"] for row in members]
                or [str(dtype) for dtype in layout.dtypes] != [row["dtype"] for row in members]):
            raise ValueError("registered allocation component order or dtype mismatch")
        integer(sum(row["allocation_bytes"] for row in members), 1, (1 << 63) - 1)
    return {
        "kind": "cache_registration", "schema": SCHEMA, "registration_kind": "new",
        "engine_instance": integer(values["instance_id"], 0, (1 << 63) - 1),
        "model": text(values["model_name"]), "engine_type": engine_type.name,
        "cache_world_size": integer(values["world_size"], 1, 64),
        "layout_hints": hints(values["layout_hints"]),
        "server_policy": {"separate_object_groups": separate, "full_sw_kv": full},
        "registered_tensor_count": count,
        "excluded_tensor_indices": sorted(set(range(count)) - used),
        "wire_engine_groups": groups, "lmcache_tokens_per_chunk": chunk,
        "kernel_groups": kernel_rows, "object_groups": object_rows,
    }
