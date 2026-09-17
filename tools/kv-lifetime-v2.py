"""L1 allocation-generation observations for the separately versioned KV journal.

The owning source installer must pin the memory, allocator and L1-manager modules.
No backend imports, tensor access, allocator leases, device work or background work.
A successful event query must precede validate(); weak objects are never held while
waiting for device completion. These witnesses do not establish copied contents.
"""

import functools
import weakref


class Lifetime:
    def __init__(self, journal, *, memory_type, tensor_allocator_type, l1_manager_type,
                 max_allocations=65536, allocator_roots=()):
        if type(max_allocations) is not int or not 1 <= max_allocations <= 65536:
            raise ValueError("invalid allocation witness bound")
        self.journal = journal
        self.memory_type = memory_type
        self.tensor_allocator_type = tensor_allocator_type
        self.l1_manager_type = l1_manager_type
        self.max_allocations = max_allocations
        self.generations = {}
        self.objects = {}
        self.installed = False
        self.allocator_routes = [
            (owner, field, owner.free, owner.batched_free)
            for owner, field in allocator_roots
        ]
        self.free_hooks = None
        self.readers = {name: getattr(memory_type, name)
                        for name in ("is_valid", "get_shapes", "get_dtypes", "get_size")}

    def _retire(self, token, reason):
        with self.journal.condition:
            state = self.generations[token]
            if not state["retired"]:
                state["retired"] = True
                self.journal.emit("allocation_retired", allocation=token, reason=reason)

    def _retire_object(self, obj, reason):
        with self.journal.condition:
            token = self.objects.get(id(obj))
            if token is not None and self.generations[token]["ref"]() is obj:
                self._retire(token, reason)

    def _before_free(self, obj):
        # TensorMemoryAllocator.free returns the address before invalidate().
        # Observing invalidate alone leaves an unobserved physical-release gap.
        self._retire_object(obj, "free_started")

    def _before_batch_free(self, objects):
        if not isinstance(objects, list) or len(objects) > self.max_allocations:
            raise ValueError("unbounded allocation retirement batch")
        for obj in objects:
            self._retire_object(obj, "free_started")

    def _after_write_reservation(self, result):
        if not isinstance(result, dict) or len(result) > self.max_allocations:
            raise ValueError("unbounded L1 reservation result")
        for outcome in result.values():
            if not isinstance(outcome, tuple) or len(outcome) != 2:
                raise ValueError("unsupported L1 reservation result")
            # The pinned L1 manager returns a non-None object only for a
            # successful reservation. Retire even if its error enum disagrees.
            obj = outcome[1]
            if obj is not None:
                self._retire_object(obj, "write_reserved")

    def install(self):
        """Install at process construction, after whole-source verification.

        Each wrapper preserves the original result and exception. Retirement is
        conservative: a failed free/resize need not make a study qualifying.
        Installation returns undo entries for the owning atomic source installer.
        """
        if self.installed:
            raise ValueError("allocation observer already installed")
        specifications = [
            (self.tensor_allocator_type, "free", self._before_free, False),
            (self.tensor_allocator_type, "batched_free", self._before_batch_free, False),
            (self.memory_type, "invalidate",
             lambda obj: self._retire_object(obj, "invalidated"), False),
            (self.memory_type, "set_used_size",
             lambda obj: self._retire_object(obj, "size_changed"), False),
            (self.l1_manager_type, "reserve_write", self._after_write_reservation, True),
        ]
        staged = []
        for owner, name, observation, after in specifications:
            original = getattr(owner, name)
            if not callable(original) or getattr(original, "_grill_lifetime_v2", False):
                raise ValueError("incompatible allocation observer method")

            def make_call(original, observation, after, name):
                @functools.wraps(original)
                def call(instance, *args, **kwargs):
                    if after:
                        result = original(instance, *args, **kwargs)
                        self.journal.observe(observation, result)
                        return result
                    if name in ("invalidate", "set_used_size"):
                        value = instance
                    elif args:
                        value = args[0]
                    else:
                        argument = "memory_objs" if name == "batched_free" else "memory_obj"
                        value = kwargs.get(argument)
                    self.journal.observe(observation, value)
                    return original(instance, *args, **kwargs)
                call._grill_lifetime_v2 = True
                return call

            staged.append((owner, name, original,
                           make_call(original, observation, after, name)))
        replaced = []
        try:
            for owner, name, original, replacement in staged:
                self.journal.replace(owner, name, replacement)
                replaced.append((owner, name, original, replacement))
        except Exception:
            for owner, name, original, replacement in reversed(replaced):
                if getattr(owner, name) is replacement:
                    setattr(owner, name, original)
            raise
        self.installed = True
        self.free_hooks = (self.tensor_allocator_type.free, self.tensor_allocator_type.batched_free)
        self.readers.update((name, getattr(self.memory_type, name))
                            for name in ("invalidate", "set_used_size"))
        self.reserve_hook = self.l1_manager_type.reserve_write
        return replaced

    @staticmethod
    def _components(obj, registration, object_group):
        if (type(object_group) is not int
                or not 0 <= object_group < len(registration["object_groups"])):
            raise ValueError("allocation object group outside registration")
        group = registration["object_groups"][object_group]
        if group["object_group_id"] != object_group:
            raise ValueError("allocation object group mismatch")
        members = [registration["kernel_groups"][kid] for kid in group["kernel_group_ids"]]
        shapes, dtypes = obj.get_shapes(), obj.get_dtypes()
        if len(shapes) != len(members) or len(dtypes) != len(members):
            raise ValueError("allocation component inventory mismatch")
        observed = []
        for member, shape, dtype in zip(members, shapes, dtypes, strict=True):
            dimensions = list(shape)
            if (len(dimensions) != 4 or any(type(d) is not int for d in dimensions)
                    or dimensions != member["object_shape"] or str(dtype) != member["dtype"]):
                raise ValueError("allocation component shape or dtype mismatch")
            observed.append({"kernel_group_id": member["kernel_group_id"],
                             "shape": dimensions, "dtype": str(dtype)})
        size = obj.get_size()
        if (type(size) is not int or not 1 <= size < (1 << 63)
                or size != sum(member["allocation_bytes"] for member in members)):
            raise ValueError("allocation size differs from registered layout")
        return observed, size

    def _allocator_route(self, root):
        allocator = root
        if type(root) is not self.tensor_allocator_type:
            route = next((route for route in self.allocator_routes if type(root) is route[0]), None)
            if route is None:
                raise ValueError("unobserved selected allocator class")
            owner, field, free, batched_free = route
            if (getattr(root.free, "__func__", None) is not free
                    or getattr(root.batched_free, "__func__", None) is not batched_free
                    or owner.free is not free or owner.batched_free is not batched_free):
                raise ValueError("selected allocator delegation changed")
            allocator = getattr(root, field)
        if (self.free_hooks is None or type(allocator) is not self.tensor_allocator_type
                or getattr(allocator.free, "__func__", None) is not self.free_hooks[0]
                or getattr(allocator.batched_free, "__func__", None) is not self.free_hooks[1]):
            raise ValueError("selected allocator does not reach observed tensor free hooks")

    def _reader_bindings(self, obj):
        if type(obj) is not self.memory_type:
            raise ValueError("allocation reader belongs to an unknown object type")
        for name, function in self.readers.items():
            method = getattr(obj, name)
            if (name in vars(obj) or getattr(self.memory_type, name) is not function
                    or getattr(method, "__self__", None) is not obj
                    or getattr(method, "__func__", None) is not function):
                raise ValueError("allocation reader binding changed")

    def _manager_binding(self, manager):
        if (not self.installed or type(manager) is not self.l1_manager_type
                or "reserve_write" in vars(manager)
                or getattr(manager.reserve_write, "__func__", None) is not self.reserve_hook):
            raise ValueError("L1 reservation observation binding changed")

    def bind(self, key, obj, registration, object_group, l1_manager):
        """Bind an actual L1 copy operand, without retaining its object or allocator."""
        with self.journal.condition:
            self._reader_bindings(obj)
            self._manager_binding(l1_manager)
            if (type(key) is not dict
                    or set(key) != {"hash", "model", "kv_rank", "group", "salt"}
                    or type(key["group"]) is not int or key["group"] != object_group):
                raise ValueError("allocation key does not name its registered group")
            if (not self.installed or type(obj) is not self.memory_type
                    or type(l1_manager) is not self.l1_manager_type
                    or obj.is_valid() is not True
                    or obj.parent_allocator is not l1_manager._memory_manager._allocator):
                raise ValueError("copy operand is not a live selected L1 allocation")
            self._allocator_route(obj.parent_allocator)
            components, size = self._components(obj, registration, object_group)
            token = self.objects.get(id(obj))
            if token is not None:
                state = self.generations[token]
                if state["ref"]() is obj and not state["retired"]:
                    if (state["key"] != key or state["components"] != components
                            or state["bytes"] != size):
                        raise ValueError("live allocation changed identity or layout")
                    return token
            if len(self.generations) >= self.max_allocations:
                raise ValueError("allocation witness bound exceeded")
            token = len(self.generations) + 1
            reference = weakref.ref(
                obj, lambda unused, token=token: self.journal.observe(
                    self._retire, token, "collected"))
            self.generations[token] = {
                "ref": reference, "key": dict(key), "components": components,
                "bytes": size, "retired": False,
                "allocator": weakref.ref(l1_manager._memory_manager._allocator),
                "manager": weakref.ref(l1_manager),
            }
            self.objects[id(obj)] = token
            self.journal.emit("allocation", allocation=token, key=self.generations[token]["key"],
                              components=components, bytes=size)
            return token

    def validate(self, token):
        """Check only AFTER the operation's existing event reports completion.

        Dereferencing a weak object before querying an unready event could extend
        its destructor lifetime through the query. Callers must preserve this order.
        """
        with self.journal.condition:
            if type(token) is not int or token not in self.generations:
                raise ValueError("unknown allocation generation")
            state = self.generations[token]
            if state["retired"]:
                raise ValueError("allocation retired before witnessed completion")
            obj = state["ref"]()
            self._reader_bindings(obj)
            self._manager_binding(state["manager"]())
            if (obj is None or obj.is_valid() is not True
                    or obj.parent_allocator is not state["allocator"]()):
                raise ValueError("allocation no longer live at witnessed completion")
            self._allocator_route(obj.parent_allocator)
            shapes, dtypes = obj.get_shapes(), obj.get_dtypes()
            size = obj.get_size()
            if (len(shapes) != len(state["components"])
                    or len(dtypes) != len(state["components"])
                    or type(size) is not int or size != state["bytes"]):
                raise ValueError("allocation layout changed before witnessed completion")
            for shape, dtype, component in zip(shapes, dtypes, state["components"], strict=True):
                dimensions = list(shape)
                if (any(type(d) is not int for d in dimensions)
                        or dimensions != component["shape"] or str(dtype) != component["dtype"]):
                    raise ValueError("allocation component changed before witnessed completion")
