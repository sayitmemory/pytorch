from contextlib import contextmanager

import torch
import torch._custom_ops
from torch._C import _ExcludeDispatchKeyGuard, DispatchKey, DispatchKeySet
from torch._export.exported_program import ModuleCallSignature
from torch._functorch.eager_transforms import _unwrap_all_tensors_from_functional
from torch._higher_order_ops.wrap import wrap
from torch._ops import HigherOrderOperator
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import ProxyTorchDispatchMode, track_tensor_tree
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import (
    _get_current_dispatch_mode,
    _pop_mode_temporarily,
)


export_tracepoint = HigherOrderOperator("export_tracepoint")


export_tracepoint.fallthrough(DispatchKey.PythonDispatcher)
export_tracepoint.fallthrough(DispatchKey.PythonTLSSnapshot)
export_tracepoint.fallthrough(DispatchKey.ADInplaceOrView)
export_tracepoint.fallthrough(DispatchKey.BackendSelect)
export_tracepoint.fallthrough(DispatchKey.AutocastCPU)
export_tracepoint.fallthrough(DispatchKey.AutogradCPU)


@export_tracepoint.py_impl(ProxyTorchDispatchMode)
def export_tracepoint_dispatch_mode(*args, **kwargs):
    mode = _get_current_dispatch_mode()
    assert mode is not None, "Mode should always be enabled for python fallback key"
    with _pop_mode_temporarily() as mode:
        if not mode.enable_tracing:
            return export_tracepoint(*args, **kwargs)
        p_args, p_kwargs = pytree.tree_map(mode.tracer.unwrap_proxy, (args, kwargs))
        proxy = mode.tracer.create_proxy(
            "call_function", export_tracepoint, p_args, p_kwargs
        )
        return track_tensor_tree(args, proxy, constant=None, tracer=mode.tracer)


@export_tracepoint.py_impl(FakeTensorMode)
def export_tracepoint_fake_tensor_mode(*args, **kwargs):
    return args


@export_tracepoint.py_impl(DispatchKey.Functionalize)
def export_tracepoint_functionalize(*args, **kwargs):
    reapply_views = torch._C._functionalization_reapply_views_tls()
    unwrapped_args = _unwrap_all_tensors_from_functional(
        args, reapply_views=reapply_views
    )
    unwrapped_kwargs = _unwrap_all_tensors_from_functional(kwargs, reapply_views=reapply_views)
    with _ExcludeDispatchKeyGuard(DispatchKeySet(DispatchKey.Functionalize)):
        return export_tracepoint(*unwrapped_args, **unwrapped_kwargs)


@export_tracepoint.py_impl(DispatchKey.CPU)
def export_tracepoint_dispatch_mode(*args, **kwargs):
    return args


def _wrap_submodule(mod, path, module_call_signatures):
    assert isinstance(mod, torch.nn.Module)
    assert path != ""
    parent = None
    submodule = mod
    for name in path.split("."):
        parent = submodule
        submodule = getattr(submodule, name)

    from torch._dynamo import assume_constant_result

    @assume_constant_result
    def update_module_call_signatures(path, in_spec, out_spec):
        assert path not in module_call_signatures
        module_call_signatures[path] = ModuleCallSignature(inputs=[], outputs=[], in_spec=in_spec, out_spec=out_spec)

    class WrappedModule(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.module_call_signatures = module_call_signatures

        def forward(self, *args, **kwargs):
            flat_args, in_spec = pytree.tree_flatten((args, kwargs))

            def flat_gm(*flat_args):
                flat_args = export_tracepoint(*flat_args, kind="module_call_in", path=path)
                args, kwargs = pytree.tree_unflatten(flat_args, in_spec)
                res = self.inner(*args, **kwargs)
                flat_res, out_spec = pytree.tree_flatten(res)
                flat_res = export_tracepoint(*flat_res, kind="module_call_out", path=path)
                update_module_call_signatures(path, in_spec, out_spec)
                return flat_res

            flat_res = wrap(flat_gm, *flat_args)
            return pytree.tree_unflatten(flat_res, self.module_call_signatures[path].out_spec)

    setattr(parent, name, WrappedModule(submodule))
    return parent, name, submodule


@contextmanager
def _wrap_submodules(f, preserve_signature, module_call_signatures):
    tasks = []

    try:
        for path in preserve_signature:
            tasks.append(_wrap_submodule(f, path, module_call_signatures))
        yield
    finally:
        for parent, name, submodule in tasks:
            setattr(parent, name, submodule)
