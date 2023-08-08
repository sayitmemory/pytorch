# Owner(s): ["module: dynamo"]
import dataclasses
import unittest

import torch
import torch._dynamo as torchdynamo
from torch._export import export, dynamic_dim, DEFAULT_EXPORT_DYNAMO_CONFIG
from torch._export.utils import register_dataclass_as_pytree_node
from torch._export.constraints import constrain_as_size, constrain_as_value
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.utils._pytree import tree_flatten, tree_unflatten, LeafSpec, TreeSpec
from functorch.experimental.control_flow import map
from contextlib import contextmanager
from dataclasses import dataclass


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestDynamismExpression(TestCase):
    def test_export_inline_constraints(self):

        def f(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        ref = f(*inp)

        gm = export(f, inp)
        res = gm(*inp)

        self.assertTrue(torchdynamo.utils.same(ref, res))

        gm = make_fx(f, tracing_mode="symbolic")(*inp)
        res = gm(*inp)
        self.assertTrue(torchdynamo.utils.same(ref, res))

    def test_export_constraints_error(self):
        def invalid_size(x):
            b = x.item()
            constrain_as_size(b, min=0, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Unable to set min size"):
            export(invalid_size, inp)

        def invalid_input_conflict_with_inline_constraints(x):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([6]),)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid value 6 for range"):
            export(invalid_input_conflict_with_inline_constraints, inp)

        def invalid_input_conflict_with_input_constraints(x):
            return x + 1

        inp = torch.zeros([3])
        inp_constraints = [
            dynamic_dim(inp, 0) > 5,
        ]
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "not in range"):
            export(
                invalid_input_conflict_with_input_constraints,
                (inp,),
                constraints=inp_constraints,
            )


        def conflicting_constraints(x):
            b = x.item()
            constrain_as_size(b, min=2, max=3)
            constrain_as_size(b, min=4, max=5)
            return torch.full((b, 1), 1)

        inp = (torch.tensor([3]),)

        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Invalid ranges"):
            export(conflicting_constraints, inp)

    def test_export_assume_static_by_default(self):
        def branch_on_shape(x: torch.Tensor):
            if x.shape[0] == 4:
                return x + 1
            else:
                return x

        inp = (torch.rand(4, 5),)

        # Being able to export means shape is preserved as static
        export(branch_on_shape, inp)


@unittest.skipIf(not torchdynamo.is_dynamo_supported(), "dynamo isn't support")
class TestExport(TestCase):

    def _test_export_same_as_eager(self, f, args, kwargs=None):
        kwargs = kwargs or {}
        exported_program = export(f, args, kwargs)
        reversed_kwargs = {key: kwargs[key] for key in reversed(kwargs)}
        self.assertEqual(exported_program(*args, **kwargs), f(*args, **kwargs))
        self.assertEqual(exported_program(*args, **reversed_kwargs), f(*args, **reversed_kwargs))

    def test_basic(self):
        def f(x, y):
            return x[0] + y

        inp = ([torch.ones(1, 3)], torch.ones(1, 3))
        self._test_export_same_as_eager(f, inp)

    def test_raise_user_error_when_guard_on_data_dependent_operation(self):
        def fn_ddo(x):
            y = x.nonzero()
            z = y.shape[0]
            if z > 2:
                return x.cos()
            else:
                return x.sin()

        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "trying to get a value out of symbolic int"
        ):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),), constraints=None)

    def test_if_functional(self):
        def foo(x):
            z = x + 4
            z.add_(4)
            y = z.view(x.shape)
            return x.cos() + y.cos()

        gm = export(foo, (torch.tensor([2, 3, 5]),), constraints=None)

        view_count = 0
        for node in gm.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.add_.Tensor:
                # No more inplace mutation
                self.assertNotEqual(
                    node.target,
                    torch.ops.aten.add_.Tensor,
                    "There shouldn't be any inplace mutation node in the graph."
                )
            if node.op == "call_function" and node.target == torch.ops.aten.view.default:
                view_count += 1

        # There should be nonzero view nodes in the graph
        self.assertTrue(view_count > 0)

    def test_export_mod_constraints(self):
        class BasicDynamiShapeModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return x.view(x.shape[0] - 1, -1)

        m = BasicDynamiShapeModel()
        a = torch.randn(3, 4)
        constraints = [3 <= dynamic_dim(a, 0), dynamic_dim(a, 1)]
        with self.assertRaisesRegex(
            torch._dynamo.exc.UserError,
            (
                "Some dynamic dimensions need to be specialized because "
                "the constraints inferred for them are too complex to specify"
                ".*\n.*\\[0\\], which was marked dynamic, must be specialized to 3"
                ".*\n.*\\[1\\], which was marked dynamic, must be specialized to 4"
            ),
        ):
            torch._export.export(m, (a,), constraints=constraints)
        em = torch._export.export(m, (a,))
        x = torch.randn(3, 5)
        with self.assertRaisesRegex(RuntimeError, "\\[1\\] is specialized at 4"):
            em(x)

    def test_export_constrain_static(self):
        def f(x, y):
            b = x.item()
            constrain_as_size(b, min=2, max=5)
            c = y.dim()
            constrain_as_value(c, min=1, max=3)
            z = y[0:c]
            return torch.empty((b, y.shape[0])), z

        x = torch.tensor([3])
        y = torch.randn([8, 8, 6])
        example_inputs = (x, y)
        constraints = [dynamic_dim(y, 0) >= 6, dynamic_dim(y, 0) <= 10]
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError, "It appears that you're trying to set a constraint " +
            "on a value which we evaluated to have a static value of 3. "
        ):
            export(f, example_inputs, {}, constraints)

    def test_not_correct_dim(self):
        def f(x):
            return x.cos()

        def g(x):
            return x + 4

        inp_for_f = torch.tensor(5)
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Cannot mark 0-dimension tensors to be dynamic"):
            constraints = [dynamic_dim(inp_for_f, 0)]

        inp_for_f_mul_dim = torch.ones(5, 5)
        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            "Expected the dimension passed to dynamic_dim to be in the range \\[0:1\\]"
        ):
            constraints = [dynamic_dim(inp_for_f_mul_dim, 2)]

        inp_for_g = 4
        with self.assertRaisesRegex(torchdynamo.exc.UserError, "Expected tensor as input to dynamic_dim"):
            constraints = [dynamic_dim(inp_for_g, 0)]

    def test_map(self):
        def list_tensor_map(xs, y, z):
            def body(x, y, z):
                return x + y + z

            return map(body, xs, y, z)

        inps = (torch.ones(6, 4), torch.tensor(5), torch.tensor(4))
        self._test_export_same_as_eager(list_tensor_map, inps)

    def test_export_func_with_kwargs(self):
        def kw_func(arg1, arg2, kw1, kw2):
            return arg1 + arg2, kw1 + kw2

        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs = {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_export_func_with_pytree_kwargs(self):
        def kw_func(arg1, arg2, a, b):
            return arg1 + a["kw1"] + b[0], arg2 + a["kw2"] + b[1]

        args = (torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"a": {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}, "b": [torch.ones(2, 3), torch.ones(3, 4)]}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_export_func_with_default_kwargs(self):
        def kw_func(arg1, arg2, a, b=1):
            return arg1 + arg2, a["kw1"] + a["kw2"] + b

        def kw_func2(arg1, arg2, a=1, b=2):
            return arg1 + a, arg2 + b


        args = (torch.ones(6, 4), torch.ones(1, 1))
        kwargs1 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}}
        kwargs2 = {"a": {"kw1": torch.ones(1, 1), "kw2": torch.ones(6, 4)}, "b": 2}
        self._test_export_same_as_eager(kw_func, args, kwargs1)
        self._test_export_same_as_eager(kw_func, args, kwargs2)
        kwargs3 = {"b": 1}
        self._test_export_same_as_eager(kw_func2, args, kwargs3)

    def test_export_func_with_var_postional_args(self):
        def kw_func(arg1, arg2, *args):
            return arg1 + args[0], arg2 + args[1]

        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        self._test_export_same_as_eager(kw_func, args)

    def test_export_func_with_keyword_only_args(self):
        def kw_func(arg1, arg2, *args, kw1, kw2):
            return arg1 + args[0] + kw1, arg2 + args[1] + kw2

        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_export_func_with_var_keyword_args(self):
        def kw_func(arg1, arg2, *args, kw1, kw2, **kwargs):
            return arg1 + args[0] + kw1 + kwargs["kw3"], arg2 + args[1] + kw2 + kwargs["kw4"]

        args = (torch.ones(2, 3), torch.ones(3, 4), torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": torch.ones(2, 3), "kw2": torch.ones(3, 4), "kw3": torch.ones(2, 3), "kw4": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_export_func_with_var_keyword_pytree_args(self):
        def kw_func(arg1, arg2, *args, kw1, kw2, **kwargs):
            return arg1 + arg2[0][0] + args[0] + kw1[0] + kwargs["kw3"][0], arg2[1] + args[1] + kw2 + kwargs["kw4"]

        args = (torch.ones(2, 3), [(torch.ones(2, 3), ), torch.ones(3, 4)], torch.ones(2, 3), torch.ones(3, 4))
        kwargs = {"kw1": (torch.ones(2, 3), ), "kw2": torch.ones(3, 4),
                  "kw3": (torch.ones(2, 3), torch.ones(3, 4)), "kw4": torch.ones(3, 4)}
        self._test_export_same_as_eager(kw_func, args, kwargs)

    def test_linear_conv(self):

        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                x_conv = self.conv(x)
                x_linear = self.linear(x_conv)
                return x_linear.cos()

        ep = export(Foo(), (torch.randn(20, 16, 50, 100),))
        for node in ep.graph.nodes:
            if (
                node.op == "placeholder" and
                node.name in ep.graph_signature.inputs_to_buffers or
                node.name in ep.graph_signature.inputs_to_parameters
            ):
                self.assertTrue("source_fn" in node.meta)
                self.assertTrue("nn_module_stack" in node.meta)


    def test_error_does_not_reference_eager_fallback(self):
        def fn_ddo(x):
            y = x.nonzero()
            z = y.shape[0]
            if z > 2:
                return x.cos()
            else:
                return x.sin()

        with self.assertRaisesRegex(
            torchdynamo.exc.UserError,
            r"^(?!.*fall back to eager).*"
        ):
            _ = export(fn_ddo, (torch.tensor([2, 3, 5]),), constraints=None)

    def test_pytree_regster_data_class(self):

        @dataclass
        class MyDataClass:
            x: int
            y: int
            z: int = None

        dt = MyDataClass(x=3, y=4)
        flat, spec = tree_flatten(dt)
        self.assertTrue(spec, LeafSpec())
        self.assertTrue(len(flat) == 1)

        register_dataclass_as_pytree_node(MyDataClass)

        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyDataClass,
                (
                    MyDataClass,
                    ['x', 'y'],
                    ['z']
                ),
                [LeafSpec(), LeafSpec()]
            )
        )
        self.assertEqual(flat, [3, 4])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

        # Override the registration with keep none fields
        register_dataclass_as_pytree_node(MyDataClass, return_none_fields=True)

        flat, spec = tree_flatten(dt)
        self.assertEqual(
            spec,
            TreeSpec(
                MyDataClass,
                (
                    MyDataClass,
                    ['x', 'y', 'z'],
                    [],
                ),
                [LeafSpec(), LeafSpec(), LeafSpec()]
            )
        )
        self.assertEqual(flat, [3, 4, None])

        orig_dt = tree_unflatten(flat, spec)
        self.assertTrue(isinstance(orig_dt, MyDataClass))
        self.assertEqual(orig_dt.x, 3)
        self.assertEqual(orig_dt.y, 4)
        self.assertEqual(orig_dt.z, None)

    def test_pytree_regster_nested_data_class(self):

        @dataclass
        class Inner:
            x: int
            y: int

        @dataclass
        class Outer:
            xy: Inner
            ab: Inner

        xy = Inner(1, 2)
        ab = Inner(3, 4)
        dt = Outer(xy, ab)
        inp = {"dt1": (dt, ({},)), "dt2": ((torch.ones(1),), dt)}

        register_dataclass_as_pytree_node(Inner)
        register_dataclass_as_pytree_node(Outer)

        flat, spec = tree_flatten(inp)
        self.assertEqual(flat, [1, 2, 3, 4, torch.ones(1), 1, 2, 3, 4])

        unflat = tree_unflatten(flat, spec)
        self.assertEqual(unflat, inp)

    def test_export_dynamo_config(self):
        class MyModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size=4, hidden_size=5, num_layers=1)

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                return self.lstm(inputs)


        config = DEFAULT_EXPORT_DYNAMO_CONFIG
        mod = MyModule()

        @contextmanager
        def _patch_config(kwargs):
            orig_config_dict = dataclasses.asdict(config)

            try:
                for k, v in kwargs.items():
                    setattr(config, k, v)
                yield
            finally:
                for k, v in orig_config_dict.items():
                    setattr(config, k, v)

        inp = (torch.rand(5, 4), )
        exported_program = export(mod, inp)

        with _patch_config({"allow_rnn": False}):
            with self.assertRaisesRegex(
                torch._dynamo.exc.Unsupported,
                "TorchDynamo purposely graph breaks on RNN, GRU, LSTMs"
            ):
                _ = export(mod, inp)

    def test_module(self):

        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a, b = x
                a_conv = self.conv(a)
                a_linear = self.linear(a_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return (a_linear.cos() + b_linear.sin(), a_linear.sin() + b_linear.cos())

        inp_container = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = ((torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)),)

        self.assertTrue(torch.allclose(ep(*inp_test)[0], ep_rexported(*inp_test)[0]))
        self.assertTrue(torch.allclose(ep(*inp_test)[1], ep_rexported(*inp_test)[1]))

    def test_module_with_dict_container_inp_out(self):

        class MyLinear(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.randn(20, 98)
                self.bias = torch.randn(20)

            def forward(self, x):
                return torch.nn.functional.linear(x, self.weight, self.bias)

        class Foo(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3)
                self.linear = MyLinear()

            def forward(self, x):
                a1, a2 = x["a"]
                b = x["b"]
                a1_conv = self.conv(a1)
                a1_linear = self.linear(a1_conv)
                a2_conv = self.conv(a2)
                a2_linear = self.linear(a2_conv)
                b_conv = self.conv(b)
                b_linear = self.linear(b_conv)
                return {"a": a1_linear.cos() + b_linear.sin(), "b": a2_linear.sin() + b_linear.cos()}

        inp_container = ({"a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)), "b": torch.randn(20, 16, 50, 100)},)

        ep = export(Foo(), inp_container)
        ep_rexported = export(ep.module(), inp_container)

        inp_test = ({"a": (torch.randn(20, 16, 50, 100), torch.randn(20, 16, 50, 100)), "b": torch.randn(20, 16, 50, 100)},)

        self.assertTrue(torch.allclose(ep(*inp_test)["a"], ep_rexported(*inp_test)["a"]))
        self.assertTrue(torch.allclose(ep(*inp_test)["b"], ep_rexported(*inp_test)["b"]))

    def test_args_type_checked(self):
        def fn(x):
            return x + 1

        inp = torch.rand(2, 2)
        with self.assertRaisesRegex(torch._dynamo.exc.UserError, "to be a tuple"):
            # Intentionally not wrapping `inp` in a tuple to trigger the error
            _ = export(fn, inp)


if __name__ == '__main__':
    run_tests()
