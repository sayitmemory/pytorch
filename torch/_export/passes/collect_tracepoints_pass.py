import operator

import torch

from torch.ao.quantization.fx._decomposed import quantized_decomposed_lib  # noqa: F401
from torch.fx.passes.infra.pass_base import PassBase, PassResult

__all__ = ["CollectTracepointsPass"]


class CollectTracepointsPass(PassBase):
    """
    Performs constant folding and constant propagation.
    """

    def __init__(self, specs) -> None:
        super().__init__()
        self.specs = specs

    def call(self, gm):
        for module in gm.modules():
            if not isinstance(module, torch.fx.GraphModule):
                continue
            for node in module.graph.nodes:
                if node.op != "call_function":
                    continue
                if node.target == torch.ops.higher_order.export_tracepoint:
                    for i, arg in enumerate(node.args):
                        kind = node.kwargs["kind"]
                        if kind == "module_call_in":
                            self.specs[node.kwargs['path']].inputs.append(arg.name)
                        elif kind == "module_call_out":
                            self.specs[node.kwargs['path']].outputs.append(arg.name)
                        else:
                            assert False
                        for user in node.users:
                            assert user.op == "call_function"
                            assert user.target == operator.getitem
                            assert isinstance(user.args[1], int)
                            if user.args[1] == i:
                                break
                        else:
                            assert False
                        user.replace_all_uses_with(arg)
                    users = list(node.users)
                    for user in users:
                        assert len(user.users) == 0
                        gm.graph.erase_node(user)
                    gm.graph.erase_node(node)
            return PassResult(gm, True)
