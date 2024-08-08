# mypy: allow-untyped-defs
import logging
from typing import List

import torch
from torch import Tensor
from torch._dynamo.utils import counters

from .. import config

from ..pattern_matcher import Arg, CallFunction, Match, register_graph_pattern
from .split_cat import construct_pattern_matcher_pass

aten = torch.ops.aten
log = logging.getLogger(__name__)


# def check_device(a: Tensor, b: Tensor) -> bool:
#     return a.is_cuda and b.is_cuda

def check_device(input: Tensor, indices: Tensor, values: Tensor) -> bool:
    return input.is_xpu and indices.is_xpu and values.is_xpu

def check_dtype(a: Tensor) -> bool:
    return a.dtype == torch.float16 or a.dtype == torch.bfloat16

def print_atomic_cast_pattern(match: Match, inputs: List[torch.fx.Node]):
    node = match.nodes[-1]
    log.debug(
        "atomic cast %s with input shape: %s",
        node.target,
        ", ".join(
            str(input.meta["val"].shape) if "val" in input.meta else "None"
            for input in inputs
        ),
    )

def should_insert_cast(input, indices, values) -> bool:
    input_meta = input.meta.get("val")
    values_meta = values.meta.get("val")
    if input_meta is None or values_meta is None:
        return False
    if input_meta.device.type != "xpu" or values_meta.device.type != "xpu":
        return False
    if input_meta.dtype == torch.bfloat16 or input_meta.dtype == torch.float16:
        return True
    else:
        return False

@register_graph_pattern(
    CallFunction(aten.index_put.default, Arg(), Arg(), Arg(), Arg()),
    pass_dict=construct_pattern_matcher_pass("atomic_cast_pass"),
)
def atomic_cast_index_put(
    match: Match,
    input: torch.fx.Node,
    indices: torch.fx.immutable_collections.immutable_list,
    values: torch.fx.Node,
    accum: bool,
):
    breakpoint()
    def repl(input, indices, values, accum):
        return aten.index_put(input.to(torch.float32), indices, values, accum).to(input.dtype)
    # print(type(input), type(indices), type(values),type(accum))

    if should_insert_cast(input, indices, values):
        counters["inductor"]["atomic_cast_index_put"] += 1
        match.replace_by_example(repl, [input, indices, values, accum])
        # print_atomic_cast_pattern(match, [input, indices, values])