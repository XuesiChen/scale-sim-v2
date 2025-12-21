# scalesim/memory/liveness_estimator.py
from dataclasses import dataclass

@dataclass
class ConvShape:
    C: int # input channels
    OH: int # output height
    OW: int # output width
    OC: int # output channels
    KH: int # kernel height
    KW: int # kernel width
    stride: int # stride

@dataclass
class Tile:
    T_oh: int # output height tile size
    T_ow: int # output width tile size
    T_oc: int # output channels tile size

# the function estimate the amount of bytes needed to store in SRAM
# to avoid extra DRAM accesses for a given convolutional layer shape and output tile
def estimate_live_bytes_conv(shape: ConvShape, tile: Tile,
                             act_bytes: int = 1,
                             psum_bytes: int = 4):
    # IFMAP footprint needed for the output tile
    T_ih = (tile.T_oh - 1) * shape.stride + shape.KH
    T_iw = (tile.T_ow - 1) * shape.stride + shape.KW
    B_I = T_ih * T_iw * shape.C * act_bytes # live bytes for IFMAP

    # PSUM footprint for the output tile
    B_O = tile.T_oh * tile.T_ow * tile.T_oc * psum_bytes # live bytes for OFMAP/PSUM
    return int(B_I), int(B_O)

def choose_unified_io_split(unified_bytes: int, B_I: int, B_O: int,
                            lambda_psum: float = 4.0,
                            min_ifmap_frac: float = 0.10):
    """
    Returns (S_ifmap, S_ofmap) s.t. S_if + S_of = unified_bytes.
    lambda_psum>1 biases toward psum residency (good under DRAM BW limit).
    """
    S = unified_bytes
    S_min_if = int(S * min_ifmap_frac)

    denom = (B_I + lambda_psum * B_O)
    if denom <= 0:
        S_if = max(S_min_if, S // 2)
        return S_if, S - S_if

    alpha = B_I / denom
    S_if = int(alpha * S)
    S_if = max(S_min_if, min(S_if, S - 1))
    return S_if, S - S_if
