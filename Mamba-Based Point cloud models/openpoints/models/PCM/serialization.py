"""
Point, z-order and h-order are copied form PointCept
https://github.com/Pointcept/Pointcept
"""
import torch
from .z_order import xyz2key as z_order_encode_
from .hilbert import encode as hilbert_encode_
from addict import Dict

class Point(Dict):
    """
    Point Structure of Pointcept

    A Point (point cloud) in Pointcept is a dictionary that contains various properties of
    a batched point cloud. The property with the following names have a specific definition
    as follows:

    - "coord": original coordinate of point cloud;
    - "grid_coord": grid coordinate for specific grid size (related to GridSampling);
    Point also support the following optional attributes:
    - "offset": if not exist, initialized as batch size is 1;
    - "batch": if not exist, initialized as batch size is 1;
    - "feat": feature of point cloud, default input of model;
    - "grid_size": Grid size of point cloud (related to GridSampling);
    (related to Serialization)
    - "serialized_depth": depth of serialization, 2 ** depth * grid_size describe the maximum of point cloud range;
    - "serialized_code": a list of serialization codes;
    - "serialized_order": a list of serialization order determined by code;
    - "serialized_inverse": a list of inverse mapping determined by code;
    (related to Sparsify: SpConv)
    - "sparse_shape": Sparse shape for Sparse Conv Tensor;
    - "sparse_conv_feat": SparseConvTensor init with information provide by Point;
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # If one of "offset" or "batch" do not exist, generate by the existing one
        if "batch" not in self.keys() and "offset" in self.keys():
            self["batch"] = offset2batch(self.offset)
        elif "offset" not in self.keys() and "batch" in self.keys():
            self["offset"] = batch2offset(self.batch)

    def serialization(self, order="z", depth=None, shuffle_orders=False):
        """
        Point Cloud Serialization

        relay on ["grid_coord" or "coord" + "grid_size", "batch", "feat"]
        """
        assert "batch" in self.keys()
        if "grid_coord" not in self.keys():
            # if you don't want to operate GridSampling in data augmentation,
            # please add the following augmentation into your pipline:
            # dict(type="Copy", keys_dict={"grid_size": 0.01}),
            # (adjust `grid_size` to what your want)
            assert {"grid_size", "coord"}.issubset(self.keys())
            self["grid_coord"] = torch.div(
                self.coord - self.coord.min(0)[0], self.grid_size, rounding_mode="trunc"
            ).int()

        #插入
        #思考-------------------------------------------------------------------------------------------------------------
        spherical_orders = {"rtp", "rpt", "trp", "tpr", "prt", "ptr"}
        cart_orders = {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
        hilbert_orders = {"z", "z-trans", "hilbert"}

        # if depth is None:
        #     # Adaptive measure the depth of serialization cube (length = 2 ^ depth)
        #     depth = int(self.grid_coord.max()).bit_length()

        if depth is None:
            # 对于 z/hilbert 分支，还是按原来的方式估计 depth
            if any(o in hilbert_orders for o in order):
                depth = int(self.grid_coord.max()).bit_length()
            else:
                # 如果完全不用 z/hilbert，给一个安全的缺省，比如 1
                depth = 1


        self["serialized_depth"] = depth
        # Maximum bit length for serialization code is 63 (int64)
        # assert depth * 3 + len(self.offset).bit_length() <= 63
        # Here we follow OCNN and set the depth limitation to 16 (48bit) for the point position.
        # Although depth is limited to less than 16, we can encode a 655.36^3 (2^16 * 0.01) meter^3
        # cube with a grid size of 0.01 meter. We consider it is enough for the current stage.
        # We can unlock the limitation by optimizing the z-order encoding function if necessary.

        # assert depth <= 16
        # 只有真的会走 z/hilbert 分支时才检查 bit 限制
        if any(o in hilbert_orders for o in order):
            assert depth * 3 + len(self.offset).bit_length() <= 63
            assert depth <= 16

        # The serialization codes are arranged as following structures:
        # [Order1 ([n]),
        #  Order2 ([n]),
        #   ...
        #  OrderN ([n])] (k, n)
        code = [
            encode(self.grid_coord, self.batch, depth, order=order_) for order_ in order
        ]
        #根据当前顺序生成一个长度为 n 的一维编码（n = 点数 * batch_size）。
        code = torch.stack(code)
        '''
        shape ≈ [k, n]：
        k：有多少种序列化顺序（orders 的长度）
        n：flatten 之后的点数（多 batch 合在一起）
        '''
        order = torch.argsort(code) #对每一行 code[i] 按升序排序，返回排序后的索引。
        #code[i] 是第 i 种顺序对应的编码。
        inverse = torch.zeros_like(order).scatter_(
            dim=1,
            index=order,
            src=torch.arange(0, code.shape[1], device=order.device).repeat(
                code.shape[0], 1
            ),
        )
        #打乱“不同遍历顺序（order）的排列顺序
        if shuffle_orders:
            perm = torch.randperm(code.shape[0])
            code = code[perm]
            order = order[perm]
            inverse = inverse[perm]

        self["serialized_code"] = code
        self["serialized_order"] = order
        self["serialized_inverse"] = inverse

@torch.inference_mode()
def offset2bincount(offset):
    return torch.diff(
        offset, prepend=torch.tensor([0], device=offset.device, dtype=torch.long)
    )

@torch.inference_mode()
def offset2batch(offset):
    bincount = offset2bincount(offset)
    return torch.arange(
        len(bincount), device=offset.device, dtype=torch.long
    ).repeat_interleave(bincount)

@torch.inference_mode()
def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

@torch.inference_mode()
def encode(grid_coord, batch=None, depth=16, order="z"):
    # 先解析是否带有模式后缀，比如 "rtp-snake"
    if "-" in order:
        base_order, pattern = order.split("-", 1)
    else:
        base_order, pattern = order, None

    if base_order in {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}:
        return encode_cts(grid_coord, batch, depth, order)
    # assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}

    # 2）极坐标：普通扫描 / 蛇形扫描
    spherical_orders = {"rtp", "rpt", "trp", "tpr", "prt", "ptr"}
    if base_order in spherical_orders:
        if pattern == "snake":
            return encode_spherical_snake(grid_coord, batch=batch, order=base_order)
        else:
            return encode_spherical(grid_coord, batch=batch, order=base_order)
    # 3）z-order / hilbert
    assert base_order in {"z", "z-trans", "hilbert", "hilbert-trans"}, \
        f"未知 order: {order}"

    if order == "z":
        code = z_order_encode(grid_coord, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], depth=depth)
    else:
        raise NotImplementedError
    if batch is not None:
        batch = batch.long()
        code = batch << depth * 3 | code
    return code

@torch.inference_mode()
def encode_cts(grid_coord, batch=None, depth=None, order="xyz"):
    assert order in {"xyz", "xzy", "yxz", "yzx", "zxy", "zyx"}
    index_ = []
    sub_order2index = {"x": 0, "y": 1, "z": 2}
    for sub_order in order:
        index_.append(sub_order2index[sub_order])

    grid_coord = grid_coord[:, index_]

    coords1 = grid_coord[:, 0]
    coords2 = grid_coord[:, 1]
    coords3 = grid_coord[:, 2]

    assert batch is not None

    max_coords1 = torch.max(coords1)
    max_coords2 = torch.max(coords2)
    max_coords3 = torch.max(coords3)

    low_code = coords1
    code = low_code
    max_base = max_coords1
    i = 0
    for new_code, max_new_code in zip([max_coords2, max_coords3, batch], [max_coords2, max_coords3, 1]):
        if i == 2:
            code = new_code * max_base + low_code
        else:
            sign = (new_code % 2 == 0).to(torch.float32) * 2 - 1    #模拟奇数行倒序、偶数行正序
            #偶数行从“左边”开始；奇数行相当于“从右边那一列”开始编码。
            new_code_ = (1 - (sign + 1) / 2) + new_code
            new_code_, sign = new_code_.to(torch.int64), sign.to(torch.int64)
            code = new_code_ * max_base + sign * low_code
        i += 1
        low_code = code
        max_base = max_base * max_new_code
    return code

def z_order_encode(grid_coord: torch.Tensor, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    code = z_order_encode_(x, y, z, b=None, depth=depth)
    return code


def hilbert_encode(grid_coord: torch.Tensor, depth: int = 16):
    return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)


@torch.inference_mode()
def encode_spherical(grid_coord, batch=None, order="rtp"):
    """
    通用极坐标编码：
    - grid_coord: [N, 3]，列为 [r_idx, theta_idx, phi_idx]（已经做过 min-shift）
    - batch: [N]，每个点的 batch 编号
    - order: 例如 "rtp", "rpt", "trp", "tpr", "prt", "ptr"
             三个字符必须正好是 {'r', 't', 'p'} 的一个排列
    扫描规则：
    - order[0] 作为最外层维度
    - order[1] 作为中间维度
    - order[2] 作为最内层维度
    - 在中间/内层用简单的“行优先”扫描（后面你想加蛇形也可以在这里改）
    """
    assert len(order) == 3 and set(order) == {"r", "t", "p"}, \
        f"encode_spherical 仅支持 'r','t','p' 的排列，收到 {order}"

    axis2idx = {"r": 0, "t": 1, "p": 2}
    # 按照 order 取出三维
    idx0 = axis2idx[order[0]]  # 最外层
    idx1 = axis2idx[order[1]]  # 中间层
    idx2 = axis2idx[order[2]]  # 最内层

    a = grid_coord[:, idx0].long()
    b = grid_coord[:, idx1].long()
    c = grid_coord[:, idx2].long()

    # 从数据中推断每个维度的 bins 数（因为前面做过 min-shift，所以从 0 开始）
    max_a = int(a.max().item()) + 1
    max_b = int(b.max().item()) + 1
    max_c = int(c.max().item()) + 1

    # 内层：b, c 的 2D 网格，行优先扫描
    inner_size = max_c                # c 的跨度
    code_inner = b * inner_size + c   # [N]

    # 外层：a
    base = max_b * inner_size         # 每一个 a 的“壳层”能容纳的格点数
    code = a * base + code_inner      # [N]

    # batch 编码，避免不同 batch 打乱
    if batch is not None:
        batch = batch.long()
        batch_base = max_a * base     # 每个 batch 自己的 code 范围大小
        code = batch * batch_base + code

    return code


@torch.inference_mode()
def encode_spherical_snake(grid_coord, batch=None, order="rtp"):
    """
    极坐标蛇形扫描版本：
    - grid_coord: [N, 3]，列为 [r_idx, theta_idx, phi_idx]（已做过 min-shift）
    - order: "rtp"/"rpt"/"trp"/"tpr"/"prt"/"ptr"
      表示哪一维是半径、哪一维是 θ / φ
    扫描逻辑（以 rtp 为例）：
    1）r 作为最外层 -> 壳层
    2）每个壳层内，把 (θ, φ) 看成 2D 网格：
        - φ 方向分层（纬向带），从北到南
        - 每一条 φ 带内部，对 θ 做蛇形（偶数 φ 正向，奇数 φ 反向）
    3）壳层之间做一次半径方向蛇形：
        - 偶数 r：壳层从小到大扫描
        - 奇数 r：壳层从大到小扫描（整体反转）
    """
    assert len(order) == 3 and set(order) == {"r", "t", "p"}, \
        f"encode_spherical_snake 仅支持 'r','t','p' 的排列，收到 {order}"

    axis2idx = {"r": 0, "t": 1, "p": 2}
    idx0 = axis2idx[order[0]]  # 视为“半径维”（壳层）
    idx1 = axis2idx[order[1]]  # 视为“纬向维”
    idx2 = axis2idx[order[2]]  # 视为“经向维”

    # 拆分三个维度
    a = grid_coord[:, idx0].long()  # 半径 r-like
    b = grid_coord[:, idx1].long()  # φ-like（纬度）
    c = grid_coord[:, idx2].long()  # θ-like（经度）

    # 推断 bins 数
    max_a = int(a.max().item()) + 1
    max_b = int(b.max().item()) + 1
    max_c = int(c.max().item()) + 1

    # ---------- 壳层内：2D 蛇形 (b, c) ----------
    # 行号 = b，每一条 b 带上对 c 做正反交替
    # 偶数 b: c 正向；奇数 b: c 反向
    even_row = (b % 2 == 0)
    c_snake = torch.where(even_row, c, (max_c - 1 - c))   # [N]
    code_shell = b * max_c + c_snake                      # [N]，0..(max_b*max_c-1)

    shell_size = max_b * max_c  # 每个壳层最多格点数

    # ---------- 壳层之间：半径方向蛇形 ----------
    # 偶数 a: 正向展开；奇数 a: 整个壳层反向
    # 这样全局 index 在 r 维度上也像“Z 字形”走
    even_shell = (a % 2 == 0)
    # 对于奇数壳层，把壳层内部顺序整体反转
    code_shell_snake = torch.where(
        even_shell,
        code_shell,
        (shell_size - 1 - code_shell)
    )

    # 全局 code（不含 batch）
    code = a * shell_size + code_shell_snake  # [N]

    # ---------- batch 编码 ----------
    if batch is not None:
        batch = batch.long()
        batch_base = max_a * shell_size
        code = batch * batch_base + code

    return code

