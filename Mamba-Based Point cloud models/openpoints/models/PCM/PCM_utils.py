import torch
import torch.nn as nn
import torch.nn.functional as F
from .serialization import Point
import math

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, output_dim, hidden_dim=2048, num_layers=3, bias=False):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def serialization(pos, feat, x_res=None, order="z", layers_outputs=None, grid_size=0.02):
    if layers_outputs is None:
        layers_outputs = []

    bs, n_p, _ = pos.size()

    # ===================== 1. random 序列化：所有点随机排序 =====================
    if order == "random":
        device = pos.device

        # 为每个 batch 单独生成随机排列
        rand_order = torch.stack(
            [torch.randperm(n_p, device=device) for _ in range(bs)],
            dim=0
        )  # [B, N]

        batch_offsets = (torch.arange(bs, device=device).unsqueeze(1) * n_p)  # [B, 1]
        flat_order = (rand_order + batch_offsets).reshape(-1)  # [B*N]

        pos = pos.flatten(0, 1)[flat_order].reshape(bs, n_p, -1).contiguous()
        feat = feat.flatten(0, 1)[flat_order].reshape(bs, n_p, -1).contiguous()

        if x_res is not None:
            x_res = x_res.flatten(0, 1)[flat_order].reshape(bs, n_p, -1).contiguous()

        for i in range(len(layers_outputs)):
            layers_outputs[i] = (
                layers_outputs[i]
                .flatten(0, 1)[flat_order]
                .reshape(bs, n_p, -1)
                .contiguous()
            )

        return pos, feat, x_res

    # ===================== 2. 其余普通序列化保持原逻辑 =====================
    if not isinstance(order, list):
        order = [order]

    # 把坐标体素化到网格
    scaled_coord = pos / grid_size
    grid_coord = torch.floor(scaled_coord).to(torch.int64)
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]
    grid_coord = grid_coord - min_coord

    # 每个点属于哪个 batch
    batch_idx = (
        torch.arange(0, pos.shape[0], 1.0)
        .unsqueeze(1)
        .repeat(1, pos.shape[1])
        .to(torch.int64)
        .to(pos.device)
    )

    point_dict = {
        'batch': batch_idx.flatten(),
        'grid_coord': grid_coord.flatten(0, 1),
    }
    point_dict = Point(**point_dict)

    # 计算该顺序下的排列索引与逆索引
    point_dict.serialization(order=order)

    # 根据排序索引重排所有张量
    order = point_dict.serialized_order

    pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
    feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = (
            layers_outputs[i]
            .flatten(0, 1)[order]
            .reshape(bs, n_p, -1)
            .contiguous()
        )

    return pos, feat, x_res
#笛卡尔坐标
# #把点云排成 Mamba 需要的 1D 序列
# def serialization(pos, feat, x_res=None, order="z", layers_outputs=[], grid_size=0.02):
#     bs, n_p, _ = pos.size()
#     if not isinstance(order, list):
#         order = [order]
# 
#     #把坐标体素化到网格
#     scaled_coord = pos / grid_size
#     grid_coord = torch.floor(scaled_coord).to(torch.int64)  #得到整数网格坐标 (x_bin, y_bin, z_bin)
#     min_coord = grid_coord.min(dim=1, keepdim=True)[0]
#     grid_coord = grid_coord - min_coord
# 
#     #每个点属于哪个 batch
#     batch_idx = torch.arange(0, pos.shape[0], 1.0).unsqueeze(1).repeat(1, pos.shape[1]).to(torch.int64).to(pos.device)
# 
#     point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1), }
#     point_dict = Point(**point_dict)
#     #计算该顺序下的排列索引与逆索引
#     point_dict.serialization(order=order)
#     #根据排序索引重排所有张量
#     order = point_dict.serialized_order
#     inverse_order = point_dict.serialized_inverse
# 
#     pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
#     feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
#     if x_res is not None:
#         x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
# 
#     for i in range(len(layers_outputs)):
#         layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
#     return pos, feat, x_res




# 把点云排成 Mamba 需要的 1D 序列（3D 球坐标序列化，优化版）
def serialization_spherical(
        pos,
        feat,
        x_res=None,
        order="z",
        layers_outputs=None,     # ★ 修改 3：默认改为 None
        grid_size=0.02,         # 目前不直接用到，只保留接口
        n_r_bins=32,            # 半径壳层数（可在 config 里调）
        n_theta_bins=64,        # 水平角 bins
        n_phi_bins=32           # 垂直角 bins
):
    """
    使用球坐标 (r, theta, phi) 做网格化，再交给 Point.serialization 按 order 排序。

    - 半径 r：在每个 batch 内归一化到 [0,1]，再等间距量化到 n_r_bins 个壳层；
    - theta：[-pi, pi] → [0, 2pi)，再等间距量化到 n_theta_bins；
    - phi：[0, pi] → 等间距量化到 n_phi_bins。
    """
    if layers_outputs is None:             # ★ 修改 4：安全处理默认列表
        layers_outputs = []

    bs, n_p, _ = pos.size()
    if not isinstance(order, list):
        order = [order]

    # =====================【1. 坐标中心 & 笛卡尔 → 球坐标】=====================
    # 1) 以每个 batch 的几何中心为原点（均值中心比 bbox 更稳）
    center = pos.mean(dim=1, keepdim=True)        # [B, 1, 3]
    pos_centered = pos - center                   # [B, N, 3]

    # 2) 笛卡尔 → 球坐标
    x = pos_centered[..., 0]                      # [B, N]
    y = pos_centered[..., 1]
    z = pos_centered[..., 2]

    eps = 1e-8
    # 半径 rho
    rho = torch.sqrt(x * x + y * y + z * z + eps)   # [B, N]
    # 方位角 theta ∈ (-pi, pi]，绕 z 轴
    theta = torch.atan2(y, x)                       # [B, N]
    # 极角 phi ∈ [0, pi]，与 z 轴夹角
    cos_phi = torch.clamp(z / (rho + eps), -1.0, 1.0)
    phi = torch.acos(cos_phi)                       # [B, N]

    # =====================【2. r / θ / φ 量化到整数索引】=====================

    # 2.1 半径：每个 batch 内归一化到 [0,1]，再等间距切成 n_r_bins 个壳层
    max_r = rho.max(dim=1, keepdim=True)[0]         # [B,1]
    # 防止整个 batch 都在原点附近导致除以 ~0
    max_r = torch.where(max_r < 1e-3, torch.full_like(max_r, 1e-3), max_r)
    r_norm = rho / (max_r + eps)                    # [B, N] in [0,1]

    # 注意乘以 (n_r_bins-1) 而不是 n_r_bins，避免边界落到 n_r_bins
    r_idx = torch.floor(r_norm * (n_r_bins - 1e-6))
    r_idx = torch.clamp(r_idx, 0, n_r_bins - 1)     # [B, N]

    # 2.2 theta：[-pi, pi] → [0, 2pi)，再按等间距分成 n_theta_bins
    theta_shift = theta + math.pi                   # [0, 2pi)
    theta_step = 2 * math.pi / n_theta_bins
    theta_idx = torch.floor(theta_shift / theta_step)
    theta_idx = torch.clamp(theta_idx, 0, n_theta_bins - 1)   # [B, N]

    # 2.3 phi：[0, pi] → 等间距分成 n_phi_bins
    phi_step = math.pi / n_phi_bins
    phi_idx = torch.floor(phi / phi_step)
    phi_idx = torch.clamp(phi_idx, 0, n_phi_bins - 1)         # [B, N]

    # 2.4 组合成球坐标网格：
    #      grid_coord[b, n] = (r_bin, theta_bin, phi_bin)
    grid_coord = torch.stack(
        [r_idx, theta_idx, phi_idx], dim=-1
    ).to(torch.int64)                              # [B, N, 3]

    # 2.5 每个 batch 内做 min-shift，保证从 0 开始，范围更紧凑
    min_coord = grid_coord.min(dim=1, keepdim=True)[0]        # [B, 1, 3]
    grid_coord = grid_coord - min_coord                        # [B, N, 3]

    # =====================【3. 构造 Point，调用 Point.serialization】=====================

    # 每个点属于哪个 batch
    batch_idx = (
        torch.arange(0, pos.shape[0], 1.0, device=pos.device)
        .unsqueeze(1)
        .repeat(1, pos.shape[1])
        .to(torch.int64)
    )  # [B, N]

    point_dict = {
        "batch": batch_idx.flatten(),                 # [B*N]
        "grid_coord": grid_coord.flatten(0, 1),       # [B*N, 3]
    }
    point_dict = Point(**point_dict)

    # 根据给定 order（可能是 "rtp[-snake]" 等）计算排序索引
    point_dict.serialization(order=order)

    # =====================【4. 按序列顺序重排所有张量】=====================
    order_idx = point_dict.serialized_order          # [B*N]
    # inverse_order = point_dict.serialized_inverse  # 目前没用到，可以保留

    pos = pos.flatten(0, 1)[order_idx].reshape(bs, n_p, -1).contiguous()
    feat = feat.flatten(0, 1)[order_idx].reshape(bs, n_p, -1).contiguous()
    if x_res is not None:
        x_res = x_res.flatten(0, 1)[order_idx].reshape(bs, n_p, -1).contiguous()

    for i in range(len(layers_outputs)):
        layers_outputs[i] = (
            layers_outputs[i]
            .flatten(0, 1)[order_idx]
            .reshape(bs, n_p, -1)
            .contiguous()
        )

    return pos, feat, x_res







# # 把点云排成 Mamba 需要的 1D 序列（3D 球坐标序列化，优化版）
# def serialization_spherical(
#         pos,
#         feat,
#         x_res=None,
#         order="z",
#         layers_outputs=None,
#         grid_size=0.02,      # 只是保留接口，当前半径不再直接用它
#         n_r_bins=32,         # 半径壳层数（可在 config 里调）
#         n_theta_bins=64,     # 水平角 bins
#         n_phi_bins=32        # 垂直角 bins
# ):
#     """
#     使用球坐标 (r, theta, phi) 做网格化，再交给 Point.serialization 按 order 排序。
#
#     - 半径 r：在每个 batch 内归一化到 [0,1]，再等间距量化到 n_r_bins 个壳层；
#     - theta：[-pi, pi] → [0, 2pi)，再等间距量化到 n_theta_bins；
#     - phi：[0, pi] → 等间距量化到 n_phi_bins。
#     """
#     if layers_outputs is None:
#         layers_outputs = []
#
#     bs, n_p, _ = pos.size()
#     if not isinstance(order, list):
#         order = [order]
#
#     # =====================【1. 坐标中心 & 笛卡尔 → 球坐标】=====================
#     # 1) 以每个 batch 的几何中心为原点（均值中心比 bbox 更稳）
#     center = pos.mean(dim=1, keepdim=True)        # [B, 1, 3]
#     pos_centered = pos - center                   # [B, N, 3]
#
#     # 2) 笛卡尔 → 球坐标
#     x = pos_centered[..., 0]                      # [B, N]
#     y = pos_centered[..., 1]
#     z = pos_centered[..., 2]
#
#     eps = 1e-8
#     # 半径 rho
#     rho = torch.sqrt(x * x + y * y + z * z + eps)   # [B, N]
#     # 方位角 theta ∈ (-pi, pi]，绕 z 轴
#     theta = torch.atan2(y, x)                       # [B, N]
#     # 极角 phi ∈ [0, pi]，与 z 轴夹角
#     cos_phi = torch.clamp(z / (rho + eps), -1.0, 1.0)
#     phi = torch.acos(cos_phi)                       # [B, N]
#
#     # =====================【2. r / θ / φ 量化到整数索引】=====================
#
#     # 2.1 半径：每个 batch 内归一化到 [0,1]，再等间距切成 n_r_bins 个壳层
#     # max_r: [B,1]
#     max_r = rho.max(dim=1, keepdim=True)[0]
#     # 防止整个 batch 都在原点附近导致除以 ~0
#     max_r = torch.where(max_r < 1e-3, torch.full_like(max_r, 1e-3), max_r)
#     r_norm = rho / (max_r + eps)                  # [B, N] in [0,1]
#
#     # 注意乘以 (n_r_bins-1) 而不是 n_r_bins，避免边界落到 n_r_bins
#     r_idx = torch.floor(r_norm * (n_r_bins - 1e-6))
#     r_idx = torch.clamp(r_idx, 0, n_r_bins - 1)   # [B, N]
#
#     # 2.2 theta：[-pi, pi] → [0, 2pi)，再按等间距分成 n_theta_bins
#     theta_shift = theta + math.pi                 # [0, 2pi)
#     theta_step = 2 * math.pi / n_theta_bins
#     theta_idx = torch.floor(theta_shift / theta_step)
#     theta_idx = torch.clamp(theta_idx, 0, n_theta_bins - 1)   # [B, N]
#
#     # 2.3 phi：[0, pi] → 等间距分成 n_phi_bins
#     phi_step = math.pi / n_phi_bins
#     phi_idx = torch.floor(phi / phi_step)
#     phi_idx = torch.clamp(phi_idx, 0, n_phi_bins - 1)         # [B, N]
#
#     # 2.4 组合成球坐标网格：
#     #      grid_coord[b, n] = (r_bin, theta_bin, phi_bin)
#     grid_coord = torch.stack(
#         [r_idx, theta_idx, phi_idx], dim=-1
#     ).to(torch.int64)                              # [B, N, 3]
#
#     # 2.5 每个 batch 内做 min-shift，保证从 0 开始，范围更紧凑
#     min_coord = grid_coord.min(dim=1, keepdim=True)[0]        # [B, 1, 3]
#     grid_coord = grid_coord - min_coord                        # [B, N, 3]
#
#     # =====================【3. 构造 Point，调用 Point.serialization】=====================
#
#     # 每个点属于哪个 batch
#     batch_idx = (
#         torch.arange(0, pos.shape[0], 1.0, device=pos.device)
#         .unsqueeze(1)
#         .repeat(1, pos.shape[1])
#         .to(torch.int64)
#     )  # [B, N]
#
#     point_dict = {
#         "batch": batch_idx.flatten(),                 # [B*N]
#         "grid_coord": grid_coord.flatten(0, 1),       # [B*N, 3]
#     }
#     point_dict = Point(**point_dict)
#
#     # 根据给定 order（可能是 "rtp[-snake]" 等）计算排序索引
#     point_dict.serialization(order=order)
#
#     # =====================【4. 按序列顺序重排所有张量】=====================
#     order_idx = point_dict.serialized_order          # [B*N]
#     # inverse_order = point_dict.serialized_inverse  # 目前没用到，可以保留
#
#     pos = pos.flatten(0, 1)[order_idx].reshape(bs, n_p, -1).contiguous()
#     feat = feat.flatten(0, 1)[order_idx].reshape(bs, n_p, -1).contiguous()
#     if x_res is not None:
#         x_res = x_res.flatten(0, 1)[order_idx].reshape(bs, n_p, -1).contiguous()
#
#     for i in range(len(layers_outputs)):
#         layers_outputs[i] = (
#             layers_outputs[i]
#             .flatten(0, 1)[order_idx]
#             .reshape(bs, n_p, -1)
#             .contiguous()
#         )
#
#     return pos, feat, x_res



# 把点云排成 Mamba 需要的 1D 序列（改为 3D 球坐标序列化）
# def serialization_spherical(pos, feat, x_res=None, order="z", layers_outputs=None, grid_size=0.02,n_r_bins=32, n_theta_bins=64, n_phi_bins=32):
#     if layers_outputs is None:
#         layers_outputs = []
#     bs, n_p, _ = pos.size()
#     if not isinstance(order, list):
#         order = [order]
#
#     # =====================【修改开始：笛卡尔 → 球坐标】=====================
#     # 1) 以每个 batch 的块中心为球坐标原点
#     #    center: [B, 1, 3]
#     # center = (pos.min(dim=1, keepdim=True)[0] + pos.max(dim=1, keepdim=True)[0]) / 2.0
#     center = pos.mean(dim=1, keepdim=True)  # 均值中心，比 bbox 中心更稳定
#     pos_centered = pos - center  # [B, N, 3]
#
#     # 2) 笛卡尔坐标 → 球坐标 (rho, theta, phi)
#     x = pos_centered[..., 0]  # [B, N]
#     y = pos_centered[..., 1]
#     z = pos_centered[..., 2]
#
#     eps = 1e-8
#     # 半径 rho
#     rho = torch.sqrt(x * x + y * y + z * z + eps)  # [B, N]
#     # 方位角 theta ∈ (-pi, pi]，绕 z 轴
#     theta = torch.atan2(y, x)                     # [B, N]
#     # 极角 phi ∈ [0, pi]，与 z 轴夹角
#     cos_phi = torch.clamp(z / (rho + eps), -1.0, 1.0)
#     phi = torch.acos(cos_phi)                     # [B, N]
#
#     # 3) 球坐标量化到整数网格索引
#     #这里简单复用 grid_size 作为半径和角度的步长（你后面可以单独拆参数）
#     #改---------------------------------------------------------------------
#     # r_step = grid_size
#     # angle_step = grid_size
#     #
#     # # [B, N] → 整数 index
#     # r_idx = torch.floor(rho / r_step)   #把半径切为：第 0 层，第 1 层，第 2 层
#     # # “把角度空间均匀划分成多个扇区，点位于哪个水平角度扇区”                  # [0, pi]
#     # theta_idx = torch.floor((theta + math.pi) / angle_step)  # 把 theta 从 [-pi,pi] 映射到 [0, 2pi)
#     #
#     # phi_idx = torch.floor(phi / angle_step)
#
#     # 3) 球坐标量化到整数网格索引
#     # 半径还是用原来的 grid_size，当作“壳层厚度”
#     # r_step = grid_size
#
#     # 角度用“固定 bin 数”，和 grid_size 解耦，便于单独调节
#     # n_theta_bins = 64   # 水平角 bins 数
#     # n_phi_bins   = 32   # 垂直角 bins 数
#
#     max_r = rho.max(dim=1, keepdim=True)[0]  # [B,1]
#     r_norm = rho / (max_r + eps)
#     n_r_bins = R  # 新超参，例如 32 或 64
#     r_idx = torch.floor(r_norm * n_r_bins)
#     r_idx = torch.clamp(r_idx, 0, n_r_bins - 1)
#
#     theta_step = 2 * math.pi / n_theta_bins  # [-pi,pi] → [0,2pi)，再除以步长
#     phi_step   = math.pi / n_phi_bins        # [0,pi]
#
#     # [B, N] → 整数 index
#     # 半径：第几个壳层
#     r_idx = torch.floor(rho / r_step)
#     # 水平角：第几个扇区
#     theta_idx = torch.floor((theta + math.pi) / theta_step)
#     # 垂直角：第几条纬向条带
#     phi_idx = torch.floor(phi / phi_step)
#
#     # 可选：把极少数数值溢出的点 clamp 回合法范围
#     theta_idx = torch.clamp(theta_idx, 0, n_theta_bins - 1)
#     phi_idx = torch.clamp(phi_idx, 0, n_phi_bins - 1)
#
#     # 组成 "球坐标网格"： [rho_bin, theta_bin, phi_bin]
#     # 4) 构造球坐标网格（用于序列化排序）
#     grid_coord = torch.stack([r_idx, theta_idx, phi_idx], dim=-1).to(torch.int64)  # [B, N, 3]
#
#     # 平移到从 0 开始，保证非负、范围更紧凑
#     min_coord = grid_coord.min(dim=1, keepdim=True)[0]
#     grid_coord = grid_coord - min_coord
#     # =====================【修改结束：笛卡尔 → 球坐标】=====================
#
#     # 每个点属于哪个 batch
#     batch_idx = (
#         torch.arange(0, pos.shape[0], 1.0, device=pos.device)
#         .unsqueeze(1)
#         .repeat(1, pos.shape[1])
#         .to(torch.int64)
#     )   #[B, G]
#     #grid_coord = 点在“三维整数网格”中的编号；grid_coord = (r_idx, θ_idx, φ_idx)
#     # 构造 Point，并调用 Point.serialization（里面还是原来的 CTS / z / hilbert）:contentReference[oaicite:1]{index=1}
#     point_dict = {'batch': batch_idx.flatten(), 'grid_coord': grid_coord.flatten(0, 1)}
#     point_dict = Point(**point_dict)
#
#     # 计算该顺序下的排列索引与逆索引
#     point_dict.serialization(order=order)
#
#     # 根据排序索引重排所有张量
#     order = point_dict.serialized_order
#     inverse_order = point_dict.serialized_inverse  # 目前没用到，但可以保留
#
#     #重排 pos（坐标）
#     pos = pos.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
#     feat = feat.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
#     if x_res is not None:
#         x_res = x_res.flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
#
#     for i in range(len(layers_outputs)):
#         layers_outputs[i] = layers_outputs[i].flatten(0, 1)[order].reshape(bs, n_p, -1).contiguous()
#
#     return pos, feat, x_res



# def square_distance(src, dst):
#     """
#     Calculate Euclid distance between each two points.
#     src^T * dst = xn * xm + yn * ym + zn * zm；
#     sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
#     sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
#     dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
#          = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
#     Input:
#         src: source points, [B, N, C]
#         dst: target points, [B, M, C]
#     Output:
#         dist: per-point square distance, [B, N, M]
#     """
#     B, N, _ = src.shape
#     _, M, _ = dst.shape
#     #dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
#     #dist += torch.sum(src ** 2, -1).view(B, N, 1)
#     #dist += torch.sum(dst ** 2, -1).view(B, 1, M)
#     src_ = src.unsqueeze(2)
#     dst_ = dst.unsqueeze(1)
#     dist = (src_ - dst_) ** 2
#     dist = dist.sum(dim=-1)
#     return dist

def square_distance(src, dst):
    """
    Calculate squared Euclidean distance between each two points.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point squared distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape

    dist = -2 * torch.matmul(src, dst.transpose(1, 2))
    dist += torch.sum(src ** 2, dim=-1).view(B, N, 1)
    dist += torch.sum(dst ** 2, dim=-1).view(B, 1, M)

    # Numerical stability: avoid tiny negative values caused by fp error
    dist = torch.clamp(dist, min=0.0)
    return dist

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def knn_point(nsample, xyz, new_xyz, training=True):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    if training:
        sqrdists = square_distance(new_xyz, xyz)
        _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
    else:
        N = new_xyz.shape[1]
        idx_list = []
        n_splits = N // 1024
        if n_splits * 1024 != N:
            n_splits += 1
        start, end = 0, 1024
        for i in range(n_splits):
            end = min(end, N)
            sqrdists = square_distance(new_xyz[:, start: end], xyz)
            _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=True)
            idx_list.append(group_idx)
            start += 1024
            end += 1024
        group_idx = torch.cat(idx_list, dim=1)
    return group_idx

# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
        module,
        n_layer,
        initializer_range=0.02,  # Now only used for embedding layer.
        rescale_prenorm_residual=True,
        n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


