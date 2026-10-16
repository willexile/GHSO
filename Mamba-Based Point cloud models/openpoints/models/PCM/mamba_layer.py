import math
import os
import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.selective_scan_interface import (
        selective_scan_fn,
        mamba_inner_fn,
        bimamba_inner_fn,
        mamba_inner_fn_no_out_proj,
    )
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
from timm.models.layers import DropPath


def _split_mamba_debug_return(ret):
    """
    兼容两种返回形式：
    1) out
    2) (out, params_for_debug)
    """
    if isinstance(ret, tuple):
        if len(ret) >= 2:
            return ret[0], ret[1]
        return ret[0], None
    return ret, None


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    return obj


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="none",
        norm_fn=nn.LayerNorm,
        save_hidden_attn=False,
        hidden_attn_save_dir="vis_attn",
        hidden_attn_target_layer=None,
        hidden_attn_target_layers=None,
        hidden_attn_target_len=None,
        hidden_attn_save_once=True,
        # save_hidden_attn=False,
        # hidden_attn_save_dir="vis_attn",
        # hidden_attn_target_layer=None,
        # hidden_attn_target_len=None,
        # hidden_attn_save_once=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type

        # hidden attention 保存控制
        self.save_hidden_attn = save_hidden_attn
        self.hidden_attn_save_dir = hidden_attn_save_dir
        self.hidden_attn_target_layer = hidden_attn_target_layer
        self.hidden_attn_target_layers = None if hidden_attn_target_layers is None else set(hidden_attn_target_layers)
        self.hidden_attn_target_len = hidden_attn_target_len
        self.hidden_attn_save_once = hidden_attn_save_once
        self.hidden_attn_saved = False
        self.last_params_for_debug = None
        # self.save_hidden_attn = save_hidden_attn
        # self.hidden_attn_save_dir = hidden_attn_save_dir
        # self.hidden_attn_target_layer = hidden_attn_target_layer
        # self.hidden_attn_target_len = hidden_attn_target_len
        # self.hidden_attn_save_once = hidden_attn_save_once
        # self.hidden_attn_saved = False
        # self.last_params_for_debug = None

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))
        self.D._no_weight_decay = True

        if self.bimamba_type in ["v2", "v3"]:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        if self.bimamba_type == "v3":
            self.norm = norm_fn(self.d_inner)

    def _maybe_save_hidden_attn(self, params_for_debug, seqlen):
        self.last_params_for_debug = params_for_debug

        if params_for_debug is None:
            return
        if not self.save_hidden_attn:
            return

        # 支持单个目标 layer 或多个目标 layers
        if self.hidden_attn_target_layers is not None:
            if self.layer_idx not in self.hidden_attn_target_layers:
                return
        elif self.hidden_attn_target_layer is not None:
            if self.layer_idx != self.hidden_attn_target_layer:
                return

        if self.hidden_attn_target_len is not None and seqlen != self.hidden_attn_target_len:
            return
        if self.hidden_attn_save_once and self.hidden_attn_saved:
            return

        os.makedirs(self.hidden_attn_save_dir, exist_ok=True)
        save_path = os.path.join(
            self.hidden_attn_save_dir,
            f"params_layer_{self.layer_idx}_L_{seqlen}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt",
        )
        torch.save(_detach_to_cpu(params_for_debug), save_path)
        self.hidden_attn_saved = True
        print(f"[HiddenAttn] saved params_for_debug to: {save_path}")

    # def _maybe_save_hidden_attn(self, params_for_debug, seqlen):
    #     self.last_params_for_debug = params_for_debug
    #
    #     if params_for_debug is None:
    #         return
    #     if not self.save_hidden_attn:
    #         return
    #     if self.hidden_attn_target_layer is not None and self.layer_idx != self.hidden_attn_target_layer:
    #         return
    #     if self.hidden_attn_target_len is not None and seqlen != self.hidden_attn_target_len:
    #         return
    #     if self.hidden_attn_save_once and self.hidden_attn_saved:
    #         return
    #
    #     os.makedirs(self.hidden_attn_save_dir, exist_ok=True)
    #     save_path = os.path.join(
    #         self.hidden_attn_save_dir,
    #         f"params_layer_{self.layer_idx}_L_{seqlen}_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.pt",
    #     )
    #     torch.save(_detach_to_cpu(params_for_debug), save_path)
    #     self.hidden_attn_saved = True
    #     print(f"[HiddenAttn] saved params_for_debug to: {save_path}")

    def forward(self, hidden_states, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        if self.bimamba_type == "v3":
            hidden_states = torch.cat([hidden_states, hidden_states.flip([-2])], dim=0)

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())

        if self.use_fast_path and inference_params is None:
            if self.bimamba_type == "v2":
                A_b = -torch.exp(self.A_b_log.float())

                out_ret = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,
                    None,
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out, params_for_debug = _split_mamba_debug_return(out_ret)

                out_b_ret = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out_b, _ = _split_mamba_debug_return(out_b_ret)

                # 只保存正向分支的 debug 参数，便于 vis_attn_matrix.py 直接读取
                self._maybe_save_hidden_attn(params_for_debug, seqlen)

                out = F.linear(
                    rearrange(out + out_b.flip([-1]), "b d l -> b l d"),
                    self.out_proj.weight,
                    self.out_proj.bias,
                )

            elif self.bimamba_type == "v3":
                out_ret = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,
                    None,
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out, params_for_debug = _split_mamba_debug_return(out_ret)

                self._maybe_save_hidden_attn(params_for_debug, seqlen)

                out = rearrange(out, "b d l -> b l d")
                out = out[0, :, :] + out[1, :, :].flip([-2])
                out = self.norm(out)
                out = F.linear(out, self.out_proj.weight, self.out_proj.bias)

            else:
                out_ret = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,
                    None,
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out, params_for_debug = _split_mamba_debug_return(out_ret)

                self._maybe_save_hidden_attn(params_for_debug, seqlen)

        else:
            # 非 fused fast-path 下，selective_scan_fn 不返回 vis_attn_matrix 需要的 params_for_debug
            x, z = xz.chunk(2, dim=1)

            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv:])

            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )

            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)

            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)

        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"

        xz = self.in_proj(hidden_states.squeeze(1))
        x, z = xz.chunk(2, dim=-1)

        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.linear(dt, self.dt_proj.weight)
        A = -torch.exp(self.A_log.float())

        if selective_state_update is None:
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z,
                dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class MambaBlock(nn.Module):
    def __init__(
            self, dim, layer_idx, bimamba_type,
            norm_cls=nn.LayerNorm, fused_add_norm=False,
            residual_in_fp32=False, drop_path=0., ssm_cfg={}, factory_kwargs={}
    ):
        super().__init__()
        mixer_cls = partial(
            Mamba,
            layer_idx=layer_idx,
            bimamba_type=bimamba_type,
            **ssm_cfg,
            **factory_kwargs
        )
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states, residual=None, inference_params=None):
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)