import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
# for profiling
import os
# import nvtx

from einops import rearrange
from timm.models.layers import DropPath
# from timm.models.vision_transformer import Mlp
from .modules import Mlp, to_2tuple

from opensora.acceleration.checkpoint import auto_grad_checkpoint
from opensora.acceleration.communications import gather_forward_split_backward, split_forward_gather_backward
from opensora.acceleration.parallel_states import get_sequence_parallel_group
from opensora.models.layers.blocks import (
    Attention,
    CaptionEmbedder,
    MultiHeadCrossAttention,
    PatchEmbed3D,
    SeqParallelAttention,
    SeqParallelMultiHeadCrossAttention,
    T2IFinalLayer,
    TimestepEmbedder,
    approx_gelu,
    get_1d_sincos_pos_embed,
    get_2d_sincos_pos_embed,
    get_layernorm,
    t2i_modulate,
)
from opensora.registry import MODELS
from opensora.utils.ckpt_utils import load_checkpoint

# ----- quarot utils --------
# import qdiff.quarot.quarot_utils as quarot_utils
from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda
import gc

class QuarotMlp(Mlp):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            norm_layer=None,
            bias=True,
            drop=0.,
            use_conv=False,
    ):
        super().__init__(
            in_features, 
            hidden_features,
            out_features,
            act_layer,
            norm_layer,
            bias,
            drop,
            use_conv,
            )

    def forward(self, x, set_ipdb=False):

        dtype_ = self.fc1.weight.dtype
        device_ = x.device

        x = torch.matmul(x.double(), self.Q).to(device_, dtype_)

        # K = 1
        # had_K = self.Q.shape[0]
        # x = matmul_hadU_cuda(x, had_K, K)

        x = self.fc1(x).to(torch.float16)
        x = self.act(x)
        x = self.drop1(x)
        x = self.norm(x)

        # quarot: rotate middle act: X*H
        x = torch.matmul(x.double(), self.H).to(device_, dtype_)

        # K = 1
        # had_K = self.H.shape[0]
        # x = matmul_hadU_cuda(x, had_K, K)

        x = self.fc2(x).to(torch.float16)
        x = self.drop2(x)
        return x


class QKRotater(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,q,k,Q=None):
        device_ = q.device
        dtype_ = q.dtype
        # rotate Q,K activation
        q = torch.matmul(q.double(), Q).to(device_, dtype_)
        k = torch.matmul(k.double(), Q).to(device_, dtype_)
        return q,k


class QuarotAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        enable_flashattn: bool = False,
        separate_qkv: bool = True,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.enable_flashattn = enable_flashattn

        # INFO: the original opensora code, use the self.qkv as a large linear layer with 3x channels
        # it is not compatible with quantization which use the same set of quant param for the whole layer
        # INFO: only support Attention for now, doesnot support SeqParallel
        self.separate_qkv = separate_qkv  # if needed, will be assiganed True from outside
        if self.separate_qkv:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.k = nn.Linear(dim, dim, bias=qkv_bias)
            self.v = nn.Linear(dim, dim, bias=qkv_bias)
        else: # default case for opensora
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.qk_rotater = QKRotater()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        # print(f"attn, x.shape: {x.shape}")
        if self.separate_qkv:
            q = self.q(x).unsqueeze(2)
            k = self.k(x).unsqueeze(2)
            v = self.v(x).unsqueeze(2)
            qkv = torch.cat([q,k,v], dim=2)  # concat to match the original code
        else:
            qkv = self.qkv(x)
        qkv_shape = (B, N, 3, self.num_heads, self.head_dim)
        if self.enable_flashattn:
            qkv_permute_shape = (2, 0, 1, 3, 4)
        else:
            qkv_permute_shape = (2, 0, 3, 1, 4)
        qkv = qkv.view(qkv_shape).permute(qkv_permute_shape)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        self.qk_rotater(q,k,Q=self.attn_Q)  # for easy register_hook

        if self.enable_flashattn:
            from flash_attn import flash_attn_func

            x = flash_attn_func(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                softmax_scale=self.scale,
            )
        else:
            dtype = q.dtype
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)  # translate attn to float32
            attn = attn.to(torch.float32)
            attn = attn.softmax(dim=-1)
            attn = attn.to(dtype)  # cast back attn to original dtype
            attn = self.attn_drop(attn)
            x = attn @ v

        x_output_shape = (B, N, C)
        if not self.enable_flashattn:
            x = x.transpose(1, 2)
        x = x.reshape(x_output_shape)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class QuarotSTDiTBlock(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads,
        d_s=None,
        d_t=None,
        mlp_ratio=4.0,
        drop_path=0.0,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        separate_qkv=True,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.enable_flashattn = enable_flashattn
        self._enable_sequence_parallelism = enable_sequence_parallelism
        

        if enable_sequence_parallelism:
            self.attn_cls = SeqParallelAttention
            self.mha_cls = SeqParallelMultiHeadCrossAttention
        else:
            # self.attn_cls = QuarotAttention
            self.attn_cls = Attention   # debug with no quarot attention
            self.mha_cls = MultiHeadCrossAttention
        # assert self.attn_cls == QuarotAttention

        self.norm1 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.attn = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=enable_flashattn,
            separate_qkv=separate_qkv,
        )
        self.cross_attn = self.mha_cls(hidden_size, num_heads)
        self.norm2 = get_layernorm(hidden_size, eps=1e-6, affine=False, use_kernel=enable_layernorm_kernel)
        self.mlp = QuarotMlp(
            in_features=hidden_size, hidden_features=int(hidden_size * mlp_ratio), act_layer=approx_gelu, drop=0
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.scale_shift_table = nn.Parameter(torch.randn(6, hidden_size) / hidden_size**0.5)

        # temporal attention
        self.d_s = d_s
        self.d_t = d_t

        if self._enable_sequence_parallelism:
            sp_size = dist.get_world_size(get_sequence_parallel_group())
            # make sure d_t is divisible by sp_size
            assert d_t % sp_size == 0
            self.d_t = d_t // sp_size

        self.attn_temp = self.attn_cls(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            enable_flashattn=self.enable_flashattn,
            separate_qkv=separate_qkv,
        )
    
    def quarot_rotate_weight(self):

        # quarot: inintialize the hadamard matrix
        # feature_size = 1152
        # hidden_size = 4608
        # self.Q = random_hadamard_matrix(feature_size, 'cpu')
        # self.H = random_hadamard_matrix(hidden_size, 'cpu')
        # gc.collect()  # cleanup memory
        # torch.cuda.empty_cache()

        dtype_ = torch.float32

        self.mlp.Q = self.Q
        self.mlp.H = self.H
        self.attn.attn_Q = self.attn_Q
        self.attn_temp.attn_Q = self.attn_Q

        self.mlp.fc1.to(dtype_)
        self.mlp.fc2.to(dtype_)

        # W_ = self.attn.q.weight.data.reshape([self.hidden_size,
        #                                 self.attn.num_heads,
        #                                 self.hidden_size//self.attn.num_heads])
        # self.attn.q.weight.data = torch.matmul(W_.cuda().double(), self.attn_Q).to(dtype_).reshape([self.hidden_size, self.hidden_size])

        # W_ = self.attn.k.weight.data.reshape([self.hidden_size,
        #                                 self.attn.num_heads,
        #                                 self.hidden_size//self.attn.num_heads])
        # self.attn.k.weight.data = torch.matmul(W_.cuda().double(), self.attn_Q).to(dtype_).reshape([self.hidden_size, self.hidden_size])

        # quarot: rotate the MLP(fc1): W*Q
        # INFO: always make the 2 fc layers FP32
        self.mlp.fc1.weight.data = torch.matmul(self.mlp.fc1.weight.data.double().cuda(), self.Q).to(dtype_)

        # K = 1
        # had_K = self.Q.shape[0]
        # self.mlp.fc1.weight.data = matmul_hadU_cuda(self.mlp.fc1.weight.data.cuda(), had_K, K)

        # quarot: rotate the MLP(fc2):  W*H
        self.mlp.fc2.weight.data = torch.matmul(self.mlp.fc2.weight.data.double().cuda(), self.H).to(dtype_)

        # K = 1
        # had_K = self.H.shape[0]
        # self.mlp.fc2.weight.data = matmul_hadU_cuda(self.mlp.fc2.weight.data.cuda(), had_K, K)


    def forward(self, x, y, t, mask=None, tpe=None, set_ipdb=False):

        if torch.isnan(x).any():
            import ipdb; ipdb.set_trace()

        B, N, C = x.shape

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table[None] + t.reshape(B, 6, -1)
        ).chunk(6, dim=1)
        x_m = t2i_modulate(self.norm1(x), shift_msa, scale_msa)
        # spatial branch
        x_s = rearrange(x_m, "B (T S) C -> (B T) S C", T=self.d_t, S=self.d_s)
        # print(f"conduct attn spitial"]
        x_s = self.attn(x_s)
        x_s = rearrange(x_s, "(B T) S C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_s)

        # temporal branch
        x_t = rearrange(x, "B (T S) C -> (B S) T C", T=self.d_t, S=self.d_s)
        if tpe is not None:
            x_t = x_t + tpe
        # print(f"conduct attn temp")
        x_t = self.attn_temp(x_t)
        x_t = rearrange(x_t, "(B S) T C -> B (T S) C", T=self.d_t, S=self.d_s)
        x = x + self.drop_path(gate_msa * x_t)

        # cross attn
        x = x + self.cross_attn(x, y, mask)

        # mlp
        output = self.drop_path(gate_mlp * self.mlp(t2i_modulate(self.norm2(x), shift_mlp, scale_mlp), set_ipdb=set_ipdb))
        if set_ipdb:
            import ipdb; ipdb.set_trace()
            
        x = x + output
        
        if set_ipdb:
            import ipdb; ipdb.set_trace()

        return x


@MODELS.register_module()
class QuarotSTDiT(nn.Module):
    def __init__(
        self,
        input_size=(1, 32, 32),
        in_channels=4,
        patch_size=(1, 2, 2),
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        pred_sigma=True,
        drop_path=0.0,
        no_temporal_pos_emb=False,
        caption_channels=4096,
        model_max_length=120,
        dtype=torch.float32,
        space_scale=1.0,
        time_scale=1.0,
        freeze=None,
        enable_flashattn=False,
        enable_layernorm_kernel=False,
        enable_sequence_parallelism=False,
        separate_qkv=True,
    ):
        super().__init__()
        self.pred_sigma = pred_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if pred_sigma else in_channels
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.input_size = input_size
        num_patches = np.prod([input_size[i] // patch_size[i] for i in range(3)])
        self.num_patches = num_patches
        self.num_temporal = input_size[0] // patch_size[0]
        self.num_spatial = num_patches // self.num_temporal
        self.num_heads = num_heads
        self.dtype = dtype
        self.no_temporal_pos_emb = no_temporal_pos_emb
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.enable_flashattn = enable_flashattn
        self.enable_layernorm_kernel = enable_layernorm_kernel
        self.separate_qkv = separate_qkv
        self.space_scale = space_scale
        self.time_scale = time_scale

        self.register_buffer("pos_embed", self.get_spatial_pos_embed())
        self.register_buffer("pos_embed_temporal", self.get_temporal_pos_embed())

        self.x_embedder = PatchEmbed3D(patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        self.y_embedder = CaptionEmbedder(
            in_channels=caption_channels,
            hidden_size=hidden_size,
            uncond_prob=class_dropout_prob,
            act_layer=approx_gelu,
            token_num=model_max_length,
        )

        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                QuarotSTDiTBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=drop_path[i],
                    enable_flashattn=self.enable_flashattn,
                    enable_layernorm_kernel=self.enable_layernorm_kernel,
                    enable_sequence_parallelism=enable_sequence_parallelism,
                    d_t=self.num_temporal,
                    d_s=self.num_spatial,
                    separate_qkv=self.separate_qkv,
                )
                for i in range(self.depth)
            ]
        )
        self.final_layer = T2IFinalLayer(hidden_size, np.prod(self.patch_size), self.out_channels)

        # init model
        self.initialize_weights()
        self.initialize_temporal()
        if freeze is not None:
            assert freeze in ["not_temporal", "text"]
            if freeze == "not_temporal":
                self.freeze_not_temporal()
            elif freeze == "text":
                self.freeze_text()

        # sequence parallel related configs
        self.enable_sequence_parallelism = enable_sequence_parallelism
        if enable_sequence_parallelism:
            self.sp_rank = dist.get_rank(get_sequence_parallel_group())
        else:
            self.sp_rank = None

        # separate qkv layer for quantization
        self.separate_qkv = True

    def convert_quarot(self):
        feature_size = self.hidden_size
        hidden_size = int(self.hidden_size*self.mlp_ratio)
        self.Q = random_hadamard_matrix(feature_size, 'cuda')  # 1152
        self.H = random_hadamard_matrix(hidden_size, 'cuda')   # 4608
        self.attn_Q = random_hadamard_matrix(feature_size//self.num_heads, 'cuda')  # 72
        gc.collect()  # cleanup memory
        torch.cuda.empty_cache()

        for block in self.blocks:
            block.Q = self.Q
            block.H = self.H
            block.attn_Q = self.attn_Q

            block.quarot_rotate_weight()

    def forward(self, x, timestep, y, mask=None):
        """
        Forward pass of STDiT.
        Args:
            x (torch.Tensor): latent representation of video; of shape [B, C, T, H, W]
            timestep (torch.Tensor): diffusion time steps; of shape [B]
            y (torch.Tensor): representation of prompts; of shape [B, 1, N_token, C]
            mask (torch.Tensor): mask for selecting prompt tokens; of shape [B, N_token]

        eturns:
            x (torch.Tensor): output latent representation; of shape [B, C, T, H, W]
        """
        x = x.to(self.dtype)
        timestep = timestep.to(self.dtype)
        y = y.to(self.dtype)

        # embedding
        x = self.x_embedder(x)  # [B, N, C]
        x = rearrange(x, "B (T S) C -> B T S C", T=self.num_temporal, S=self.num_spatial)
        x = x + self.pos_embed
        x = rearrange(x, "B T S C -> B (T S) C")

        # shard over the sequence dim if sp is enabled
        if self.enable_sequence_parallelism:
            x = split_forward_gather_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="down")

        t = self.t_embedder(timestep, dtype=x.dtype)  # [B, C]
        t0 = self.t_block(t)  # [B, 6 * C]
        y = self.y_embedder(y, self.training)  # [B, 1, N_token, C]

        # INFO: the default opensora implementation is equivalent with mask_select=True
        # the mask_select will cause variant input y.shape (varying input activation shape), making **static** quantization hard to process
        # when act_quantizer is not dynamic, we use MASK_SELECT=False, which replace selection with 0. masking to ensure the same input shape
        # however, when the prompt length is very short (much smaller than 120, e.g., the UCF101 dataset), using MASK_SELECT=False will incur bad results
        if mask is not None:
            from qdiff.quantizer.dynamic_quantizer import DynamicActQuantizer
            # DIRTY, assume that all layers in the stdit model share the same quantization configuration
            MASK_SELECT = True
            if hasattr(self.final_layer.linear, 'act_quantizer'):
                if not isinstance(self.final_layer.linear.act_quantizer, DynamicActQuantizer): # static quant param
                    if self.final_layer.linear.act_quantizer.per_group == 'token':
                        MASK_SELECT = False

            if MASK_SELECT:
                ## Original version: y is smaller 3684/3840
                if mask.shape[0] != y.shape[0]:
                    mask = mask.repeat(y.shape[0] // mask.shape[0], 1)
                mask = mask.squeeze(1).squeeze(1)
                y = y.squeeze(1).masked_select(mask.unsqueeze(-1) != 0).view(1, -1, x.shape[-1])
                y_lens = mask.sum(dim=1).tolist()
            else:
                ## Ours Version: always 3840 (max_n_prompt*bs)
                # Interestingly, the original STDiT model takes in [bs*2] as y and [bs] as mask
                if mask.shape[0] != y.shape[0]:
                    assert y.shape[0] == 2*mask.shape[0]
                    try:
                        mask_ = mask.repeat([2,1])
                    except:
                        import ipdb; ipdb.set_trace()
                else:
                    mask_ = mask

                y_lens = [y.shape[2]] * y.shape[0]
                y = y*mask_.unsqueeze(-1).unsqueeze(1)
            y = y.squeeze(1).view(1, -1, x.shape[-1])
        else:
            y_lens = [y.shape[2]] * y.shape[0]
            y = y.squeeze(1).view(1, -1, x.shape[-1])

        # blocks
        for i, block in enumerate(self.blocks):
            if i == 0:
                if self.enable_sequence_parallelism:
                    tpe = torch.chunk(
                        self.pos_embed_temporal, dist.get_world_size(get_sequence_parallel_group()), dim=1
                    )[self.sp_rank].contiguous()
                else:
                    tpe = self.pos_embed_temporal # [1, T, C]
            else:
                tpe = None
            x_ = auto_grad_checkpoint(block, x, y, t0, y_lens, tpe, set_ipdb=False) # (i==len(self.blocks)-1)

            # INFO: profiling the block
            PROFILE=False
            if PROFILE:
                torch.cuda.cudart().cudaProfilerStart()
                x = auto_grad_checkpoint(block, x, y, t0, y_lens, tpe)
                torch.cuda.cudart().cudaProfilerStop()
            else:
                x = x_

            # print(f"{i}-th block, x[0,0,0]: {x[0,0,0]}")
            # print(f"{i}-th block, x.shape: {x.shape}; y.shape: {y.shape}")

        if self.enable_sequence_parallelism:
            x = gather_forward_split_backward(x, get_sequence_parallel_group(), dim=1, grad_scale="up")
        # x.shape: [B, N, C]

        # final process
        x = self.final_layer(x, t)  # [B, N, C=T_p * H_p * W_p * C_out]
        x = self.unpatchify(x)  # [B, C_out, T, H, W]

        # cast to float32 for better accuracy
        x = x.to(torch.float32)
        return x

    def unpatchify(self, x):
        """
        Args:
            x (torch.Tensor): of shape [B, N, C]

        Return:
            x (torch.Tensor): of shape [B, C_out, T, H, W]
        """

        N_t, N_h, N_w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        T_p, H_p, W_p = self.patch_size
        x = rearrange(
            x,
            "B (N_t N_h N_w) (T_p H_p W_p C_out) -> B C_out (N_t T_p) (N_h H_p) (N_w W_p)",
            N_t=N_t,
            N_h=N_h,
            N_w=N_w,
            T_p=T_p,
            H_p=H_p,
            W_p=W_p,
            C_out=self.out_channels,
        )
        return x

    def unpatchify_old(self, x):
        c = self.out_channels
        t, h, w = [self.input_size[i] // self.patch_size[i] for i in range(3)]
        pt, ph, pw = self.patch_size

        x = x.reshape(shape=(x.shape[0], t, h, w, pt, ph, pw, c))
        x = rearrange(x, "n t h w r p q c -> n c t r h p w q")
        imgs = x.reshape(shape=(x.shape[0], c, t * pt, h * ph, w * pw))
        return imgs

    def get_spatial_pos_embed(self, grid_size=None):
        if grid_size is None:
            grid_size = self.input_size[1:]
        pos_embed = get_2d_sincos_pos_embed(
            self.hidden_size,
            (grid_size[0] // self.patch_size[1], grid_size[1] // self.patch_size[2]),
            scale=self.space_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def get_temporal_pos_embed(self):
        pos_embed = get_1d_sincos_pos_embed(
            self.hidden_size,
            self.input_size[0] // self.patch_size[0],
            scale=self.time_scale,
        )
        pos_embed = torch.from_numpy(pos_embed).float().unsqueeze(0).requires_grad_(False)
        return pos_embed

    def freeze_not_temporal(self):
        for n, p in self.named_parameters():
            if "attn_temp" not in n:
                p.requires_grad = False

    def freeze_text(self):
        for n, p in self.named_parameters():
            if "cross_attn" in n:
                p.requires_grad = False

    def initialize_temporal(self):
        for block in self.blocks:
            nn.init.constant_(block.attn_temp.proj.weight, 0)
            nn.init.constant_(block.attn_temp.proj.bias, 0)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        nn.init.normal_(self.t_block[1].weight, std=0.02)

        # Initialize caption embedding MLP:
        nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

        # Zero-out adaLN modulation layers in PixArt blocks:
        for block in self.blocks:
            nn.init.constant_(block.cross_attn.proj.weight, 0)
            nn.init.constant_(block.cross_attn.proj.bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

def recursive_find_module(module, prefix='', save_dict={}):
    for name, child_module in module.named_children():
        full_name = prefix + name if prefix else name
        if isinstance(child_module, Attention):
            save_dict[full_name] = child_module
        else:
            recursive_find_module(child_module, prefix=full_name+'.', save_dict=save_dict)
    return save_dict


@MODELS.register_module("QuarotSTDiT-XL/2")
def QuarotSTDiT_XL_2(from_pretrained=None, **kwargs):
    model = QuarotSTDiT(depth=28, hidden_size=1152, patch_size=(1, 2, 2), num_heads=16, **kwargs)
    if from_pretrained is not None:  
        load_checkpoint(model, from_pretrained)

    # INFO: support separate_qkv when loading the model to fit quantization
    if model.separate_qkv:
        if any('qkv' in k_ for k_ in model.state_dict().keys()):
            # iterate through all blocks to find all attn layers
            found_modules_d = recursive_find_module(model)
            for name, module in found_modules_d.items():
                # unpack the qkv layers
                has_bias = module.qkv.bias is not None
                qkv_dim = module.qkv.weight.shape[-1]
                qkv_weight = module.qkv.weight.reshape([3,qkv_dim,qkv_dim]).unbind(0) # [3]
                qkv_bias = module.qkv.bias.reshape([3,qkv_dim]).unbind(0)
                # load into 3 layers
                module.q = nn.Linear(qkv_dim, qkv_dim, bias=has_bias)
                module.q.weight = nn.Parameter(qkv_weight[0])
                module.q.bias = nn.Parameter(qkv_bias[0])
                module.k = nn.Linear(qkv_dim, qkv_dim, bias=has_bias)
                module.k.weight = nn.Parameter(qkv_weight[1])
                module.k.bias = nn.Parameter(qkv_bias[1])
                module.v = nn.Linear(qkv_dim, qkv_dim, bias=has_bias)
                module.v.weight = nn.Parameter(qkv_weight[2])
                module.v.bias = nn.Parameter(qkv_bias[2])
                delattr(module,'qkv')
                module.separate_qkv = True
        # if the model are already converted, skip the rest
    
    if 'dtype' in kwargs.keys():
        model.to(kwargs['dtype'])

    model.convert_quarot()

    return model


