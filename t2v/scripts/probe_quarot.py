import os
import sys
# sys.path.append(".")

import torch
import colossalai
import torch.distributed as dist
from mmengine.runner import set_random_seed

from opensora.datasets import save_sample
from opensora.registry import MODELS, SCHEDULERS, build_module
from opensora.utils.config_utils import parse_configs
from opensora.utils.misc import to_torch_dtype
from opensora.acceleration.parallel_states import set_sequence_parallel_group
from colossalai.cluster import DistCoordinator


def load_prompts(prompt_path):
    with open(prompt_path, "r") as f:
        prompts = [line.strip() for line in f.readlines()]
    return prompts

class SaveOutput:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        # self.outputs.append(module_in[0].abs().max(dim=1)[0])
        self.outputs.append(module_in[0])
    def clear(self):
        self.outputs = []

class SaveQKRoate:

    def __init__(self):
        self.Q = []
        self.K = []
        self.Q_rotate = []
        self.K_rotate = []

    def __call__(self, module, module_in, module_out):
        self.Q.append(module_in[0])
        self.K.append(module_in[1])
        self.Q_rotate.append(module_out[0])
        self.K_rotate.append(module_out[1])

    def clear(self):
        self.Q = []
        self.K = []
        self.Q_rotate = []
        self.K_rotate = []

def main():
    # ======================================================
    # 1. cfg and init distributed env
    # ======================================================
    cfg = parse_configs(training=False, mode="get_calib")
    print(cfg)

    # init distributed
    # colossalai.launch_from_torch({})
    # coordinator = DistCoordinator()

    # if coordinator.world_size > 1:
        # set_sequence_parallel_group(dist.group.WORLD) 
        # enable_sequence_parallelism = True
    # else:
        # enable_sequence_parallelism = False
    
    # ======================================================
    # 2. runtime variables
    # ======================================================
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = f"cuda:{cfg.gpu}" if torch.cuda.is_available() else "cpu"
    dtype = to_torch_dtype(cfg.dtype)
    set_random_seed(seed=cfg.seed)
    prompts = load_prompts(cfg.prompt_path)
    prompts = prompts[:cfg.data_num]
    PRECOMPUTE_TEXT_EMBEDS = cfg.get('precompute_text_embeds', None)

    # ======================================================
    # 3. build model & load weights
    # =====================================
    # 
    # =================
    # 3.1. build scheduler
    scheduler = build_module(cfg.scheduler, SCHEDULERS)
    
    # 3.2. build model
    input_size = (cfg.num_frames, *cfg.image_size)
    vae = build_module(cfg.vae, MODELS)
    latent_size = vae.get_latent_size(input_size)

    # ori_model
    model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        # caption_channels=text_encoder.output_dim,
        caption_channels=4096,  # DIRTY: for T5 only
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=False,
    )

    # quarot_model
    cfg.model['type'] = 'Quarot'+cfg.model['type']
    quarot_model = build_module(
        cfg.model,
        MODELS,
        input_size=latent_size,
        in_channels=vae.out_channels,
        # caption_channels=text_encoder.output_dim,
        caption_channels=4096,  # DIRTY: for T5 only
        model_max_length=cfg.text_encoder.model_max_length,
        dtype=dtype,
        enable_sequence_parallelism=False,
    )

    # add the hooks:
    original_model_saved_fc1 = SaveOutput()
    original_model_saved_fc2 = SaveOutput()
    quarot_model_saved_fc1 = SaveOutput()
    quarot_model_saved_fc2 = SaveOutput()
    quarot_model_saved_qk = SaveQKRoate()

    # i_block = 0
    for i_block in range(len(model.blocks)):
        if i_block in [0,26]:
            model.blocks[i_block].mlp.fc1.register_forward_hook(original_model_saved_fc1)
            model.blocks[i_block].mlp.fc2.register_forward_hook(original_model_saved_fc2)
            quarot_model.blocks[i_block].mlp.fc1.register_forward_hook(quarot_model_saved_fc1)
            quarot_model.blocks[i_block].mlp.fc2.register_forward_hook(quarot_model_saved_fc2)
        # if i_block == 26:
            # quarot_model.blocks[i_block].attn.qk_rotater.register_forward_hook(quarot_model_saved_qk)

    if PRECOMPUTE_TEXT_EMBEDS is not None:
        text_encoder = None
    else:
        text_encoder = build_module(cfg.text_encoder, MODELS, device=device)  # T5 must be fp32
        text_encoder.y_embedder = model.y_embedder  # hack for classifier-free guidance

    # 3.3. move to device & eval
    vae = vae.to(device, dtype).eval()
    model = model.to(device, dtype).eval()
    quarot_model = quarot_model.to(device, dtype).eval()

    # 3.4. support for multi-resolution
    model_args = dict()
    if cfg.multi_resolution:
        image_size = cfg.image_size
        hw = torch.tensor([image_size], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        ar = torch.tensor([[image_size[0] / image_size[1]]], device=device, dtype=dtype).repeat(cfg.batch_size, 1)
        model_args["data_info"] = dict(ar=ar, hw=hw)

    # ======================================================
    # 4. inference
    # ======================================================
    sample_idx = 0
    save_dir = cfg.save_dir
    os.makedirs(save_dir, exist_ok=True)
    calib_data = {}
    input_data_list = []
    output_data_list = []
    if PRECOMPUTE_TEXT_EMBEDS is not None:
        model_args['precompute_text_embeds'] = torch.load(cfg.precompute_text_embeds)

    for i in range(0, len(prompts), cfg.batch_size):
        batch_prompts = prompts[i : i + cfg.batch_size]
        if PRECOMPUTE_TEXT_EMBEDS is not None:  # also feed in the idxs for saved text_embeds
            model_args['batch_ids'] = torch.arange(i,i+cfg.batch_size)

        n = len(batch_prompts)
        init_noise = torch.randn(n, *(vae.out_channels, *latent_size), device=device)
        
        samples, cur_calib_data, out_data = scheduler.sample(
            model,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            return_trajectory=True,
            additional_args=model_args,
            init_noise=init_noise,
        )

        samples_quarot, cur_calib_data_quarot, out_data_quarot = scheduler.sample(
            quarot_model,
            text_encoder,
            sampler_type=cfg.sampler,
            z_size=(vae.out_channels, *latent_size),
            prompts=batch_prompts,
            device=device,
            return_trajectory=True,
            additional_args=model_args,
            init_noise=init_noise,
        )

        # ------ save the activations -----

        # for i in range(len(original_model_saved.outputs)):
            # original_model_saved.outputs[i] = original_model_saved.outputs[i].mean(dim=1) # [B, N_token, C] -> [B,C]
        # for i in range(len(quarot_model_saved.outputs)):
            # quarot_model_saved.outputs[i] = quarot_model_saved.outputs[i].mean(dim=1) # [B, N_token, C] -> [B,C]
        d = {}
        d['ori_fc1'] = original_model_saved_fc1.outputs
        d['ori_fc2'] = original_model_saved_fc2.outputs
        d['quarot_fc1'] = quarot_model_saved_fc1.outputs
        d['quarot_fc2'] = quarot_model_saved_fc2.outputs
        torch.save(d, './t2v/utils_files/quarot/quarot_model_saved_acts.pth')
        import ipdb; ipdb.set_trace()

        # torch.save(original_model_saved_fc1.outputs, './t2v/utils_files/quarot/original_model_saved_fc1.pth')  # [N_block, N_timestep]
        # torch.save(original_model_saved_fc2.outputs, './t2v/utils_files/quarot/original_model_saved_fc2.pth')  # [N_block, N_timestep]
        # torch.save(quarot_model_saved_fc1.outputs, './t2v/utils_files/quarot/quarot_model_saved_fc1.pth')
        # torch.save(quarot_model_saved_fc2.outputs, './t2v/utils_files/quarot/quarot_model_saved_fc2.pth')

        d = {}
        d['q'] = quarot_model_saved_qk.Q
        d['k'] = quarot_model_saved_qk.K
        d['q_rotate'] = quarot_model_saved_qk.Q_rotate
        d['k_rotate'] = quarot_model_saved_qk.K_rotate
        torch.save(d, './t2v/utils_files/quarot/quarot_model_saved_qks.pth')
        import ipdb; ipdb.set_trace()

        for key in cur_calib_data:
            if not key in calib_data.keys():
                calib_data[key] = cur_calib_data[key]
            else:
                calib_data[key] = torch.cat([cur_calib_data[key], calib_data[key]], dim=1)

        input_data_list.append(cur_calib_data)
        output_data_list.append(out_data)
        # save samples
        samples = vae.decode(samples.to(dtype))
        # if coordinator.is_master():
        for idx, sample in enumerate(samples):
            print(f"Prompt: {batch_prompts[idx]}")
            save_path = os.path.join(save_dir, f"sample_{sample_idx}")
            save_sample(sample, fps=cfg.fps, save_path=save_path)
            sample_idx += 1

    # ======================================================
    # 5. save calibration data
    # ======================================================
    torch.save(calib_data, os.path.join(save_dir, "calib_data.pt"))
    if cfg.save_inp_oup:
        torch.save(input_data_list, os.path.join(save_dir, "input_list.pt"))
        torch.save(output_data_list, os.path.join(save_dir, "output_list.pt"))


if __name__ == "__main__":
    main()
