import torch
from opensora.models.stdit.stdit import STDiTBlock, STDiT_XL_2
from opensora.models.stdit.stdit_quarot import QuarotSTDiTBlock, QuarotSTDiT_XL_2

from qdiff.quarot.quarot_utils import random_hadamard_matrix, matmul_hadU_cuda, get_hadK
import gc

device = 'cuda'
dtype = torch.float16
num_frames = 1

# --- test the error of hardamard ---
feature_size = 1152
hidden_size = 4608

x = torch.normal(0,1,size=[1, 1024, 1152]).to(device, dtype)
w = torch.normal(0,1,size=[4608, 1152]).to(device, dtype)

FIXED_HADAMARD=False

if FIXED_HADAMARD:
    hadK, K = get_hadK(feature_size)
    x_rotate = matmul_hadU_cuda(x, hadK, K).to(dtype)
    w_rotate = matmul_hadU_cuda(w, hadK, K).to(dtype)
else:
    Q = random_hadamard_matrix(feature_size, 'cuda').to(dtype)
    H = random_hadamard_matrix(hidden_size, 'cuda').to(dtype)
    gc.collect()  # cleanup memory
    torch.cuda.empty_cache()

    x_rotate = torch.matmul(x, Q).to(dtype)
    w_rotate = torch.matmul(w, Q).to(dtype)

y1 = torch.matmul(x, w.T)
y2 = torch.matmul(x_rotate, w_rotate.T)

# print(y1)
print((y1-y2).abs().max())
# torch.testing.assert_close(y1, y2, atol=1.e-5, rtol=1.e-3)

# -------------------------------
class SaveOutput:
    def __init__(self):
        self.outputs = []
    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)
    def clear(self):
        self.outputs = []

ckpt_path = "./logs/split_ckpt/OpenSora-v1-HQ-16x512x512-split.pth"

model = STDiT_XL_2(
    input_size=(num_frames,64,64),
    enable_flashattn=True,
    dtype=dtype,
    from_pretrained=ckpt_path)

model_quarot = QuarotSTDiT_XL_2(
    input_size=(num_frames,64,64),
    enable_flashattn=True,
    dtype=dtype,
    from_pretrained=ckpt_path)

model = model.to(device)
model_quarot = model_quarot.to(device)


# ori_block = STDiTBlock(hidden_size=1152, mlp_ratio=4.0, num_heads=16)
# quarot_block = QuarotSTDiTBlock(hidden_size=1152, mlp_ratio=4.0, num_heads=16)
## reinit the weights
# sd = ori_block.state_dict()
# quarot_block.load_state_dict(sd)  # let the 2 block have the same weights
# quarot_block.quarot_preprocess()  # process the weights

dummy_input = 5*torch.rand([1,1024,1152], dtype=dtype)   # [BS, C_in, C_out]
dummy_input = dummy_input.cuda()

# ------ test blocks -------

# for i_block in range(len(model.blocks)):
#     print('Testing Block {}'.format(i_block))

#     ori_block = model.blocks[i_block]
#     quarot_block = model_quarot.blocks[i_block]

#     # add hook
#     handles = []
#     save_output = SaveOutput()
#     handle0 = ori_block.mlp.fc1.register_forward_hook(save_output)
#     handle1 = quarot_block.mlp.fc1.register_forward_hook(save_output)

#     ori_out = ori_block.mlp(dummy_input)
#     quarot_out = quarot_block.mlp(dummy_input)

#     ori_fc1_out = save_output.outputs[0]
#     quarot_fc1_out = save_output.outputs[1]

#     ## check output
#     # print('Diff after FC1: ', (ori_fc1_out - quarot_fc1_out).abs())
#     # print('Diff after Whole Block: ', (ori_out - quarot_out).abs())
#     # torch.testing.assert_close(ori_fc1_out, quarot_fc1_out, atol=1.e-4, rtol=1.e-3)
#     torch.testing.assert_close(ori_out, quarot_out, atol=1.e-4, rtol=1.e-3)

# ------ test model -------
print('Test the model output..')
x = torch.rand([2, 4, num_frames, 64, 64], device='cuda', dtype=dtype)
timestep = torch.rand([2], device='cuda', dtype=dtype)
y = torch.rand([2, 1, 120, 4096], device='cuda', dtype=dtype)
mask = torch.ones([1, 120], device='cuda', dtype=dtype).int()

ori_model_out = model(x, timestep, y, mask)
quarot_model_out = model_quarot(x, timestep, y, mask)
torch.testing.assert_close(ori_model_out, quarot_model_out, atol=1.e-3, rtol=1.e-3)


