import torch
from dalle2_pytorch import DALLE2, DiffusionPriorNetwork, DiffusionPrior, Unet, Decoder, CLIP
import os
import numpy as np
import cv2
from sklearn.preprocessing import MinMaxScaler

clip = CLIP(
    dim_text = 512,
    dim_image = 512,
    dim_latent = 512,
    num_text_tokens = 49408,
    text_enc_depth = 6,
    text_seq_len = 256,
    text_heads = 8,
    visual_enc_depth = 6,
    visual_image_size = 256,
    visual_patch_size = 32,
    visual_heads = 8
)

# 打印读取到的文本向量
text = []
file_path = 'D:\Desktop/python mode/Novel cover generation/dataset/text.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()
for line in lines:
    string = line.replace("tensor(", "").replace(")", "").split()  
    float_list = [float(num) for num in string]
    tensor = torch.tensor(float_list)
    text.append(tensor)
text = torch.stack(text)
scaler = MinMaxScaler(feature_range=(0, 10000),copy=True)
text = scaler.fit_transform(text)
text = torch.Tensor(text[0:4]).long()

# 获取图像数据
images_path = []
images = []
path_imgs = 'D:\Desktop/python mode/Novel cover generation/dataset/images_1/'
for files in os.listdir(path_imgs):
    image_path = path_imgs + files
    img_name =  files[0:-4]
    images_path.append(image_path)
for image_path in images_path:
    image = cv2.resize(cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1),(256,256))
    images.append(image)
images = torch.Tensor(images[0:4]).long().float()
images = images.permute(0, 3, 1, 2)

# mock data

# text = torch.randint(0, 49408, (4, 256))
# images = torch.randn(4, 3, 256, 256)

print(images.shape)  #（已转变为张量后）
print(images.dtype)
print(text.shape)  #（已转变为张量后）
print(text.dtype)

# train

loss = clip(
    text,
    images,
    return_loss = True
)

loss.backward()

# do above for many steps ...

# prior networks (with transformer)

prior_network = DiffusionPriorNetwork(
    dim = 512,
    depth = 6,
    dim_head = 64,
    heads = 8
)

diffusion_prior = DiffusionPrior(
    net = prior_network,
    clip = clip,
    timesteps = 1000,
    sample_timesteps = 64,
    cond_drop_prob = 0.2
)

loss = diffusion_prior(text, images)
loss.backward()

# do above for many steps ...

# decoder (with unet)

unet1 = Unet(
    dim = 128,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults=(1, 2, 4, 8),
    cond_on_text_encodings = False   # set to True for any unets that need to be conditioned on text encodings
)

unet2 = Unet(
    dim = 16,
    image_embed_dim = 512,
    cond_dim = 128,
    channels = 3,
    dim_mults = (1, 2, 4, 8, 16)
)

decoder = Decoder(
    unet = (unet1, unet2),
    image_sizes = (128, 256),
    clip = clip,
    timesteps = 100,
    image_cond_drop_prob = 0.1,
    text_cond_drop_prob = 0.5
)

for unet_number in (1, 2):
    loss = decoder(images, unet_number = unet_number) # this can optionally be decoder(images, text) if you wish to condition on the text encodings as well, though it was hinted in the paper it didn't do much
    loss.backward()

# do above for many steps

# 保存模型
# diffusion_prior_save_path = "D:\Desktop/python mode/Novel cover generation/dalle-2/diffusion_prior_model.pt"
# decoder_save_path = "D:\Desktop/python mode/Novel cover generation/dalle-2/decoder_model.pt"

# torch.save(diffusion_prior.state_dict(), diffusion_prior_save_path)
# torch.save(decoder.state_dict(), decoder_save_path)

dalle2 = DALLE2(
    prior = diffusion_prior,
    decoder = decoder
)

dalle2_save_model_path = "D:\Desktop/python mode/Novel cover generation/dalle-2/dalle2_model.pt"
torch.save(dalle2.state_dict(), dalle2_save_model_path)

images = dalle2(
    ['一人成仙'],
    cond_scale = 2. # classifier free guidance strength (> 1 would strengthen the condition)
)


cv2.imshow('一人成仙',images)