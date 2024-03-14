import os
import numpy as np
import cv2
import torch
from sklearn.preprocessing import MinMaxScaler


texts = []
images_path = []
img = []
path_imgs = 'D:\Desktop/python mode/Novel cover generation/dataset/images_1/'
for files in os.listdir(path_imgs):
    image_path = path_imgs + files
    img_name =  files[0:-4]
    images_path.append(image_path)
    texts.append(str(img_name))

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
scaler = MinMaxScaler(feature_range=(0, 49408),copy=True)
text1 = scaler.fit_transform(text)
text1 = torch.Tensor(text1).long()


for image_path in images_path:
    image = cv2.resize(cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1),(256,256))
    img.append(image)
img = torch.Tensor(img)
img = img.permute(0, 3, 1, 2)
img = img.float() / 255.0
print(len(text))
print(len(img))
print(img[1])
print(img[1].shape)
print(img.shape)
print(img.max())  #（已转变为张量后）
print(img.min())
print(text.max())  #（已转变为张量后）
print(text.min())
# print(text.shape)
# print(text1[11:14])
# print(text[11:14])
# print(texts[11:14])

#text为文本数据，img为图像数据

# #文本编码
# import torch
# from transformers import AutoTokenizer, AutoModel


# tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
# model = AutoModel.from_pretrained('bert-base-chinese')

# encoded_texts = []

# for text in texts:
#     # 编码文本为Transformer特征
#     tokens = tokenizer.encode(text, truncation=True, padding='max_length', max_length=10)
#     input_ids = torch.tensor(tokens).unsqueeze(0)  # 添加batch维度

#     # 输入模型获得Transformer特征
#     with torch.no_grad():
#         outputs = model(input_ids)
#         features = outputs.last_hidden_state.squeeze(0)  # 去除batch维度

#     # 获取[CLS]标记的隐藏状态作为句子级别的表示
#     sentence_features = features[0]

#     # 对特征进行截断或填充为256长度
#     if sentence_features.size(0) < 256:
#         padding = torch.zeros(256 - sentence_features.size(0))
#         sentence_features = torch.cat((sentence_features, padding), dim=0)
#     else:
#         sentence_features = sentence_features[:256]
#     encoded_texts.append(sentence_features)

# # 打印编码结果
# # for i, text in enumerate(texts):
# print(f'Text: {texts[11]}')
# print(f'Encoded Features: {encoded_texts[11].shape}')
# print(encoded_texts[11])
# print(len(encoded_texts))

# # 指定要保存的文件路径
# file_path = 'D:\Desktop/python mode/Novel cover generation/dataset/text.txt'

# # 将文本写入txt文件
# with open(file_path, 'w') as file:
#     for vector in encoded_texts:
#         file.write(' '.join(map(str, vector)) + '\n')

# with open(file_path, 'r', encoding='utf-8') as file:
#     lines = file.readlines()

# encoded_texts = []
# # # 打印读取到的文本
# for line in lines:
#     string = line.replace("tensor(", "").replace(")", "").split()  # 去除换行符
#     float_list = [float(num) for num in string]
#     tensor = torch.tensor(float_list)
#     encoded_texts.append(tensor)
# encoded_texts = torch.stack(encoded_texts)
# print(encoded_texts[11])