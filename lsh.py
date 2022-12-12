# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 15:39:48 2022

@author: ybs
"""
import argparse
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import numpy as np
import re
import imagehash
import torch
import torch.nn
import torchvision.models as models
from torch.autograd import Variable 
import torch.cuda
import torchvision.transforms as transforms
import numpy as np
import glob
from lshashpy3 import LSHash
from PIL import Image
from sklearn.decomposition import PCA

TARGET_IMG_SIZE = 224
img_to_tensor = transforms.ToTensor()

def make_model():
    model=models.vgg19(pretrained=True)	
    new_classifier = torch.nn.Sequential(*list(model.children())[-1][:6])
    model=model.eval()	# 一定要有这行，不然运算速度会变慢（要求梯度）而且会影响结果
    model.cuda()	# 将模型从CPU发送到GPU,如果没有GPU则删除该行
    model.classifier = new_classifier
    return model
    
#特征提取
def extract_feature(model,imgpath):
    model.eval()		# 必须要有，不然会影响特征提取结果 
    img=Image.open(imgpath)		# 读取图片
    img = img.resize((TARGET_IMG_SIZE, TARGET_IMG_SIZE)) 
    img = img.convert('RGB')  # Make sure img is color
    tensor=img_to_tensor(img)	
    tensor = Variable(tensor)
    tensor = tensor.reshape(1, 3, TARGET_IMG_SIZE, TARGET_IMG_SIZE)
    tensor = tensor.cuda()	# 如果只是在cpu上跑的话要将这行去掉
    result = model(tensor)
    result_npy = result.data.cpu().numpy()	# 保存的时候一定要记得转成cpu形式的，将数据移至CPU中,不然可能会出错
    return result_npy[0] 	# 返回的矩阵shape是[1, 4096]，这么做是为了让shape变回[4096]

def vgg_Extraction(imgpath):           #提取imgpath图片的特征
    model=make_model()
    tmp = extract_feature(model, imgpath)
    dhash = np.array(tmp)              #将返回的图片特征存入np数组中
    signature=dhash.flatten()          #数组拉直
    return signature
def Register(image_path:str,signature):       #注册，存储特征
    new_feature=dict()
    new_feature[image_path]=signature
    feature_old = np.load('features.npy',allow_pickle=True).item()
    feature_old.update(new_feature)
    np.save('features.npy',feature_old)
def lshash(detect_image:str, hash_size: int, table_num: int):
    lsh = LSHash(hash_size,4096,table_num,matrices_filename='plane.npz',
                 hashtable_filename='hash.npz',overwrite=False)         #plane.npz和hash.npz分别存储超平面信息和hash表信息
    if os.path.exists('features.npy'):                                  #feature.npy存储所有图片的特征                 
        feature_file = np.load('features.npy',allow_pickle=True)
        ff = feature_file.item()
        for f in ff:
            lsh.index(ff[f], extra_data=f)                              #生成索引
    else:
        file_list=[]
        features=dict()
        for r, d, f in os.walk('101'):                                  #取出本地所有图片放入file_list
            for name in f:
                if name.endswith('.jpg'):
                    file_list.append(os.path.join(r, name))
        for fh in file_list:
            try:
                signature = vgg_Extraction(fh)
                #print(fh)
                #print(signature)
                lsh.index(signature, extra_data=fh)
            except IOError:
                continue
            #img_name = os.path.split(fh)[1]
            features[fh]=signature
        np.save("features.npy", features)
    lsh.save()
    imageSignature=vgg_Extraction(detect_image)
    lsh_search = lsh.query(imageSignature,num_results=10,distance_func=('cosine'))
    #query_results = [(name, float(dist)) for ((vec, name), dist) in lsh_search]
    results = [(name, float(dist)) for ((vec, name), dist) in lsh_search if(float(dist)<0.277)] #cosine距离小于0.277的放入结果中
    if(len(results) > 0):
        print('Find ',len(results),'duplicate files for ',detect_image)
        for (name,dist)in results:
            print(name,", dist=",dist)
    else:
        Register(detect_image,imageSignature) 
def vgg_lshash(detect_image:str,hash_size:int,table_num:int):
    lshash(detect_image,hash_size,table_num)
'''
def main():
    vgg_lshash('2.jpg',4,16)
if __name__ == "__main__":
    main()
'''