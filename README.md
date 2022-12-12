# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 12:30:23 2022

@author: ybs
"""

## Usage

```
Usage:  lsh.py为主要程序,在需要检测的地方引用vgg_lshash函数

参数说明：detect_image(待检测图片的位置)，hash_size(LSHash的长度),table_num(哈希表的个数)
使用方法见：use.py
```

## 环境配置 非常重要！！！！

```
1.下载anaconda3

参考https://blog.csdn.net/qq_45344586/article/details/124028689

2.下载与电脑GPU相匹配的cuda     

参考https://blog.csdn.net/m0_45447650/article/details/123704930

3.下载与cuda版本匹配的pytorch  

参考https://blog.csdn.net/weixin_42496865/article/details/124002488

```

## 可能存在的问题
```

1.import某个包错误，就pip install 对应的包

2.vgg19网络模型下载失败，可打开同级目录/vgg19model，将模型文件复制至对应的文件夹中。eg:win10系统中的文件夹位置在：

~/.cache/torch/hub/checkpoints/

3.及时百度

```

## 函数说明
```

1.def make_model():

参考https://blog.csdn.net/me_yundou/article/details/109218273, 获得vgg19网络的倒数第二层，输出为4096维。

2.def extract_feature(model,imgpath):

获得路径为imagepath的图片的4096维特征

3.def vgg_Extraction(imgpath):

特征存入数组并作为图片的signature返回

4.def lshash(detect_image:str, hash_size: int, table_num: int):

参数为：图片路径，lsh哈希长度，哈希表数目

主要逻辑：

（1）构建lsh相关设置。

 # lsh = LSHash(hash_size,4096,table_num,matrices_filename='plane.npz',hashtable_filename='hash.npz',overwrite=False)     

 # 说明：LSHash为lshashpy3包中的函数，初始化哈希长度为hash_size，所需哈希表数目为table_num，输入数据为4096位的LSHash内容，指定相关的超平面文件和哈希表文件并存入磁盘，overwrite标识是否覆盖矩阵文件。具体参考https://pypi.org/project/lshashpy3/

（2）提取系统中已授权的图片signature签名（通过vgg_Extraction函数生成，存入features.npy文件）文件features.npy，为其中每条签名生成lsh.index。最后保存lsh文件。

 # lsh.index(signature, extra_data='图片的路径')

（3）提取待检测图片的signature，通过'lsh.query'函数获取查询信息。并返回cosine距离小于0.277的所有图片信息。

 # lsh_search = lsh.query(imageSignature,num_results=10,distance_func=('cosine'))

 # 说明：查询与imageSignature相似的图片信息，至多返回num_results个结果，结果按照distance_func排序。注意此处的distance_func只能为cosine，因为signature是小数数组。

（4）如果查询结果为0，就跳转到Register函数，将图片signature存入features.npy文件。

5.def Register(image_path:str,signature): 

将字典映射image_path：signature存入features.npy文件。
```

