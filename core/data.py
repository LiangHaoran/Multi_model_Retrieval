import os
import pandas as pd
import numpy as np
import base64
from tqdm import tqdm
from option import get_args
import warnings
from sklearn.externals import joblib
import torch
from torch.autograd import Variable
import time
import torch.utils.data as Data
import re
import argparse
warnings.filterwarnings('ignore')


def process_boxes(data):
    """
    process boxes
    :param data:
    :return:
    """
    boxes = np.array(data['boxes'])
    for i in tqdm(range(boxes.shape[0])):
        data['boxes'][i] = np.frombuffer(base64.b64decode(boxes[i]), dtype=np.float32).reshape(data['num_boxes'][i], 4)
    return data


def process_features(data):
    """
    process features
    :param data:
    :return:
    """
    features = np.array(data['features'])
    for i in tqdm(range(features.shape[0])):
        data['features'][i] = np.frombuffer(base64.b64decode(features[i]), dtype=np.float32).reshape(
            data['num_boxes'][i], 2048)
    return data


def process_class_labels(data):
    """
    process class_labels
    :param data:
    :return:
    """
    class_labels = np.array(data['class_labels'])
    for i in tqdm(range(class_labels.shape[0])):
        data['class_labels'][i] = np.frombuffer(base64.b64decode(class_labels[i]), dtype=np.int64).reshape(
            data['num_boxes'][i])
    return data


def convert_pos(num_boxes, boxes, H, W):
    """
    convert box position to 5-dim feature
    :param num_boxes:
    :param boxes:
    :param H:
    :param W:
    :return:
    """
    out = []
    for i in range(num_boxes.shape[0]):
        pos_list = []
        for j in range(num_boxes[i]):
            temp = boxes[i][j, :]
            pos_list.append([temp[0] / W[i], temp[2] / W[i], temp[1] / H[i], temp[3] / H[i],
                             ((temp[2] - temp[0]) * (temp[3] - temp[1])) / (W[i] * H[i]), ])

        pos_list = np.array(pos_list)
        out.append(pos_list)
    return np.array(out)


def split_query(query):
    """
    split query
    :param query:
    :return: query and max len
    """
    out = []
    query_len = []
    for i in range(query.shape[0]):
        tem = str(query[i]).split(" ")
        query_len.append(len(tem))
        out.append(tem)
    out = np.array(out)
    query_len = np.array(query_len)
    return out, max(query_len)


def process_decode(opt):
    """
    decode
    :param opt:
    :return:
    """
    # read raw data
    data = pd.read_csv(opt.raw_data_dir+'/'+opt.file_name+'.tsv', sep='\t')
    start = time.time()
    # boxes
    data = process_boxes(data=data)
    # features
    data = process_features(data=data)
    # class_labels
    data = process_class_labels(data=data)
    end = time.time()
    # save
    save_dir = os.path.join(opt.save_dir)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    joblib.dump(data, save_dir + '/' + opt.file_name + '_processed.jl.z')
    # del
    del data
    # print
    print('raw data decoded, time:', (end-start))


def process_query(query):
    """
    query --> query index
    :param query:
    :return: longtensor
    """
    # split
    query, max_len = split_query(query=query)
    # word --> index
    with open('/home/poac/code/Multi_modal_Retrieval/experiments/pretrained_models/bert-base-uncased-vocab.txt',
              mode="r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    word_index = {v: k for k, v in enumerate(lines)}
    # get len list
    len_list = []
    for i in range(len(query)):
        len_list.append(len(query[i]))
    len_list = np.array(len_list)
    # word --> index,  and padding
    query_index = []
    unsee = 0
    for i in range(len(query)):
        tem = query[i]
        tem_ = []
        for j in range(len(tem)):
            if tem[j] not in word_index:
                tem_.append(0)
                unsee += 1
            else:
                tem_.append(word_index[tem[j]])
        # padding 0
        if len_list[i] < max_len:
            for k in range(max_len - len_list[i]):
                tem_.append(0)
        tem_ = np.array(tem_).reshape(1, max_len)
        query_index.append(tem_)
    query_index = np.array(query_index).reshape(-1, max_len)
    print('unsee:', unsee)
    return Variable(torch.LongTensor(query_index))


def process_label(query):
    """
    label --> label index
    :param query:
    :return: longtensor
    """
    # word --> index
    with open('/home/poac/code/Multi_modal_Retrieval/experiments/pretrained_models/bert-base-uncased-vocab.txt',
              mode="r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    word_index = {v: k for k, v in enumerate(lines)}
    # get len list
    len_list = []
    for i in range(len(query)):
        len_list.append(len(query[i]))
    len_list = np.array(len_list)
    max_len = max(len_list)
    # word --> index,  and padding
    query_index = []
    unsee = 0
    for i in range(len(query)):
        tem = query[i]
        tem_ = []
        for j in range(len(tem)):
            if tem[j] not in word_index:
                tem_.append(0)
                unsee += 1
            else:
                tem_.append(word_index[tem[j]])
        # padding 0
        if len_list[i] < max_len:
            for k in range(max_len - len_list[i]):
                tem_.append(0)
        tem_ = np.array(tem_).reshape(1, max_len)
        query_index.append(tem_)
    query_index = np.array(query_index).reshape(-1, max_len)
    print('unsee:', unsee)
    return Variable(torch.LongTensor(query_index))


def get_label(path):
    """
    get label-->id and id-->label
    :param path:
    :return:
    """
    with open(path) as f:
        lines = f.readlines()
        label2id = {l.split('\n')[0].split('\t')[1]:int(l.split('\n')[0].split('\t')[0]) for l in lines[1:]}
        id2label = {int(l.split('\n')[0].split('\t')[0]):l.split('\n')[0].split('\t')[1] for l in lines[1:]}
    return label2id, id2label


def get_label_index(label):
    """
    label --> word --> index
    :param label:
    :return:
    """
    label2id, id2label = get_label(path='/home/poac/AnomalyDetectionDataset/kdd_cup_2020/multimodal_labels.txt')
    # label --> word
    label_word = []
    for i in range(len(label)):
        tem = ''
        for j in range(len(label[i])):
            # repalce by ','
            tem = tem + ',' + id2label[label[i][j]]
            tem = re.sub(" ", ",", tem)
            tem = re.sub('\(', ",", tem)
            tem = re.sub('\)', ",", tem)
            tem = re.sub('\.', ',', tem)
            tem = re.sub('&', ',', tem)
        # split
        tem = tem.split(",")
        # filt none
        tem = [x for x in tem if x != '']
        # tolist
        tem = list(tem)
        label_word.append(tem)

    # word --> index
    label_index = process_label(query=label_word)
    return label_index


def get_iamge_seq(features, boxes, num_boxes):
    """
    get image features and position features
    concate image and position
    :param features:
    :param boxes:
    :return:
    """
    # get max box
    max_box = max(num_boxes)
    features = np.array(features)
    boxes = np.array(boxes)
    image_seq = []
    for i in range(features.shape[0]):
        f_tem = features[i]
        b_tem = boxes[i]
        cat = np.concatenate([f_tem, b_tem], axis=1)
        # 补齐
        if num_boxes[i] < max_box:
            zeros = np.zeros((max_box - num_boxes[i], cat.shape[1]))
            cat = np.concatenate([cat, zeros], axis=0)
        image_seq.append(cat)
    image_seq = np.array(image_seq)
    return image_seq


def get_input(opt):
    """
    get input for net
    :param opt:
    :return:
    """
    # read data
    data = joblib.load(opt.save_dir + '/' + opt.file_name + '_processed.jl.z')
    # query
    query = np.array(data['query'])
    # box feature
    box_feature = np.array(data['features'])
    # box label
    box_label = np.array(data['class_labels'])
    # box position
    boxes = np.array(data['boxes'])
    # image high
    imag_h = np.array(data['image_h'])
    # image width
    imag_w = np.array(data['image_w'])
    # number boxes
    num_boxes = np.array(data['num_boxes'])

    # query --> query index
    query = process_query(query=query)

    # class labels --> class index
    box_label = get_label_index(label=box_label)

    # position
    box_pos = convert_pos(num_boxes=num_boxes, boxes=boxes, H=imag_h, W=imag_w)

    # image and position --> image seq
    image_seq = get_iamge_seq(features=box_feature, boxes=box_pos, num_boxes=num_boxes)
    return query, box_label, image_seq


def get_train_loader(opt):
    """
    将图像序列表示和文本序列表示转成loader
    :param opt:
    :return:
    """
    query, box_label, image_seq = get_input(opt=opt)
    image_seq = torch.from_numpy(image_seq).float()
    # loader
    torch_dataset = Data.TensorDataset(query, box_label, image_seq)
    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=2)
    return loader


if __name__ == '__main__':
    opt = get_args()
    get_train_loader(opt=opt)




