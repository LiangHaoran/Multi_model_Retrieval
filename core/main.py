import os
import torch.optim as optim
from torch.autograd import Variable
from transformers import BertTokenizer, BertModel, BertForMaskedLM, BertConfig
from option import get_args
from iapr_utils import *
from utils import *
from model import ImageNet,TextNet, EmbNet
import torch
from data import get_train_loader
from tqdm import tqdm
import time

def train(opt, epoch):
    """
    train on epoch
    :param opt:
    :param epoch:
    :return:
    """
    # define net
    imageNet.train()
    textNet.train()
    embNet.train()

    accum_loss = 0
    for batch_idx, (query, box_label, image_seq) in enumerate(train_loader):
        start = time.time()
        # directly get image code
        query, box_label, image_seq = Variable(query).cuda(), Variable(box_label).cuda(), Variable(image_seq).cuda()
        image_seq = imageNet.forward(image_seq)   # (batch, 1024)

        # query and box_label, share net
        query, box_label = embNet(query, box_label)

        # triplet loss
        imgae_triplet_loss_, text_triplet_loss_, \
        imgae_text_triplet_loss_, text_image_triplet_loss_, \
        len_triplets_ = CrossModel_triplet_loss(image_seq, query, box_label, opt.margin)

        loss = imgae_triplet_loss_ + text_triplet_loss_ + imgae_text_triplet_loss_ + text_image_triplet_loss_

        if len_triplets_ > 0:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            accum_loss = loss.data.item()

        if batch_idx % 20 == 0 and batch_idx != 0:
            end = time.time()
            print("batch: %d, epoch: %d, accum_loss: %.6f, time: %.6f" % (batch_idx, epoch, accum_loss, (end-start)))

    # save model
    # save models
    model_dir = os.path.join(opt.outf, 'models')
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    torch.save(textNet, model_dir + '/textNet.pkl')


def test(args,epoch):
    imageNet.eval()
    textNet.eval()

    tst_image_binary, tst_text_binary, tst_label, tst_time = compute_result_CrossModel(test_loader, imageNet, textNet,tokenizer)
    db_image_binary, db_text_binary, db_label, db_time = compute_result_CrossModel(db_loader, imageNet,  textNet,tokenizer)
    # print('test_codes_time = %.6f, db_codes_time = %.6f'%(tst_time ,db_time))

    it_mAP = compute_mAP_MultiLabels(db_text_binary, tst_image_binary, db_label, tst_label)
    ti_mAP = compute_mAP_MultiLabels(db_image_binary, tst_text_binary, db_label, tst_label)
    print("epoch: %d, retrieval it_mAP: %.6f, retrieval ti_mAP: %.6f" %(epoch, it_mAP, ti_mAP))

    # f = open('result/' + args.cv_dir + 'mAP.txt', 'a')
    # f.write('Epoch:'+str(epoch)+':  it_mAP = '+str(it_mAP)+', ti_mAP = '+str(ti_mAP)+'\n')
    # f.close()


if __name__ == '__main__':
    # setting
    opt = get_args()
    start_epoch = 0
    total_tst_time = 0
    test_cnt = 0
    loss_print = 0
    MODEL_UPDATE_ITER = 0

    # get loader
    train_loader = get_train_loader(opt=opt)

    # define net
    imageNet = ImageNet()
    imageNet.cuda()
    # text net
    tokenizer = BertTokenizer.from_pretrained('/home/poac/code/Multi_modal_Retrieval/experiments/pretrained_models/bert-base-uncased-vocab.txt')
    textNet = TextNet(code_length=opt.hashbits)
    textNet.cuda()

    # embedding net
    embNet = EmbNet(opt)
    embNet.cuda()

    optimizer = optim.Adam(list(imageNet.parameters())+list(textNet.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)  #+list(textExtractor.parameters())

    # train and test on 10 epoch
    for epoch in tqdm(range(start_epoch, start_epoch+opt.max_epochs+1)):
        train(opt, epoch)
        if epoch % 1 == 0:
            test(opt, epoch)


