import argparse


def get_args():
    """
    parameters
    :return:
    """
    parser = argparse.ArgumentParser(description='BlockDrop Training')
    parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--image_lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--text_lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--margin', type=float, default=12, help='margin of triplet loss')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--max_epochs', type=int, default=10, help='total epochs to run')
    parser.add_argument('--hashbits', type=int, default=32)
    parser.add_argument('--cv_dir', default='checkpoints')
    parser.add_argument('--max_voca', type=int, default=30522, help='the number of voca')

    parser.add_argument('--root_dir',type=str, default='/home/disk1/zhaoyuying/dataset/iapr-tc12_255labels')
    parser.add_argument('--image_dir', type=str, default='/home/disk1/zhaoyuying/dataset/iapr-tc12_255labels/JPEGImages')
    parser.add_argument('--text_dir', type=str, default='/home/disk1/zhaoyuying/dataset/iapr-tc12_255labels/annotations')
    parser.add_argument('--outf', type=str, default='/home/poac/code/Multi_modal_Retrieval/experiments')
    parser.add_argument('--save_dir', type=str, default='/home/poac/AnomalyDetectionDataset/kdd_cup_2020/processed',
                        help='save dir of processed data')
    parser.add_argument('--file_name', type=str, default='train_sample', help='raw data file name')
    parser.add_argument('--raw_data_dir', type=str, default='/home/poac/AnomalyDetectionDataset/kdd_cup_2020',
                        help='the dir of raw data')
    #
    args = parser.parse_args()

    return args