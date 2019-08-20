from __future__ import division

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from tqdm import tqdm
from argparse import ArgumentParser
import gensim
import os
import sys
from pathlib2 import Path
import numpy as np
from timeit import default_timer as timer

import utils
from utils import maybe_cuda
from choiloader import collate_fn
from data_loader import SegTextDataSet
import accuracy

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

logger = utils.setup_logger(__name__, './logs/test_accuracy.log')


def softmax(x):
    max_each_row = np.max(x, axis=1, keepdims=True)
    exps = np.exp(x - max_each_row)
    sums = np.sum(exps, axis=1, keepdims=True)
    return exps / sums

def getSegmentsFolders(path):

    ret_folders = []
    folders = [o for o in os.listdir(path) if os.path.isdir(os.path.join(path, o))]
    for folder in folders:
        if folder.__contains__("-"):
            ret_folders.append(os.path.join(path,folder))
    return ret_folders


def main(args):
    start = timer()

    sys.path.append(str(Path(__file__).parent))

    utils.read_config_file(args.config)
    utils.config.update(args.__dict__)

    logger.debug('Running with config %s', utils.config)
    print ('Running with threshold: ' + str(args.seg_threshold))
    preds_stats = utils.predictions_analysis()

    word2vec = gensim.models.KeyedVectors.load_word2vec_format(utils.config['word2vecfile'], binary=True)

    word2vec_done = timer()
    print ('Loading word2vec ellapsed: ' + str(word2vec_done - start) + ' seconds')

    dataset_folders = [Path(utils.config['segdataset']) / 'test']
    if (args.seg_folder):
        dataset_folders = []
        dataset_folders.append(Path(utils.config['segdataset']) / args.seg_folder)
    print ('running on segment data')


    with open(args.model, 'rb') as f:
        model = torch.load(f)

    model = maybe_cuda(model)
    model.eval()

    for dataset_path in dataset_folders:

        if (args.seg_folder):
            dataset = SegTextDataSet(dataset_path, word2vec, train=False, folder=True, high_granularity=False, max_token_num=args.max_token_num)
        else :
            dataset = SegTextDataSet(dataset_path, word2vec, train=False, high_granularity=False, max_token_num=args.max_token_num)

        dl = DataLoader(dataset, batch_size=args.bs, collate_fn=collate_fn, shuffle=False)



        with tqdm(desc='Testing', total=len(dl)) as pbar:
            total_accurate = 0
            total_count = 0
            total_loss = float(0)
            acc =  accuracy.Accuracy()

            for i, (data, targets, paths) in enumerate(dl):
                if i == args.stop_after:
                    break

                pbar.update()
                output = model(data)
                targets_var = Variable(maybe_cuda(torch.cat(targets, 0), args.cuda), requires_grad=False)
                batch_loss = model.criterion(output, targets_var).item()
                output_prob = softmax(output.data.cpu().numpy())
                output_seg = output_prob[:, 1] > args.seg_threshold
                #print i, pid, path
                target_seg = targets_var.data.cpu().numpy()
                #print i, "output_seg", output_seg
                #print i, "target_seg", target_seg
                batch_accurate = (output_seg == target_seg).sum()
                total_accurate += batch_accurate
                total_count += len(target_seg)
                total_loss += batch_loss
                preds_stats.add(output_seg,target_seg)

                current_target_idx = 0
                for k, t in enumerate(targets):
                    document_sentence_count = len(t)
                    sentences_length = [s.size()[0] for s in data[k]] if args.calc_word else None
                    to_idx = int(current_target_idx + document_sentence_count)
                    h = output_seg[current_target_idx: to_idx]

                    # hypothesis and targets are missing classification of last sentence, and therefore we will add
                    # 1 for both
                    h = np.append(h, [1])
                    t = np.append(t.cpu().numpy(), [1])

                    acc.update(h,t, sentences_length=sentences_length)

                    current_target_idx = to_idx

                logger.debug('Batch %s - error %7.4f, Accuracy: %7.4f', i, batch_loss, batch_accurate / len(target_seg))
                #pbar.set_description('Testing, Accuracy={:.4}'.format(batch_accurate / len(target_seg)))

        average_loss = total_loss / len(dl)
        average_accuracy = total_accurate / total_count
        calculated_pk, _ = acc.calc_accuracy()

        logger.info('Finished testing.')
        logger.info('Average loss: %s', average_loss)
        logger.info('Average accuracy: %s', average_accuracy)
        logger.info('Pk: {:.4}.'.format(calculated_pk))
        logger.info('F1: {:.4}.'.format(preds_stats.get_f1()))


        end = timer()
        print ('Seconds to execute to whole flow: ' + str(end - start))



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--test', help='Test mode? (e.g fake word2vec)', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=1)
    parser.add_argument('--mark', help='Mark index', type=int, default=1)
    parser.add_argument('--max_token_num', help='Max sentense token num', type=int, default=0)
    parser.add_argument('--model', help='Model to run - will import and run', required=True)
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    parser.add_argument('--config', help='Path to config.json', default='./conf/config.json')
    parser.add_argument('--seg_folder', help='path to folder which contains seg documents')
    parser.add_argument('--seg_threshold', help='Threshold for binary classificetion', type=float, default=0.4)
    parser.add_argument('--calc_word', help='Whether to calc P_K by word', action='store_true')



    main(parser.parse_args())
