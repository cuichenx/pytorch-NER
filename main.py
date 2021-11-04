import random
import torch
import numpy as np
import argparse
import os
from utils import my_collate_fn, lr_decay, write_result
import time
from dataset import SCH_ElaborateExpressions
from torch.utils.data import DataLoader
from model import NamedEntityRecog
import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
from train import train_model, evaluate
import wandb

seed_num = 42
random.seed(seed_num)
torch.manual_seed(seed_num)
np.random.seed(seed_num)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Named Entity Recognition Model')
    parser.add_argument('--word_embed_dim', type=int, default=100)
    parser.add_argument('--word_hidden_dim', type=int, default=100)
    parser.add_argument('--char_embedding_dim', type=int, default=30)
    parser.add_argument('--char_hidden_dim', type=int, default=50)
    parser.add_argument('--dropout', type=float, default=0.5)
    # parser.add_argument('--pretrain_embed_path', default='data/glove.6B.100d.txt')
    parser.add_argument('--savedir', default='model/')
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--optimizer', default='sgd')
    parser.add_argument('--lr', type=float, default=0.015)
    parser.add_argument('--feature_extractor', choices=['lstm', 'cnn'], default='cnn')
    parser.add_argument('--use_char', dest='use_char', action='store_true')
    parser.add_argument('--char_feature_extractor', choices=['mlp', 'cnn'], default='cnn')
    parser.add_argument('--no_char', dest='use_char', action='store_false')
    parser.add_argument('--data_path', default='data/data.pth')
    parser.add_argument('--split_path', default='data/split_naive_1.pth')
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--number_normalized', type=bool, default=True)
    parser.add_argument('--use_crf', dest='use_crf', action='store_true')
    parser.add_argument('--no_crf', dest='use_crf', action='store_false')
    parser.add_argument('--switch_prob', type=float, default=0)
    parser.add_argument('--train_pos_ratio', type=float, default=0.5)
    parser.add_argument('--val_pos_ratio', type=float, default=0.5)
    parser.add_argument('--wandb_name', type=str, default='')
    parser.add_argument('--grouped_swap', action='store_true')
    parser.add_argument('--do_attested_classification', action='store_true')


    args = parser.parse_args()
    use_gpu = torch.cuda.is_available()
    print('visible devices:', f"GPU {os.environ.get('CUDA_VISIBLE_DEVICES', 0)}" if use_gpu else 'Not using GPU')
    print('feature extractor:', args.feature_extractor)
    print('use_char:', args.char_feature_extractor if args.use_char else False)
    print('use_crf:', args.use_crf)

    if args.wandb_name:
        wandb.init(project="hmong-seq-tagging", entity="cuichenx", name=args.wandb_name)
        wandb.config.update(args)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    eval_path = "evaluation"
    eval_temp = os.path.join(eval_path, f"temp_{args.wandb_name}")
    eval_script = os.path.join(eval_path, "conlleval")

    if not os.path.isfile(eval_script):
        raise Exception('CoNLL evaluation script not found at "%s"' % eval_script)
    if not os.path.exists(eval_temp):
        os.makedirs(eval_temp)

    pred_file = eval_temp + '/pred.txt'
    score_file = eval_temp + '/score.txt'
    args.savedir = args.savedir + args.wandb_name
    os.makedirs(args.savedir, exist_ok=True)
    model_name = args.savedir + '/' + 'best_model'

    # emb_begin = time.time()
    pretrain_word_embedding = None #build_pretrain_embedding(args.pretrain_embed_path, word_vocab, args.word_embed_dim)
    # emb_end = time.time()
    # emb_min = (emb_end - emb_begin) % 3600 // 60
    # print('build pretrain embed cost {}m'.format(emb_min))

    split = torch.load(args.split_path)
    if args.grouped_swap:
        train_elabs_swap, valtest_elabs_swap = split['train_elabs_swap'], split['valtest_elabs_swap']
    else:
        train_elabs_swap, valtest_elabs_swap = None, None

    kwargs = {"switch_prob": args.switch_prob, "do_attested_classification": args.do_attested_classification}
    train_dataset = SCH_ElaborateExpressions(args.data_path, split['train'], positive_ratio=args.train_pos_ratio,
                                             grouped_swap_elabs=train_elabs_swap, is_test=False, **kwargs)
    dev_dataset = SCH_ElaborateExpressions(args.data_path, split['val'], positive_ratio=args.val_pos_ratio,
                                           grouped_swap_elabs=valtest_elabs_swap, is_test=True, **kwargs)
    test_dataset1 = SCH_ElaborateExpressions(args.data_path, split['test'], positive_ratio=0.5,
                                             grouped_swap_elabs=valtest_elabs_swap, is_test=True, **kwargs)
    test_dataset2 = SCH_ElaborateExpressions(args.data_path, split['test'], positive_ratio=-1,
                                             grouped_swap_elabs=valtest_elabs_swap, is_test=True, **kwargs)
    i2w = train_dataset.i2w
    i2l = train_dataset.i2l
    i2c = train_dataset.i2c
    # word_vocab = WordVocabulary(args.data_path)
    # label_vocab = LabelVocabulary(args.data_path, args.do_attested_classification)
    # alphabet = Alphabet(args.data_path)


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=my_collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader1 = DataLoader(test_dataset1, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)
    test_dataloader2 = DataLoader(test_dataset2, batch_size=args.batch_size, shuffle=False, collate_fn=my_collate_fn)

    model = NamedEntityRecog(len(i2w), args.word_embed_dim, args.word_hidden_dim, len(i2c),
                             args.char_embedding_dim, args.char_hidden_dim,
                             args.feature_extractor, len(i2l), args.dropout,
                             pretrain_embed=pretrain_word_embedding, use_char=args.use_char, use_crf=args.use_crf,
                             use_gpu=use_gpu, char_feature_extractor=args.char_feature_extractor)
    if args.wandb_name:
        wandb.watch(model)
    if use_gpu:
        model = model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)  #  performs worse

    train_begin = time.time()
    print('train begin', '-' * 50)
    print()
    print()

    writer = None # SummaryWriter('log')
    batch_num = -1
    best_f1 = -1
    early_stop = 0

    for epoch in range(args.epochs):
        epoch_begin = time.time()
        print('train {}/{} epoch'.format(epoch + 1, args.epochs))
        optimizer = lr_decay(optimizer, epoch, 0.05, args.lr)
        batch_num = train_model(train_dataloader, model, optimizer, batch_num, writer, use_gpu)
        new_f1 = evaluate(dev_dataloader, model, i2w, i2l, pred_file, score_file, eval_script, use_gpu,
                          prefix='val/' if args.wandb_name else '')
        print('f1 is {} at {}th epoch on dev set'.format(new_f1, epoch + 1))
        if new_f1 > best_f1:
            best_f1 = new_f1
            print('new best f1 on dev set:', best_f1)
            early_stop = 0
            torch.save(model.state_dict(), model_name)
        else:
            early_stop += 1

        epoch_end = time.time()
        cost_time = epoch_end - epoch_begin
        print('train {}th epoch cost {}m {}s'.format(epoch + 1, int(cost_time / 60), int(cost_time % 60)))
        print()

        if early_stop > args.patience:
            print('early stop')
            break

    train_end = time.time()
    train_cost = train_end - train_begin
    hour = int(train_cost / 3600)
    min = int((train_cost % 3600) / 60)
    second = int(train_cost % 3600 % 60)
    print()
    print()
    print('train end', '-' * 50)
    print('train total cost {}h {}m {}s'.format(hour, min, second))
    print('-' * 50)
    print('feature extractor:', args.feature_extractor)
    print('use_char:', args.use_char)
    print('use_crf:', args.use_crf)
    print('-' * 50)
    model.load_state_dict(torch.load(model_name))
    test_F1 = evaluate(test_dataloader1, model, i2w, i2l, pred_file, score_file, eval_script, use_gpu,
                       prefix='test_0.5pos/' if args.wandb_name else '')
    write_result(f'test F1 on test set with 0.5 pos ratio: {test_F1}', also_print=True)
    test_F1_full = evaluate(test_dataloader2, model, i2w, i2l, pred_file, score_file, eval_script, use_gpu,
                            prefix='test_full/' if args.wandb_name else '')
    write_result(f'test F1 on test set with all test data: {test_F1_full}', also_print=True)

