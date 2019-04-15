# coding: utf-8

import load
import network
import numpy as np
import os
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.functions.loss.softmax_cross_entropy import softmax_cross_entropy

import sys
import os
import shutil


TEST_DATA_SIZE = 100000  # テストデータのサイズ
MINIBATCH_SIZE = 100  # ミニバッチサイズ
EVALUATION_SIZE = 1000  # 評価のときのデータサイズ

# ***--- add by ryuml(20190416)
START_EPOCH = 110   # 読み込む学習済みモデルファイルのエポック番号
SAVE_TMG = 10   # 学習済みモデルを保存するタイミング
LOSS_LOG_BASE = 'loss_log_epo'   # ロスのログファイル
EPO_MAX = 100   # 最大epoch数
BAR_MAX = 30   # 進捗バーの最大文字数


def main():
    # データの読み込み・加工
    if os.path.isfile('states.npy') and os.path.isfile('actions.npy'):
        states = np.load('states.npy')
        actions = np.load('actions.npy')
    else:
        #download()  # ファイルダウンロード
        load.download()  # ファイルダウンロード
        states, actions = load.load_and_save()  # データの読み込み・加工・保存
    
    test_x = states[:TEST_DATA_SIZE].copy()  # ランダムに並び替え済み
    train_x = states[TEST_DATA_SIZE:].copy()
    del states  # メモリがもったいないので強制解放
    test_y = actions[:TEST_DATA_SIZE].copy()
    train_y = actions[TEST_DATA_SIZE:].copy()
    del actions
    
    
    model = L.Classifier(network.AgentNet(), lossfun=softmax_cross_entropy)
    #if os.path.isfile('model.npz'):  # モデル読み込み
    model_name = 'model_epo{:04}.npz'.format(START_EPOCH)
    if os.path.isfile(model_name):  # モデル読み込み
        #serializers.load_npz('model.npz', model)
        serializers.load_npz(model_name, model)
    
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    
    # ログファイル初期化
    log_start_line = 'epoch,loss\n'
    f_all = open(LOSS_LOG_BASE + '{}to{}.csv'.format(START_EPOCH + 1, EPO_MAX), 'w')
    with f_all:
        f_all.write(log_start_line)
    
    # ログ保存用のリスト
    logs = []
    
    # 学習ループ
    #for epoch in range(100):
    for epoch in range(EPO_MAX):
        for i in range(100):
            if i == 0:
                percent = 0.0
            else:
                percent = i / 100.0 * 100.0
            
            sys.stderr.write('\r' + get_str_bar(percent))
            sys.stderr.flush()
            index = np.random.choice(train_x.shape[0], MINIBATCH_SIZE, replace=False)
            x = chainer.Variable(train_x[index].reshape(MINIBATCH_SIZE, 1, 8, 8).astype(np.float32))
            t = chainer.Variable(train_y[index].astype(np.int32))
            optimizer.update(model, x, t)
        # 進捗バー末端部分表示
        percent = (i + 1) / 100.0
        sys.stderr.write('\r' + get_str_bar(percent) + '\n\n')
        sys.stderr.flush()
        
        # 評価
        index = np.random.choice(test_x.shape[0], EVALUATION_SIZE, replace=False)
        x = chainer.Variable(test_x[index].reshape(EVALUATION_SIZE, 1, 8, 8).astype(np.float32))
        t = chainer.Variable(test_y[index].astype(np.int32))
        tmp_loss = model(x, t).data
        
        #print('epoch :', epoch, '  loss :', tmp_loss)
        print('> Epoch: {} / {},  Loss: {}\n\n'.format(epoch, EPO_MAX, tmp_loss))
        
        
        f_all = open(LOSS_LOG_BASE + '{}to{}.csv'.format(START_EPOCH + 1, EPO_MAX), 'a')
        with f_all:
            f_all.write('{},{}\n'.format(epoch, tmp_loss))
        
        # ログの保存
        logs.append(tmp_loss)
        
        # SAVE_TMGでログファイルとモデルを保存
        if (epoch + 1) % SAVE_TMG == 0:
            #serializers.save_npz('model.npz', model)  # モデル保存
            model_f_name = 'model_epo{:04}.npz'.format(epoch)
            serializers.save_npz(model_f_name, model)
            print('\n  -> Saved model file: [{}]\n'.format(model_f_name))
            
            loss_tmp_file = LOSS_LOG_BASE + '{}to{}.csv'.format(START_EPOCH, epoch)
            with open(loss_tmp_file, 'w') as f_tmp:
                f_tmp.write('epoch,loss\n')
                for log_i in range(len(logs)):
                    f_tmp.write('{},{}\n'.format(log_i, logs[log_i]))
            print('  -> Wrote loss log file: [{}]\n'.format(loss_tmp_file))
        
        # モデルファイルの一時保存	
        serializers.save_npz('model_backup.npz', model)
        
        # ログファイルの出力
        f_all = open(LOSS_LOG_BASE + '{}to{}.csv'.format(START_EPOCH + 1, EPO_MAX), 'a')
        with f_all:
            f_all.write('{},{}\n'.format(epoch, tmp_loss))
    
    print('\n>>> Finished.\n')


def get_str_bar(percent):
    str_bar = ''
    
    percent *= 1.0   # エラー対処
    # バーの大きさ（文字数）の取得
    str_len = int(BAR_MAX * (percent / 100.0))
    
    if str_len == 0:
        str_bar = '[>' + ' ' * (BAR_MAX - 1) + ']'
    else:
        str_bar = '[' + '=' * (str_len - 1) + '>' + ' ' * (BAR_MAX - str_len + 1) + ']'
    
    str_bar += ' - ({:.2f} %)'.format(percent)

    return str_bar


if __name__ == '__main__':
	main()
