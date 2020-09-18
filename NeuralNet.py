import numpy as np
import chainer
from sklearn.model_selection import train_test_split
import chainer.links as L
import chainer.functions as F
from chainer import Sequential
from chainer import Sequential

'''ニューラルネットワーククラス
'''
class NeuralNet():
    
    def __init__(self, idim, hdim, odim):
        
        self.n_input = idim    #入力層の次元数
        self.n_hidden = hdim   #隠れ層の数
        self.n_output = odim   #出力層の次元

        #chainer.print_runtime_info()


    '''ニューラルネットワークの準備を行う関数
    入力:
        x: 入力データ
        t: 正解データ
    '''
    def prepare(self, x, t):
        
        x = x.astype('float32')
        t = t.astype('float32')
        indices = np.arange(len(t)) #インデックスリスト

        #データ全体を訓練データ&検証データセットとテストデータセットに分割
        self.x_train_val, self.x_test, self.t_train_val, self.t_test, self.train_indices, self.test_indices \
            = train_test_split(x, t, indices, test_size=0.3)
        #訓練データと検証データセットに分割
        self.x_train, self.x_val, self.t_train, self.t_val = train_test_split(self.x_train_val, self.t_train_val, test_size=0.3)

        #ネットワークの定義
        self.net = Sequential(
            L.Linear(self.n_input, self.n_hidden), F.relu,
            L.Linear(self.n_hidden, self.n_hidden), F.relu,
            L.Linear(self.n_hidden, self.n_output)
        )

        #最適化手法の定義
        self.optimizer = chainer.optimizers.SGD(lr=0.001)

        #ネットワークのパラメータを optimizer 設定
        self.optimizer.setup(self.net)


    '''ネットワークを訓練する関数
    入力: 
        n_epoch: エポック数
        n_batchsize: バッチサイズ数
    '''
    def train(self, n_epoch, n_batchsize):
        iteration = 0

        # ログの保存用
        results_train = {
            'loss': [],
            'accuracy': []
        }
        results_valid = {
            'loss': [],
            'accuracy': []
        }


        for epoch in range(n_epoch):

            # データセット並べ替えた順番を取得
            order = np.random.permutation(range(len(self.x_train)))

            # 各バッチ毎の目的関数の出力と分類精度の保存用
            loss_list = []
            accuracy_list = []

            for i in range(0, len(order), n_batchsize):
                # バッチを準備
                index = order[i:i+n_batchsize]
                x_train_batch = self.x_train[index,:]
                t_train_batch = self.t_train[index]

                # 予測値を出力
                y_train_batch = self.net(x_train_batch)
                #なぜか(x,1)で出力されるわ
                y_train_batch = y_train_batch.T
                y_train_batch = y_train_batch[0]

                # 目的関数を適用し、分類精度を計算
                #print('y_train_batch.shape =',y_train_batch.shape)
                #print('t_train_batch.shape =',t_train_batch.shape)
                loss_train_batch = F.mean_squared_error(y_train_batch, t_train_batch)

                loss_list.append(loss_train_batch.array)

                # 勾配のリセットと勾配の計算
                self.net.cleargrads()
                loss_train_batch.backward()

                # パラメータの更新
                self.optimizer.update()

                # カウントアップ
                iteration += 1

            # 訓練データに対する目的関数の出力と分類精度を集計
            loss_train = np.mean(loss_list)

            # 1エポック終えたら、検証データで評価
            # 検証データで予測値を出力
            with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
                y_val = self.net(self.x_val)

            # 目的関数を適用し、分類精度を計算
            #なぜか(x,1)で出力されるわ
            y_val = y_val.T
            y_val = y_val[0]            
            loss_val = F.mean_squared_error(y_val, self.t_val)

            # 結果の表示
            #print('epoch: {}, iteration: {}, loss (train): {:.4f}, loss (valid): {:.4f}'.format(
            #    epoch, iteration, loss_train, loss_val.array))

            # ログを保存
            results_train['loss'].append(loss_train)
            results_valid['loss'].append(loss_val.array)

        #ネットワークの保存
        chainer.serializers.save_npz('sample.net', self.net)


    '''訓練済みのネットワークを用いた推論を行う関数
    入力:
        input_data: 推論を行いたい入力データ
    出力: 
        y_test: 予測されたデータ
    '''
    def inference(self, input_data):
        loaded_net = Sequential(
            L.Linear(self.n_input, self.n_hidden), F.relu,
            L.Linear(self.n_hidden, self.n_hidden), F.relu,
            L.Linear(self.n_hidden, self.n_output)
        )
        chainer.serializers.load_npz('sample.net', loaded_net)
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            y_test = loaded_net(input_data)

        #self.t_test.data.dtype = np.float32
        y_test = y_test[:].array
        y_test = y_test.T
        y_test = y_test[0]

        return y_test