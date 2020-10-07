import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import skimage.util
from skimage.draw import line
import seaborn as sns
import copy
import chainer
import random
import itertools
from sklearn.model_selection import train_test_split

import Const
import Buildings
import func
import NeuralNet
import REM
import Results
import Kernel

#matplotの設定
plt.style.use('ggplot') 
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)
plt.rcParams["font.size"] = 24


#定数クラスの設定
const = Const.Const()

def simulate(fres):

    pid = os.getpid()

    #ランダムシードをプロセスIDで初期化
    random.seed(pid)

    '''
    画像から建物を抽出する処理
    '''
    #RGBで画像を読み込み
    im = cv2.imread('sample.png')
    rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    #道路と文字を0化 (209or210)
    th, im_mask1 = cv2.threshold(im_gray, const.TH_BUILDINGS, 1, cv2.THRESH_BINARY_INV)
    im_mask1 = 1 - im_mask1
    #白い部分を0化
    th, im_mask2 = cv2.threshold(im_gray, 250, 1, cv2.THRESH_BINARY_INV)
    im = im_gray * im_mask1
    im = im * im_mask2
    th, im = cv2.threshold(im, 180, 255, cv2.THRESH_BINARY_INV)
    im = 255 - im

    '''
    画像をメッシュ化する処理
    '''
    im_shape = im.shape
    #print('im_shape =',im_shape)
    lat_pad_left, lat_pad_right = func.calc_pading_num(im_shape[0], const.AREA[const.AREA_INDEX][4])
    lon_pad_left, lon_pad_right = func.calc_pading_num(im_shape[1], const.AREA[const.AREA_INDEX][5])
    #サイズをマップサイズに合わせるよう周りをゼロパディング
    im_mesh = cv2.copyMakeBorder(im,lat_pad_left,lat_pad_right,lon_pad_left,lon_pad_right\
        ,cv2.BORDER_CONSTANT,0)
    #print('im_mesh.shape =',im_mesh.shape)
    #画像をメッシュサイズで分割
    im_mesh_size = im_mesh.shape
    im_blocks = skimage.util.view_as_blocks(im_mesh, (int(im_mesh_size[0]/const.AREA[const.AREA_INDEX][4]), \
        int(im_mesh_size[1]/const.AREA[const.AREA_INDEX][5]))).copy()
    #print('blocks.shape =',im_blocks.shape)
    #画像表示
    #print('im =',im)
    #plt.imshow(im_mesh)
    #plt.show()

    for node in range(const.N_NODE_MIN, const.N_NODE_MAX, const.M_NODE_EPO):
        
        #結果クラス
        results = Results.Results()

        #print('process id : ', os.getpid())
        print('node =',node)

        for trial in range(const.N_TRIALS):

            #print('trial =',trial)

            '''
            マップの作成
            '''
            rem = REM.REM()
            rem.SpacialColShadowing(const.SIZE_MESH, const.AREA[const.AREA_INDEX][4], const.AREA[const.AREA_INDEX][5],\
                const.VAR, const.COR_DIST)
            
            rem.make_rem(node)

            '''
            建物抽出
            '''
            #mapクラスの定義
            build = Buildings.Buildings()
            l_rssi = []
            l_intf_rssi = []
            for index, row in rem.redata.iterrows(): 

                #正解データの画素座標変換
                pt = (int(row['tx']), int(row['ty']))
                pr = (int(row['rx']), int(row['ry']))
                ptx, pty = func.convert_gaso_to_mcode_forsim(pt, rem.l_index, rem.l_col, im_blocks.shape)
                prx, pry = func.convert_gaso_to_mcode_forsim(pr, rem.l_index, rem.l_col, im_blocks.shape)

                #二値点間の画層群を抽出 (line)
                l_gaso_x, l_gaso_y = line(ptx, pty, prx, pry)
                l_gaso = [im_mesh[l_gaso_x[j]][l_gaso_y[j]] for j in range(len(l_gaso_x))]
                build_num, init_road, last_road = func.count_building(l_gaso, const.TH_GASO)
                build.bappend(pt, pr, build_num, init_road, last_road)

                rssi = (-1)*func.PL(const.FREQ, float(row['dist'])) + float(row['shad'])\
                    - build_num * const.ATEN_BUILD
                l_rssi.append(rssi)

            df_tmp = pd.DataFrame({'rssi':l_rssi})
            rem.redata = pd.concat([rem.redata, df_tmp],axis=1)
            

            '''
            Neural Network
            '''
            build.Normalization()
            #ar_x = np.array([build.tx_norm, build.ty_norm, build.rx_norm, build.ry_norm, \
            #    build.build_dens_norm, build.buidl_rdist_norm, build.build_tdist_norm])
            ar_x = np.array([build.tx_norm, build.ty_norm, build.rx_norm, build.ry_norm, \
                build.build_dens_norm])
            ar_x = ar_x.T
            ar_t = np.array(l_rssi)
            ar_x = ar_x.astype('float32')
            ar_t = ar_t.astype('float32')

            indices = np.arange(len(ar_t)) #インデックスリスト
            #データ全体を訓練データとテストデータセットに分割
            x_train_val, x_test, t_train_val, t_test, train_indices, test_indices \
                = train_test_split(ar_x, ar_t, indices, test_size=0.3)

            #print('ar_t.shape =', ar_t.shape)
            #print('train_indices.shape =', train_indices.shape)
            #print('test_indices.shape =', test_indices.shape)

            nn = NeuralNet.NeuralNet(const.N_INPUT, const.N_HIDDEN, const.N_OUTPUT, pid)
            nn.prepare(x_train_val, x_test, t_train_val, t_test, train_indices, test_indices)
            nn.train(const.N_ITERATION, const.N_EPOCH)

            #NNを用いたRSSIの推定
            ar_nnerssi = nn.inference()
            #print('ar_nnerssi =', ar_nnerssi.shape)

            #カーネル回帰
            kernel = Kernel.Kernel(x_train_val, x_test, t_train_val, t_test, train_indices, test_indices)
            kernel.modeling()
            kernel.prediction()
            #print('kernel.ar_y =',kernel.ar_y.shape)


            #idwを用いた内挿
            for i in range(len(test_indices)):

                tx = rem.redata.at[test_indices[i],'tx']
                ty = rem.redata.at[test_indices[i],'ty']
                rx = rem.redata.at[test_indices[i],'rx']
                ry = rem.redata.at[test_indices[i],'ry']
                #ran = func.calc_range_sim(3, rx, ry)
                
                df_tmp = rem.redata[(rem.redata['tx']==tx)&(rem.redata['ty']==ty)]
                
                #rem.out_map(df_tmp, 'rx', 'ry', 'rssi')

                #rssi補間
                c = 'rssi'
                l_dist, l_rssi = func.interpolation_sim(df_tmp, c, rx, ry, 3)
    
                if len(l_dist) > 0:
                    results.rappend(ar_t[test_indices[i]], func.idw(l_rssi, l_dist), ar_nnerssi[i], kernel.ar_y[i])
                else:
                    pass

        #結果の格納
        results.calc_error()
        fres[const.N_INDEX.index(node)]['eidw'].append(results.rssi_error)
        fres[const.N_INDEX.index(node)]['enn'].append(results.nnrssi_error)
        fres[const.N_INDEX.index(node)]['kernel'].append(results.krssi_error)


'''main関数
'''
from multiprocessing import Manager, Value, Process
if __name__ == "__main__":
    with Manager() as manager:

        res = [{'eidw':manager.list(), 'enn':manager.list(), 'kernel':manager.list()} \
            for i in range(const.N_NODE_MIN, const.N_NODE_MAX, const.M_NODE_EPO)]

        l_process = []
        for i in range(const.N_CPU):
            process = Process(target=simulate, args=[res])
            process.start()
            l_process.append(process)

        for process in l_process:
            process.join()

        for node in range(const.N_NODE_MIN, const.N_NODE_MAX, const.M_NODE_EPO):
            tmp = {'eidw':[],'enn':[], 'kernel':[]}
            tmp['eidw'] = list(itertools.chain.from_iterable(res[const.N_INDEX.index(node)]['eidw'][:]))
            tmp['enn'] = list(itertools.chain.from_iterable(res[const.N_INDEX.index(node)]['enn'][:]))
            tmp['kernel'] = list(itertools.chain.from_iterable(res[const.N_INDEX.index(node)]['kernel'][:]))
            df = pd.DataFrame(tmp.values(), index=tmp.keys()).T
            
            name = str(node) + '_error.csv'
            df.to_csv(name)
