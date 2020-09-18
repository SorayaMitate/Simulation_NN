'''
電波環境マップクラス
'''

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import GeoTrans
import func
import Const

class REM():

    '''初期化関数
    '''
    def __init__(self):
        pass

    '''Serverからの出力 (csv) を整形
    入力 : csvファイル
    出力 : 整形されたデータ (データフレーム)
    '''
    def modify_csv(self, c):
        #data = pd.read_csv('sta_20190821_101126.csv')
        #data = data[data['receive_n_lon']>130].reset_index(drop=True)
        data = pd.read_csv('data.csv',index_col=0)
        #print(data)

        target = list(data['trans_mesh_code'].unique())

        tra_x = []
        tra_y = []
        rec_x = []
        rec_y = []
        rssi = []
        tra_slat = []
        tra_slon = []
        tra_nlat = []
        tra_nlon = []
        rec_slat = []
        rec_slon = []
        rec_nlat = []
        rec_nlon = []

        for code in target:
            tmp = code.split('-')
            x, y = func.make_code(tmp)
            for i, row in data[data['trans_mesh_code']==code].iterrows():
                tmp1 = row['receive_mesh_code'].split('-')
                x_tmp, y_tmp = func.make_code(tmp1)
                if (x_tmp != 0) and (y_tmp != 0) and (row['rssi_avg'] != 0.000000e+00):
                    tra_x.append(x)
                    tra_y.append(y)
                    rec_x.append(x_tmp)
                    rec_y.append(y_tmp)
                    rssi.append(10 * np.log10(row['rssi_avg']))
                    tra_slat.append(row['trans_s_lat'])
                    tra_slon.append(row['trans_s_lon'])
                    tra_nlat.append(row['trans_n_lat'])
                    tra_nlon.append(row['trans_n_lon'])
                    rec_slat.append(row['receive_s_lat'])
                    rec_slon.append(row['receive_s_lon'])
                    rec_nlat.append(row['receive_n_lat'])
                    rec_nlon.append(row['receive_n_lon'])

        self.data = pd.DataFrame({
            'tx':tra_x,
            'ty':tra_y,
            'rx':rec_x,
            'ry':rec_y,
            'rssi':rssi,
            'tra_slat':tra_slat,
            'tra_slat':tra_slat,
            'tra_nlat':tra_nlat,
            'tra_nlon':tra_nlon,
            'rec_slat':rec_slat,
            'rec_slon':rec_slon,
            'rec_nlat':rec_nlat,
            'rec_nlon':rec_nlon
        })

    
    '''対象得エリアのデータ抽出と送信機リストの作成
    入力 : 範囲 (タプル)
    出力 : 範囲のデータ (データフレーム)
    '''
    def extract_area(self, area):
        self.redata = self.data[(area[0]<=self.data['tx'])&(self.data['tx']<=area[1])\
            &(area[2]<=self.data['ty'])&(self.data['ty']<=area[3])].copy()
        self.redata = self.redata.reset_index(drop=True)
    
        #送信機リストの作成
        tra_x = list(self.redata['tx'])
        tra_y = list(self.redata['ty'])
        self.tx_list = [(tra_x[i], tra_y[i]) for i in range(len(tra_x))]
        #重複の削除
        self.tx_list = list(set(self.tx_list))
        self.tx_list = [(i[0]*(-1), i[1]) for i in self.tx_list]


    '''対象エリアの緯度経度計算
    '''
    def calc_latlong(self):
        tky2wgs = GeoTrans.GeoTrans(4301,4326)
        self.nlat = self.redata[self.redata['rx']==self.redata['rx'].max()]['rec_nlat'].unique()[0]
        self.slat = self.redata[self.redata['rx']==self.redata['rx'].min()]['rec_slat'].unique()[0]
        self.nlon = self.redata[self.redata['ry']==self.redata['ry'].max()]['rec_nlon'].unique()[0]
        self.slon = self.redata[self.redata['ry']==self.redata['ry'].min()]['rec_slon'].unique()[0]
        print('(nlon,nlat) =',tky2wgs.transform(self.nlon,self.nlat))
        print('(slon,slat) =',tky2wgs.transform(self.slon,self.slat))

    '''
    REMのメッシュ座標とインデックスの対応関係の作成
    入力 : 範囲 (タプル)
    出力 : マップのインデックス
    '''
    def make_index(self, area):
        rx = list(self.redata['rx'])
        ry = list(self.redata['ry'])
        rssi = [np.nan for i in range(len(rx))]
        #print('The number of mesh =', len(rssi))
        for i in range(area[0],area[1]):
            if (str(i)[-1] == '0') or (str(i)[-1] == '1'):
                for j in range(area[2],area[3]):
                    if (str(j)[-1] == '0') or (str(j)[-1] == '1'):
                        #REM内の存在しないメッシュコードの補充
                        #どちらか一方でも存在しないメッシュコードが存在しないとき補充
                        if (i in rx) & (j in ry) == False:
                            rx.append(i)
                            ry.append(j)
                            rssi.append(np.nan)

        tmp_df = pd.DataFrame({
            'rx':rx,
            'ry':ry,
            'rssi':rssi
        })


        for i, row in tmp_df.iterrows():
            tmp_df.at[i, 'rx'] = row['rx']*(-1)
        tmp_pivot = tmp_df.pivot_table(index='rx', columns='ry', values='rssi', dropna= False)
        print('tmp_pivot.shape =',tmp_pivot.shape)

        self.l_index = list(tmp_pivot.index)
        self.l_col = list(tmp_pivot.columns)


    '''
    空間相関シャドウイング分布の生成
    入力: メッシュサイズ, メッシュ範囲X, メッシュ範囲Y, 各メッシュの正規分布の分散, 相関距離
    出力 : 空間相関をもつシャドウィング
    '''
    def SpacialColShadowing(self, size, XSIZE, YSIZE, var, dcol):

        #2地点間の相関係数を計算する関数
        #入力 : 2メッシュ間の距離
        def calc_SpatialCorrelation(d, dcol):
            return np.exp((-1)*d*np.log(2)/dcol)

        X = np.arange(0, XSIZE)
        Y = np.arange(0, YSIZE)
        XX, YY = np.meshgrid(X,Y)

        self.l_index = list(X)
        self.l_col = list(Y)

        #二次元配列を一次元に
        X = XX.flatten()
        Y = YY.flatten()
        leng = len(X)
        S = np.zeros((leng,leng))

        #共分散行列の計算
        for i in range(leng):
            for j in range(leng):
                tmp = func.calc_dist(X[i],Y[i],X[j],Y[j])
                S[i][j] = calc_SpatialCorrelation(tmp, dcol)*(var**2)
        
        #コレスキー分解
        L = np.linalg.cholesky(S)

        #共分散行列の計算
        w = np.random.standard_normal(leng)
        M = np.dot(L, w)

        self.mesh = pd.DataFrame({
            'X':X,
            'Y':Y,
            'SHADOWING':M
        })


    '''マップを作成する関数
    入力: 
        n: ノード数
    出力:
        アドホックネットワークのREM
    '''
    def make_rem(self, n):
        
        self.mesh = self.mesh.sample(n=n)
        self.mesh = self.mesh.reset_index(drop=True)

        X = self.mesh['X'].values
        Y = self.mesh['Y'].values
        S = self.mesh['SHADOWING'].values

        XY = [(X[i], Y[i]) for i in range(n)]
        
        XXYY = np.tile(XY, n).flatten().tolist()
        XXYY = [j for i in range(n) for j in XY]
        XYXY = [i for i in XY for j in range(n)]
        SS = [i for i in S for j in range(n)]
        DIST = [func.calc_dist(XXYY[i][0],XXYY[i][1],XYXY[i][0],XYXY[i][1]) \
            for i in range(len(XXYY))]

        self.redata = pd.DataFrame({
            'tx': [i[0] for i in XXYY],
            'ty': [i[1] for i in XXYY],
            'rx': [i[0] for i in XYXY],
            'ry': [i[1] for i in XYXY],
            'shad': [i for i in SS],
            'dist': DIST
        })

        self.redata = self.redata.drop(self.redata[self.redata['dist']<=0.0].index).reset_index(drop=True)

    
    '''干渉を与えるノードを決定する関数
    入力: 
        n: 干渉を与えるノード数
    '''
    def make_interference(self, n):
        self.intf_mesh = self.mesh.sample(n=n)
        xtmp = self.intf_mesh['X'].values
        ytmp = self.intf_mesh['Y'].values
        self.intf_p = [(xtmp[i], ytmp[i]) for i in range(len(xtmp))]

    def out_map(self, df, x, y, v):

        const = Const.Const()

        rx = list(df[x])
        ry = list(df[y])
        rssi = list(df[v])
        #print('The number of mesh =', len(rssi))
        for i in range(0, const.AREA[const.AREA_INDEX][4]):
            for j in range(0, const.AREA[const.AREA_INDEX][5]):
                #REM内の存在しないメッシュコードの補充
                #どちらか一方でも存在しないメッシュコードが存在しないとき補充
                if (i in rx) & (j in ry) == False:
                    rx.append(i)
                    ry.append(j)
                    rssi.append(0.0)

        tmp = pd.DataFrame({
            'X':rx,
            'Y':ry,
            'rssi':rssi
        })

        #matplotの設定
        plt.style.use('ggplot') 
        font = {'family' : 'meiryo'}
        matplotlib.rc('font', **font)
        plt.rcParams["font.size"] = 24

        piv = tmp.pivot_table(index='X', columns='Y', values='rssi', dropna= False)
        print('piv.shape =',piv.shape)

        plt.figure()
        sns.heatmap(piv,cmap = 'jet',linewidths=0.5, linecolor='Black',square = True,\
            cbar_kws={"orientation": "horizontal"})
        plt.xlabel('latitude')
        plt.ylabel('longitude')
        plt.show()
