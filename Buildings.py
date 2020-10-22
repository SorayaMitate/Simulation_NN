'''
マップクラス
'''

class Buildings():
    def __init__(self):
        self.pt = [] #送信機座標
        self.tx = [] #送信機のx座標
        self.ty = [] #送信機のy座標
        self.rx = []
        self.ry = []
        self.build_dens = [] #送受信機間の建物の数
        self.build_tdist = [] #送信機からの最初の建物までの距離
        self.buidl_rdist = [] #受信機からの最初の建物までの距離
        self.build_gasonum = []
        self.dist = []
        #self.rssi = []

    '''マップクラスに要素を追加する関数
    入力 : 
        p : 送信機の座標 (タプル)
        dens : 直線上の建物密度
        tdist : 送信機からの最初の建物までの距離
        rdist : 受信機からの最初の建物までの距離
    '''
    def bappend(self, pt, pr, dens, tdist, rdist, dist):
        self.pt.append(pt)
        self.tx.append(pt[0])
        self.ty.append(pt[1])
        self.rx.append(pr[0])
        self.ry.append(pr[1])
        self.build_dens.append(dens)
        self.build_tdist.append(tdist)
        self.buidl_rdist.append(rdist)
        self.dist.append(dist)
        #self.build_gasonum.append(gasonum)
        #self.rssi.append(rssi)

    '''メンバ変数(リスト)を正規化する関数
    '''
    def Normalization(self):
        
        def min_max(l):
            l_min = min(l)
            l_max = max(l)
            return [(i - l_min) / (l_max - l_min) for i in l]

        self.tx_norm = min_max(self.tx)
        self.ty_norm = min_max(self.ty)
        self.rx_norm = min_max(self.rx)
        self.ry_norm = min_max(self.ry)
        self.build_dens_norm = min_max(self.build_dens)
        self.build_tdist_norm = min_max(self.build_tdist)
        self.buidl_rdist_norm = min_max(self.buidl_rdist)
        self.dist_norm = min_max(self.dist)
        #self.build_gasonum = min_max(self.build_gasonum)
        