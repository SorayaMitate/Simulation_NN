'''
リザルトクラス
'''

import pandas as pd

class Results():
    def __init__(self):
        self.l_trssi = []
        self.l_erssi = []
        self.l_nnrssi = []
        self.l_krssi = []

    def rappend(self, trssi, erssi, nnrssi, krssi):
        self.l_trssi.append(trssi)
        self.l_erssi.append(erssi)
        self.l_nnrssi.append(nnrssi)
        self.l_krssi.append(krssi)
 
    def calc_error(self):
        self.rssi_error = [self.l_trssi[i]-self.l_erssi[i] for i in range(len(self.l_trssi))]
        self.nnrssi_error = [self.l_trssi[i]-self.l_nnrssi[i] for i in range(len(self.l_trssi))]
        self.krssi_error = [self.l_trssi[i]-self.l_krssi[i] for i in range(len(self.l_trssi))]

    '''csv出力
    入力: ファイル名
    '''
    def output(self, c):
        df = pd.DataFrame({
            'onlyidw': self.rssi_error,
            'nn': self.nnrssi_error,
            'kernel':self.krssi_error
        })
        df.to_csv(c)
