'''
定数クラス
'''

class Const():
    def __init__(self):
        
        #エリア1座標
        lat_min1 = 46810
        lat_max1 = 47130
        lon_min1 = 36731
        lon_max1 = 36801
        #マップのメッシュ数
        self.XSIZE1 = 65
        self.YSIZE1 = 16
        self.AREA1 = (lat_min1,lat_max1,lon_min1,lon_max1,\
            self.XSIZE1, self.YSIZE1)
        
        #エリア2座標
        lat_min2 = 46350
        lat_max2 = 46511
        lon_min2 = 38051
        lon_max2 = 38400
        #マップのメッシュ数
        self.XSIZE2 = 34
        self.YSIZE2 = 70
        self.AREA2 = (lat_min2,lat_max2,lon_min2,lon_max2,\
            self.XSIZE2, self.YSIZE2)

        #エリア3座標
        lat_min3 = 48430
        lat_max3 = 48741
        lon_min3 = 37210
        lon_max3 = 37391
        #マップのメッシュ数
        self.XSIZE3 = 69
        self.YSIZE3 = 38
        self.AREA3 = (lat_min3,lat_max3,lon_min3,lon_max3,\
            self.XSIZE3, self.YSIZE3)

        #対象エリア
        self.AREA = [self.AREA1, self.AREA2, self.AREA3]

        #対象エリアのインデックス
        self.AREA_INDEX = 1

        #建物と道路のスレッショルド (グレースケール)
        self.TH_BUILDINGS = 209 

        #建物を構成する画素数のスレッショルド
        self.TH_GASO = 4

        '''
        Neural Net に関する設定項目
        '''
        #各層の数
        self.N_INPUT = 5
        self.N_HIDDEN = 100
        self.N_OUTPUT = 1

        #イテレーション数
        self.N_ITERATION = 500
        #エポック数
        self.N_EPOCH = 256

        #使用コア数
        self.N_CPU = 9

        #試行回数
        self.N_TRIALS = 101

        #メッシュサイズ
        self.SIZE_MESH = 5

        #シャドウイングの標準偏差
        self.VAR = 8.0

        #空間相関距離
        self.COR_DIST = 50.0

        #ノード数
        #self.N_NODE = 50
        self.N_NODE_MIN = 30
        self.N_NODE_MAX = 40
        self.M_NODE_EPO = 20
        self.N_INDEX = [i for i in range(self.N_NODE_MIN, self.N_NODE_MAX, self.M_NODE_EPO)]

        #使用周波数
        self.FREQ = 920000000

        #建物による減衰
        self.ATEN_BUILD = 3

        #電波環境を劣悪にさせるノードの割合
        self.N_NODE_INTEF = 0.1
        self.ATEN_INTEF = 5
