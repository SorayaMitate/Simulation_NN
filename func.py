import math
import numpy as np

'''パディング数を計算する関数
入力 : 
    im_size : 画像の長さ (1辺)
    map_size : REM の長さ (1辺)
出力 :
    右、左にいれるパディング数
'''
def calc_pading_num(im_size, map_size):
    while True:
        if im_size < map_size:
            break
        else:
            im_size = im_size - map_size
    
    padding_num = map_size - im_size

    def hantei(a):
        if a % 2 == 0:
            return int(a/2), int(a/2)
        else :
            return int(a/2)+1, int(a/2)

    return hantei(padding_num)
    

'''コード作成用
入力 :
    data_split : コードの文字列
出力 :
    緯度、経度コード
'''
def make_code(data_split):
    x0 = data_split[1][-2]
    x1 = data_split[2][-2]
    x2 = data_split[3][-2]
    x3 = data_split[4][-2]
    x4 = data_split[5][-2]
    x = int(x0+x1+x2+x3+x4)
    y0 = data_split[1][-1]
    y1 = data_split[2][-1]
    y2 = data_split[3][-1]
    y3 = data_split[4][-1]
    y4 = data_split[5][-1]
    y = int(y0+y1+y2+y3+y4)
    return x, y


'''ビルの数をカウントする関数 (255がビルと仮定)
入力 :
    p_list: 二値点間の直線上の画素群 (リスト)
    th: 建物と見なす画素数のスレッショルド
出力 :
    l_res1: 1つのビルを構成する画素の数 (リスト)
    l_res2: ビルまでの距離を構成する画素の数 (リスト)
'''
def count_building(p_list, th):
    
    count1 = 0
    count2 = 0
    l_build = []
    l_road = []

    tmp = 0
    for p in p_list:
        if tmp == p:
            if p == 255:
                count1 += 1
            else:
                count2 += 1
        else:
            if p == 255:
                l_road.append(count2)
                count2 = 0
                count1 +=1
            else:
                l_build.append(count1)
                count1 = 0
                count2 += 1
        tmp = p
    
    l_build.append(count1)
    l_road.append(count2)

    l_build_index = [i for i in range(len(l_build)) if l_build[i] > th]

    if len(l_build_index) > 0:    
        build_num = len(l_build_index)
        init_road = l_road[l_build_index[0]]
        last_road = l_road[l_build_index[-1]]
    
    else :
        build_num = 0
        init_road = 0
        last_road = 0

    return build_num, init_road, last_road


'''メッシュ間の距離を算出する関数
入力 :
    x,y : メッシュ座標
出力 :
    メッシュ間の距離
'''
def calc_dist(x, y, i_x, i_y):
    def surpport1(a,b):
        if (a-b) % 10 == 1 or (a-b) % 10 == 9:
            tmp = 1
        else:
            tmp = 0
        return tmp

    def surpport2(a,b):
        if a <= b:
            return b,a
        else:
            return a,b 
    
    t_x1, t_x2 = surpport2(x, i_x)
    t_y1, t_y2 = surpport2(y, i_y)
    d_x = (t_x1 - t_x2) // 10 * 2 + surpport1(t_x1, t_x2)
    d_y = (t_y1 - t_y2) // 10 * 2 + surpport1(t_y1, t_y2)
    return math.sqrt((d_x*5)**2 + (d_y*5)**2)


'''補間の範囲算出
入力 :
    ran : 補間対象範囲
    x,y : 対象メッシュの座標
出力 :
    四方範囲
'''
def calc_range(ran, x, y):
    def ex(l,a): 
        if str(a)[-1] == '0':
            if l % 2 == 0:
                ma = l // 2 * 10
                mi = l // 2 * 10
            else:
                ma = l // 2 * 10 + 1
                mi = l // 2 * 10 + 9
        else:
            if l % 2 == 0:
                ma = l // 2 * 10
                mi = l // 2 * 10
            else:
                ma = l // 2 * 10 + 9
                mi = l // 2 * 10 + 1
        return ma, mi
    
    x_max, x_min = ex(ran, x)
    y_max, y_min = ex(ran, y)

    ran = (x_max, x_min, y_max, y_min)
    return ran

'''補間の範囲算出(シミュレーション用)
入力 :
    ran : 補間対象範囲
    x,y : 対象メッシュの座標
出力 :
    四方範囲
'''
def calc_range_sim(ran, x, y):
    
    x_max = x + ran
    x_min = x - ran
    y_max = y + ran
    y_min = y - ran

    ran = (x_max, x_min, y_max, y_min)
    return ran


'''逆距離加重法
入力 :
    neibor_rssi : 対象メッシュの周囲のメッシュの RSSI (リスト)
    dist : 対象メッシュの周囲のメッシュとの距離 (リスト)
出力 :
    補間値
'''
def idw(neibor_rssi, dist):
    estimate = 0
    weight_sum = 0
    for i in range(len(neibor_rssi)):
        estimate = estimate + (1.0 / dist[i])**2 * neibor_rssi[i]
        weight_sum = weight_sum + (1.0 / dist[i])**2
    return estimate / weight_sum


'''マップ上のメッシュコードを画素座標に変換
入力 : 
    t_gaso : メッシュコード (タプル)
    l_index : 緯度方向のメッシュコード一覧 (リスト)
    l_col : 経度方向のメッシュコード一覧 (リスト)
    block : 画像をメッシュ上にしたときのブロックサイズ (タプル)
出力 : 
    画素座標
'''
def convert_gaso_to_mcode(t_gaso, l_index, l_col, size):
    tmp_x = l_index.index(t_gaso[0]*(-1))
    tmp_y = l_col.index(t_gaso[1])
    tmp_x = int(tmp_x * size[2] + size[2] / 2)
    tmp_y = int(tmp_y * size[3] + size[3] / 2)
    return (tmp_x, tmp_y)

'''マップ上のメッシュコードを画素座標に変換 (simulation用)
入力 : 
    t_gaso : メッシュコード (タプル)
    l_index : 緯度方向のメッシュコード一覧 (リスト)
    l_col : 経度方向のメッシュコード一覧 (リスト)
    block : 画像をメッシュ上にしたときのブロックサイズ (タプル)
出力 : 
    画素座標
'''
def convert_gaso_to_mcode_forsim(t_gaso, l_index, l_col, size):
    tmp_x = l_index.index(t_gaso[0])
    tmp_y = l_col.index(t_gaso[1])
    tmp_x = int(tmp_x * size[2] + size[2] / 2)
    tmp_y = int(tmp_y * size[3] + size[3] / 2)
    return (tmp_x, tmp_y)


'''補間
入力:
    df: 対象のデータフレーム(必須カラム['rx', 'ry'])
    c: 補間対象のカラム名
    x,y: 受信機座標
    *args: 補間の範囲
出力:
    l_dist: 補間に使用するメッシュまでの距離
    l_res: 補間に使用するメッシュの実現値
'''
def interpolation(df, c, x, y, *args):
    
    l_dist = []
    l_res = []
    for i in range(x - args[0], x + args[1]+1):
        if (str(i)[-1] == '0') or (str(i)[-1] == '1'):
            for j in range (y - args[2], y + args[3] +1):
                if (str(j)[-1] == '0') or (str(j)[-1] == '1'):
                    if (len(df[(df['rx']==i)&(df['ry']==j)][c]) > 0):
                        if (calc_dist(x, y, i, j) > 0.0):
                            l_dist.append(calc_dist(x, y, i, j))
                            l_res.append(df[(df['rx']==i)&(df['ry']==j)][c].values[0])

    return l_dist, l_res


'''補間 (シミュレーション用)
入力:
    df: 対象のデータフレーム(必須カラム['rx', 'ry'])
    c: 補間対象のカラム名
    x,y: 受信機座標
    *args: 補間の範囲
出力:
    l_dist: 補間に使用するメッシュまでの距離
    l_res: 補間に使用するメッシュの実現値
'''
def interpolation_sim(df, c, x, y, ran):
    
    l_dist = []
    l_res = []

    for i in range(x - ran, x + ran+1):
        for j in range (y - ran, y + ran +1):
            if (len(df[(df['rx']==i)&(df['ry']==j)][c]) > 0):
                if (calc_dist(x, y, i, j) > 0.0):
                    l_dist.append(df[(df['rx']==i)&(df['ry']==j)]['dist'].values[0])
                    l_res.append(df[(df['rx']==i)&(df['ry']==j)][c].values[0])

    return l_dist, l_res


def PL(f,dis):
    #減衰定数
    gamma = 3.0
    pii = 4.0*np.pi
    fle = 2.4*10.0**9
    lam = 299792458.0 / f
    if dis == 0:
        loss = -20*np.log10(lam/pii) + gamma*10*np.log10(0.5)
        return loss
    else:
        loss = -20.0*np.log10(lam/pii) + gamma*10*np.log10(dis)
        return loss

def calc_dist2(x1, y1, x2, y2):
    return np.sqrt(((x2-x1)*5)**2 + ((y2-y1)*5)**2)
