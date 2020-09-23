import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

import Const

const = Const.Const()

plt.rcParams["font.size"] = 24
font = {'family' : 'meiryo'}
matplotlib.rc('font', **font)


def test(s):

    s.sort()

    cdf = []
    for i in range(len(s)):
        if i == 0:
            tmp = 1.0 / len(s)
        else :
            tmp = tmp + 1.0 / len(s)
        cdf.append(tmp)
    return s, cdf

def iroiro(s):
    from statistics import mean, median, variance,stdev
    m = mean(s)
    median = median(s)
    variance = variance(s)
    stdev = stdev(s)
    print('平均: {0:.2f}'.format(m))
    print('中央値: {0:.2f}'.format(median))
    print('分散: {0:.2f}'.format(variance))
    print('標準偏差: {0:.2f}'.format(stdev))


#40ノードの正解データ
path = 'C:\\Users\\soraya-PC\\code\\Simulation_NN\\results\\'
s = path + '40' + '_error.csv'
data = pd.read_csv(s,index_col=0)
col = data.columns
print('--- proposed ---  ', col[1])
s1, cdf1 = test(list(data['enn']))
iroiro(s1)
plt.scatter(s1, cdf1, s=10,label='True value')



l_points = []
l_name = [0.2, 0.4, 0.6]
for n in l_name:

    print('wariai = ',n)
    
    s = path + str(n) + '_error.csv'

    data = pd.read_csv(s,index_col=0)
    col = data.columns

    #print('--- proposed ---  ', col[0])
    #s0, cdf0 = test(list(data['eidw']))
    #iroiro(s0)

    print('--- proposed ---  ', col[1])
    s1, cdf1 = test(list(data['enn']))
    iroiro(s1)

    l_points.append(s1)

    '''散布図
    '''
    #plt.scatter(s0, cdf0, s=10,label='RSSIExp. by Only IDW')
    #plt.scatter(s1, cdf1, s=10,label='RSSIExp. by Only NN')
    plt.scatter(s1, cdf1, s=10,label=str(n))

plt.legend(loc='upper left',fontsize=20)
plt.grid()
plt.xlabel('Expected Error [dB]',fontname="HGGothicM",fontsize=30)
plt.ylabel('CDF',fontname="HGGothicM",fontsize=30)
plt.show()

points = tuple(l_points)
name = tuple(l_name)

fig, ax = plt.subplots()
bp = ax.boxplot(points)
ax.set_xticklabels(name)
plt.legend(loc='upper left',fontsize=20)
plt.grid()
plt.xlabel('Expected Error [dB]',fontname="HGGothicM",fontsize=30)
plt.ylabel('CDF',fontname="HGGothicM",fontsize=30)
plt.show()
