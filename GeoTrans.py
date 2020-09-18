import pyproj
from pyproj import Transformer

class GeoTrans(object):
    def __init__(self, from_epsg=4301,to_epsg=4326):
        """コンストラクタ

        Keyword Arguments:
            from_epsg {int} -- [変換元のEPSGコード] (default: {4301})
            to_epsg {int} -- [変換先のEPSGコード] (default: {4326})
        """
        self.from_epsg = pyproj.Proj('+init=EPSG:{}'.format(from_epsg))
        self.to_epsg = pyproj.Proj('+init=EPSG:{}'.format(to_epsg))
        self._transformer = Transformer.from_proj(self.from_epsg,self.to_epsg)

    def transform(self,lon:float,lat:float): 
        """測地系変換

        Arguments:
            lon {float} -- [経度]
            lat {float} -- [緯度]

        Returns:
            [taple] -- [経度,緯度]
        """
        return self._transformer.transform(lon,lat)
