import pyproj

EPSG_3003 = 3003
EPSG_2196 = 2196
EPSG_3395 = 3395
EPSG_4326_WGS84 = "EPSG:4326"


# Proiezione di Peters con aree riadattate.
MY_CRS = pyproj.Proj(
    proj="cea", lon_0=0, lat_ts=45, x_0=0, y_0=0, ellps="WGS84", units="m"
).srs
