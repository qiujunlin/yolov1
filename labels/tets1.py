from math import radians, cos, sin, asin, sqrt
def fun():
    str="106.608105,29.355106;106.608105,29.355106;106.608105,29.355106;106.608105,29.355106;106.605595,29.353342;106.606981,29.346376;106.606268,29.349908;106.606933,29.346584;106.606791,29.347488;106.606981,29.346378;106.606415,29.349166;106.60698,29.34638;106.606625,29.348288;106.606981,29.346378;106.606981,29.346378;106.606981,29.346378;106.606981,29.346376;106.606981,29.346376;106.606981,29.346376;106.606981,29.346378;106.607183,29.345708;106.607286,29.34531;106.607401,29.344826;106.607378,29.344568;106.606896,29.34448;106.606141,29.34432;106.605955,29.344214;106.606396,29.34433;106.606836,29.344434;106.606963,29.344118;106.607091,29.343564"
    arr=str.split(";")
    for i, value in enumerate(arr):
        arr[i]=value.split(",")
        print(float(arr[i][0]))
        float(arr[i][1])
        if i+1<len(arr):

            print(haversine(float(arr[i][0]),float(arr[i][1]),float(arr[i+1][0]),float(arr[i+1][1])))
    print(arr)

def haversine(lon1, lat1, lon2, lat2):  # 经度1，纬度1，经度2，纬度2 （十进制度数）
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # 将十进制度数转化为弧度
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine公式
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6371  # 地球平均半径，单位为公里
    return c * r * 1000
fun()