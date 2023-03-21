import numpy as np
from scipy.signal import argrelextrema
import time
from win10toast import ToastNotifier

class Analizer():
    def __init__(self, api):
        self.api = api
        self.toaster = ToastNotifier()

    def _send_notification(self, threshold, list):
        self.toaster.show_toast(f"Wykryto dużą korektę (-{threshold}%)", f'{", ".join(list)}', duration=10)

    def get_extremes(self, array):
        array = np.array(array)
        return np.sort(np.concatenate((argrelextrema(array, np.greater)[0], argrelextrema(array, np.less)[0])))

    def extremes_convolution(self, X, y, size=5, type='min'):
        assert type in ['min', 'max']
        results = []
        for i in X:
            elements = [j for j in X if j >= i - size and j <= i + size]
            if type == 'max':
                if y[i] >= max([y[k] for k in elements]):
                    results.append(i)
            elif type == 'min':
                if y[i] <= min([y[k] for k in elements]):
                    results.append(i)
        return results

    def get_continuous_avg(self, array, size=5):
        array = np.concatenate((np.zeros((size)), np.array(array)))
        return [np.sum(array[i-size+1:i+1])/size if i >= size+size else np.sum(array[:i+1])/(i+1-size) for i in range(size, len(array))]
        

    def cyclical_analysis(self, list, interval=10, threshold=10):
        list.remove('BUSD')
        list = [f'{i}BUSD' for i in list]
        dict = {}
        while True:
            detected = []

            for i in list:
                try:
                    klines = self.api.get_historical_klines(i, self.api.client.KLINE_INTERVAL_1HOUR, days=7)
                    ratio = self.api.current_to_highest_price_ratio(i, klines)
                    if ratio < threshold*(-1):
                        detected.append(i)
                except:
                    continue

            if detected != []:
                for i in detected:
                    if i not in dict or dict[i] == 0:
                        dict[i] = 1
                    else:
                        if dict[i] < interval:
                            dict[i] += 1
                        else:
                            dict[i] = 0
                
                detected = [i for i in detected if dict[i] == 1]

                self._send_notification(threshold, detected)
                    
            time.sleep(interval*60)


# a = Analizer()

# x = a.get_continuous_avg([1,2,3,4,5,6,7,8])
# print(x)