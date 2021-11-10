from matplotlib import pyplot as plt
import numpy as np
import cv2
import math


class LineBuilder:
    def __init__(self, line, rate,D):
        self.rate = float(rate)
        self.line = line
        self.xs = list(line.get_xdata())
        self.ys = list(line.get_ydata())
        self.cid = line.figure.canvas.mpl_connect('button_press_event', self)
        self.D = D
        # print("Start")

    def __call__(self, event):
        global D

        if event.button == 3:
            global scat1
            global scat2
            if event.inaxes != self.line.axes: return
            if len(self.xs) > 1:
                self.xs = list([0, ])
                self.ys = list([0, ])
                scat1.remove()
                scat2.remove()
            if self.xs[0] == 0:
                self.xs[0] = event.xdata
                self.ys[0] = event.ydata
                scat1 = plt.scatter(event.xdata, event.ydata, s=6, edgecolor='red')
            else:
                self.xs.append(event.xdata)
                self.ys.append(event.ydata)
                scat2 = plt.scatter(event.xdata, event.ydata, s=6, edgecolor='red')
            self.line.set_data(self.xs, self.ys)
            self.line.figure.canvas.draw()

            xht = self.xs[len(self.xs) - 1]
            xtd = self.xs[len(self.xs) - 2]
            yht = self.ys[len(self.ys) - 1]
            ytd = self.ys[len(self.ys) - 2]
            x_values = [xtd, xht]
            y_values = [ytd, yht]
            plt.plot(x_values, y_values, color='green')
            self.D = abs(x_values[1]-x_values[0])
        return self.D
    def value (self):
        return 10/self.D