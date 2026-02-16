from PyQt5 import QtWidgets
import pyqtgraph as pg
import sys
import time


class TeacherAnalytics(QtWidgets.QMainWindow):

    def __init__(self, window_seconds=60):
        super().__init__()

        self.setWindowTitle("CognitiveLens - Teacher Analytics")
        self.setGeometry(150, 100, 1100, 800)

        self.window_seconds = window_seconds
        self.start_time = time.time()

        pg.setConfigOption('background', '#121212')
        pg.setConfigOption('foreground', 'w')

        self.widget = pg.GraphicsLayoutWidget()
        self.setCentralWidget(self.widget)

        #time + score arrays
        self.time_data = []
        self.sleep_data = []
        self.attention_data = []
        self.stress_data = []
        self.confusion_data = []

        #create plots
        self.p1 = self.widget.addPlot(title="Drowsiness")
        self.p1.setYRange(0, 100)

        self.widget.nextRow()

        self.p2 = self.widget.addPlot(title="Attention")
        self.p2.setYRange(0, 100)

        self.widget.nextRow()

        self.p3 = self.widget.addPlot(title="Stress")
        self.p3.setYRange(0, 100)

        self.widget.nextRow()

        self.p4 = self.widget.addPlot(title="Confusion")
        self.p4.setYRange(0, 100)

        #curves
        self.c1 = self.p1.plot(pen=pg.mkPen('#FF5555', width=3))
        self.c2 = self.p2.plot(pen=pg.mkPen('#00AAFF', width=3))
        self.c3 = self.p3.plot(pen=pg.mkPen('#FFAA00', width=3))
        self.c4 = self.p4.plot(pen=pg.mkPen('#AA00FF', width=3))

    def update(self, sleep, attention, stress, confusion):

        current_time = time.time() - self.start_time

        self.time_data.append(current_time)
        self.sleep_data.append(sleep)
        self.attention_data.append(attention)
        self.stress_data.append(stress)
        self.confusion_data.append(confusion)

        #remove old data (sliding window)
        while self.time_data and current_time - self.time_data[0] > self.window_seconds:
            self.time_data.pop(0)
            self.sleep_data.pop(0)
            self.attention_data.pop(0)
            self.stress_data.pop(0)
            self.confusion_data.pop(0)

        self.c1.setData(self.time_data, self.sleep_data)
        self.c2.setData(self.time_data, self.attention_data)
        self.c3.setData(self.time_data, self.stress_data)
        self.c4.setData(self.time_data, self.confusion_data)
