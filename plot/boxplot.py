import os

import plotly.plotly as py
import plotly.graph_objs as go

print  os.listdir('plot/results')
second_path = raw_input("Enter first file path:")
y0 = []
for line in open(second_path):
    y0.append(float(line.strip()))

second_path = raw_input("Enter second file path:")
y1 = []
for line in open(second_path):
    y1.append(float(line.strip()))

trace0 = go.Box(
    y=y0
)
trace1 = go.Box(
    y=y1
)
data = [trace0, trace1]
py.plot(data)