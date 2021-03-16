import cv2
import sklearn
import matplotlib.pyplot as plt
import numpy as np

delta_list = np.linspace(1, 10, 9)
print(delta_list)

x = np.linspace(-10, 10, 100)
y = []
for delta in delta_list:
    y_i = []
    for x_i in x:
        y_i.append(np.exp(-x_i ^ 2 / (2 * delta ^ 2)) / (np.sqrt(2 * 3.1415)) * delta)

    y.append(y_i)

for y_i in y:
    plt.plot(x, y_i, color='blue', linewidth=1.0, linestyle='--', label='blue')

plt.show()