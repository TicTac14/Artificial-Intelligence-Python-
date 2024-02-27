import matplotlib.pyplot as plt
import math
data = {"xs" : [],
        "ys": []}

for i in range(-100, 100, 1):
    data["xs"].append(i/15)
    data["ys"].append(math.cos(i/16))

plt.xlabel("x")
plt.ylabel("y")
plt.title("x vs y")

plt.plot(data["xs"], data["ys"])
plt.grid()
plt.show()


    



