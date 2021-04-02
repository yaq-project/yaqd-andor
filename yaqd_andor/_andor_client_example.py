import yaqc
import matplotlib.pyplot as plt
import time
import numpy as np


cam = yaqc.Client(38999)
fig, ax = plt.subplots()
shape = cam.get_channel_shapes()["image"]
im = ax.imshow(np.zeros(shape))


def submit():
    try:
        im_id = cam.measure()
        while cam.busy():
            time.sleep(0.1)
        a = cam.get_measured()["image"]
        print(a.max())
    except ConnectionError:
        pass
    im.set_data(a)
    im.norm.autoscale(a)
    ax.set_title(f"{im_id}")
    plt.draw()


timer = fig.canvas.new_timer(interval=200)

@timer.add_callback
def update():
    submit()

timer.start()
plt.show()
