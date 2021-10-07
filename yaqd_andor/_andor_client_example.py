import yaqc  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import time
import numpy as np

port = 38999  # daemon


def mapping_to_extent(xm, ym):
    return [xm.min(), xm.max(), ym.max(), ym.min()]


def measure_and_plot():
    try:
        im_id = cam.measure()
        while cam.busy():
            time.sleep(0.1)
        a = cam.get_measured()["image"]
    except ConnectionError:
        print("error")
        pass
    else:
        im.set_data(a)
        im.norm.autoscale(a)
        cbar.update_normal(im)
        ax.set_title(f"{im_id}")
        plt.draw()
    print(a.max(), a.min())


cam = yaqc.Client(port)
fig, (ax) = plt.subplots()
timer = fig.canvas.new_timer(interval=200)


@timer.add_callback
def update():
    measure_and_plot()


cam_map = cam.get_mappings()
xm = np.array(cam_map["x_index"])
ym = np.array(cam_map["y_index"])
shape = cam.get_channel_shapes()["image"]
im = ax.imshow(np.zeros(shape), extent=mapping_to_extent(xm, ym))
cbar = fig.colorbar(im)
measure_and_plot()
timer.start()
plt.show()
