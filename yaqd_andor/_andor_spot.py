__all__ = ["AndorSpot"]

import asyncio
import numpy as np
from time import sleep, time
import pathlib

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List, Union
from . import atcore
from . import features
from . import _andor_sdk3

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class AndorSpot(_andor_sdk3.AndorSDK3):
    _kind = "andor-spot"
    state_features = [
        "exposure_time",
        "pixel_readout_rate",
        "electronic_shuttering_mode",
        "simple_preamp_gain_control",
        "spurious_noise_filter",
        "static_blemish_correction",
    ]

    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self.nframes = self._config["nframes"]
        # overwrite channels
        self._channel_names = ["mean", "xpos", "ypos", "timestamp"]
        self._channel_mappings = {k: ["index"] for k in self._channel_names}
        self._mapping_units = {"index": "None"}
        self._channel_units = {"mean": "counts", "xpos": "None", "ypos": "None", "timestamp": "sec"}

        self._channel_shapes = {
            k: (self.nframes,) for k in self._channel_mappings
        }
        index_ai = np.arange(self.nframes)
        index = index_ai.__array_interface__
        index["data"] = index_ai.tobytes()
        self._mappings = {"index": index}


        if self._config["background_image"]:
            self.bg = np.load(pathlib.Path(self._config["background_image"]))
        else:
            self.bg = 0
        self._set_aoi()
        self._set_temperature()


    def _set_aoi(self):
        aoi_keys = ["aoi_binning", "aoi_width", "aoi_left", "aoi_height", "aoi_top"]
        binning, width, left, height, top = [self._config[k] for k in aoi_keys]
        binning = int(binning[0])  # equal xy binning, so only need 1 index

        # check if aoi is within sensor limits
        max_width = self.features["sensor_width"].get()
        max_height = self.features["sensor_height"].get()

        # handle defaults (maximum sizes)
        if left is None:
            left = 1
        if top is None:
            top = 1
        if width is None:
            width = max_width - left + 1
        if height is None:
            height = max_height - top + 1
        width //= binning
        height //= binning

        self.image_shape = (height, width)

        self.logger.debug(f"{max_width}, {max_height}, {binning}, {width}, {height}, {top}")
        w_extent = width * binning + (left - 1)
        h_extent = height * binning + (top - 1)
        if w_extent > max_width:
            raise ValueError(f"height extends over {w_extent} pixels, max is {max_width}")
        if h_extent > max_height:
            raise ValueError(f"height extends over {h_extent} pixels, max is {max_height}")

        self.features["aoi_binning"].set(f"{binning}x{binning}")
        self.features["aoi_width"].set(width)
        self.features["aoi_left"].set(left)
        self.features["aoi_height"].set(height)
        self.features["aoi_top"].set(top)

        for k in ["aoi_height", "aoi_width", "aoi_top", "aoi_left", "aoi_binning"]:
            self.logger.debug(f"{k}: {self.features[k].get()}")

    def _set_temperature(self):
        # possible_temps = self.features["temperature_control"].options()
        sensor_cooling = self._config["sensor_cooling"]
        self.features["sensor_cooling"].set(sensor_cooling)
        if sensor_cooling:
            set_temp = self.features["temperature_control"].get()
            self.logger.info(f"Sensor is cooling.  Target temp is {set_temp} C.")
            self._loop.run_in_executor(None, self._check_temp_stabilized)
        else:
            sensor_temp = self.features["sensor_temperature"].get()
            self.logger.info(f"Sensor is not cooled.  Current temp is {sensor_temp} C.")

        status = self.features["temperature_status"].get()

    def _check_temp_stabilized(self):
        set_temp = self.features["temperature_control"].get()
        sensor_temp = self.features["sensor_temperature"].get()
        diff = float(set_temp) - sensor_temp
        while abs(diff) > 1.0:
            self.logger.info(
                f"Sensor is cooling.  Target: {set_temp} C.  Current: {sensor_temp:0.2f} C."
            )
            sleep(5)
            set_temp = self.features["temperature_control"].get()
            sensor_temp = self.features["sensor_temperature"].get()
            diff = float(set_temp) - sensor_temp
        self.logger.info("Sensor temp is stabilized.")

    async def _measure(self):
        image_size_bytes = self.features["image_size_bytes"].get()
        bufs = [np.empty((image_size_bytes,), dtype="B")] * self.nframes
        timeout = max(self.features["exposure_time"].get() * 2e3, 100)
        # 2e3: seconds to ms (1e3), plus wait twice as long as acquisition before timeout
        outs = {
            k: [] for k in self._channel_names
        }
        stride = self.features["aoi_stride"].get()

        async def frames():
            """
            software timer to choke framerate
            frames will wait this long between frames, or however long it takes to 
            measure and transfer the data, whichever is _longer_
            """
            wait = self._state.get("frame_wait")
            for i in range(self.nframes):
                yield i
                await asyncio.sleep(wait)

        try:
            async for i in frames():  # use a generator with a timer?
                self.logger.info(f"{i=}")
                self.sdk.queue_buffer(self.hndl, bufs[i].ctypes.data, image_size_bytes)
                # acquire frame
                self.features["acquisition_start"]()
                outs["timestamp"].append(time())
                # executor too slow for this purpose
                # (returnedBuf, returnedSize) = await asyncio.get_running_loop().run_in_executor(
                #     None, self.sdk.wait_buffer, self.hndl, timeout
                # )
                self.sdk.wait_buffer(self.hndl, timeout)
                self.features["acquisition_stop"]()

                iframe = np.lib.stride_tricks.as_strided(
                    np.frombuffer(bufs[i], dtype=np.uint16),
                    shape=self.image_shape,
                    strides=(stride, 2),
                ) - self.bg
                mi = self.process_frame(np.ascontiguousarray(iframe))
                for k, v in mi.items():
                    outs[k].append(v)

        except Exception as err:
            self.logger.error(f'frame={i}', exc_info=True)

        self.sdk.flush(self.hndl)

        return {k: np.array(v) for k, v in outs.items()}

    def process_frame(self, frame) -> dict:
        mean = frame.mean()
        xpos = (frame * np.arange(frame.shape[0])[:, None]).mean() / mean
        ypos = (frame * np.arange(frame.shape[1])[None, :]).mean() / mean
        return {"mean": mean, "xpos": xpos, "ypos": ypos}


    def set_frame_rate(rate:int):
        """set frame rate in seconds"""
        ...


# to run without entry point
if __name__ == "__main__":
    AndorSpot.main()
