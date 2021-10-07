__all__ = ["AndorSDK3"]

import asyncio
import numpy as np
from time import sleep
import os

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Dict, Any, List, Union
from . import atcore
from . import features

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class AndorSDK3(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    def __init__(self, name, config, config_filepath):
        super().__init__(name, config, config_filepath)
        self._channel_names = ["image"]
        self._channel_mappings = {"image": ["x_index", "y_index"]}
        self._mapping_units = {"x_index": "None", "y_index": "None"}
        self._channel_units = {"image": "counts"}

        initial_cwd = os.getcwd()
        try:
            os.chdir(os.path.dirname(__file__))
            self.sdk = ATCore()  # Initialise SDK3
        finally:
            os.chdir(initial_cwd)
        # find devices
        device_count = self.sdk.get_int(self.sdk.AT_HNDL_SYSTEM, "DeviceCount")
        if device_count == 0:
            raise ConnectionError("No devices found.")
        # select device
        for i in range(device_count):
            temp = self.sdk.open(i)
            serial = self.sdk.get_string(temp, "SerialNumber")
            if serial == self._config["serial"]:
                self.hndl = temp
                print("    Serial No   : ", serial)
                break
            self.sdk.close(temp)
        else:
            raise ConnectionError(
                r"device with serial number {0} not found".format(self._config["serial"])
            )

        self.features = {}
        model = self._config["model"][0].lower()
        assert model in "ansz"
        for k, v in features.specs.items():
            if model in v.availability:
                try:
                    self.features[k] = features.obj_from_spec(self.sdk, self.hndl, v)
                except NotImplementedError:
                    self.logger.info(
                        f"feature {v.sdk_name} is supposed to be implemented, but is not!"
                    )
                    pass
                else:
                    self.logger.debug(
                        f"{k}, {self.features[k].is_implemented}, {self.features[k].is_readonly}"
                    )

        self.sensor_info = {}
        for k in ["sensor_width", "sensor_height", "pixel_height", "pixel_width"]:
            try:
                self.sensor_info[k] = self.features[k].get()
            except ATCoreException as err:
                pass
        self.logger.debug(self.sensor_info)

        """
        # implement config, state features
        self.features["spurious_noise_filter"].set(self._config["spurious_noise_filter"])
        self.features["static_blemish_correction"].set(self._config["static_blemish_correction"])
        self.features["electronic_shuttering_mode"].set(self._config["electronic_shuttering_mode"])
        self.features["simple_preamp_gain_control"].set(self._config["simple_preamp_gain_control"])
        self.features["exposure_time"].set(self._config["exposure_time"])
        # aoi currently in config, so only need to run on startup
        self._set_aoi()
        """

    async def _measure(self):
        image_size_bytes = self.features["image_size_bytes"].get()
        buf = np.empty((image_size_bytes,), dtype="B")
        timeout = self.features["exposure_time"].get() * 2e3
        # 2e3: seconds to ms (1e3), plus wait twice as long as acquisition before timeout
        try:
            self.sdk.queue_buffer(self.hndl, buf.ctypes.data, image_size_bytes)
            # acquire frame
            self.features["acquisition_start"]()
            self.logger.debug("Waiting on buffer")
            (returnedBuf, returnedSize) = await self._loop.run_in_executor(
                None, self.sdk.wait_buffer, self.hndl, timeout
            )
            self.logger.debug("Done waiting on buffer")
            self.features["acquisition_stop"]()
        except ATCoreException as err:
            self.logger.error(f"SDK3 Error {err}")

        class ArrayInterface:
            def __init__(self, buf, shape, strides):
                self.__array_interface__ = {
                    "shape": shape,
                    "typestr": "<u2",
                    "data": buf,
                    "strides": strides,
                    "version": 3,
                }

        stride = self.features["aoi_stride"].get()
        pixels = np.array(ArrayInterface(buf.data, self._channel_shapes["image"], (stride, 2)))
        pixels = np.lib.stride_tricks.as_strided(
            np.frombuffer(buf, dtype=np.uint16),
            shape=self._channel_shapes["image"],
            strides=(stride, 2),
        )
        self.logger.debug(f"{pixels.size}, {np.prod(self._channel_shapes['image'])}")
        pixels = np.ascontiguousarray(pixels)
        arrayinterface = pixels.__array_interface__
        arrayinterface["data"] = pixels.tobytes()
        self.sdk.flush(self.hndl)

        return {"image": arrayinterface}

    def get_sensor_info(self):
        return self.sensor_info

    def get_feature_names(self) -> List[str]:
        return [v.sdk_name for v in self.features.values()]

    def get_feature_value(self, k: str) -> Union[int, bool, float, str]:
        feature = self.features[k]
        return feature.get()

    def get_feature_options(self, k: str) -> List[str]:  # -> List[Union[str, float, int]]:
        feature = self.features[k]
        # if isinstance(feature, features.SDKEnum):
        return feature.options()
        # elif isinstance(feature, features.SDKFloat) or isinstance(feature, features.SDKInt):
        #     return [feature.min(), feature.max()]
        # else:
        #     raise ValueError(f"feature {feature} is of type {type(feature)}, not `SDKEnum`.")

    def close(self):
        self.sdk.close(self.hndl)
