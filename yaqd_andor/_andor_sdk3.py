__all__ = ["AndorSDK3"]

import asyncio
import numpy as np
import os

from yaqd_core import IsDaemon, IsSensor, HasMeasureTrigger, HasMapping
from typing import Any, List, Union
from . import atcore
from . import features

ATCore = atcore.ATCore
ATCoreException = atcore.ATCoreException


class AndorSDK3(HasMapping, HasMeasureTrigger, IsSensor, IsDaemon):
    state_features: List[str] = []

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
                self.logger.info(f"    Serial No   : {serial}")
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
                    self.logger.warn(
                        f"feature {v.sdk_name} is supposed to be implemented, but is not!"
                    )
                else:
                    self.logger.debug(
                        f"{k}, {self.features[k].is_implemented}, {self.features[k].is_readonly}"
                    )

        self.sensor_info = {}
        for k in ["sensor_width", "sensor_height", "pixel_height", "pixel_width"]:
            try:
                self.sensor_info[k] = self.features[k].get()
            except ATCoreException as err:
                self.logger.error(err)
        self.logger.debug(self.sensor_info)

        for key in self.state_features:
            fi = self.features[key]
            dest = self._state[key]
            if dest in ["", -1]:  # unassigned, poll for current value
                self._state[key] = fi.get()
            else:
                try:  # some things we cannot write to, even though we should
                    fi.set(dest)
                except Exception as e:
                    self.logger.error(e)
            # generate avro properties
            self.__setattr__(f"set_{key}", self.gen_setter(key))
            self.__setattr__(f"get_{key}", self.gen_getter(key))
            if self.features[key].type in ["int", "float"]:
                self.__setattr__(f"get_{key}_limits", self.gen_limits_getter(key))
            elif self.features[key].type in ["enumerated"]:
                self.__setattr__(f"get_{key}_options", self.gen_options_getter(key))

    async def _measure(self):
        image_size_bytes = self.features["image_size_bytes"].get()
        buf = np.empty((image_size_bytes,), dtype="B")
        timeout = max(self.features["exposure_time"].get() * 2e3, 100)
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

        stride = self.features["aoi_stride"].get()
        pixels = np.lib.stride_tricks.as_strided(
            np.frombuffer(buf, dtype=np.uint16),
            shape=self._channel_shapes["image"],
            strides=(stride, 2),  # binning works?
        )
        self.logger.debug(f"{pixels.size}, {np.prod(self._channel_shapes['image'])}")
        pixels = np.ascontiguousarray(pixels)
        self.sdk.flush(self.hndl)

        return {"image": pixels}

    def get_sensor_info(self):
        return self.sensor_info

    def get_feature_names(self) -> List[str]:
        return [f"{k} -> {v.sdk_name}" for k, v in self.features.items()]

    def get_feature_type(self, k: str):
        return self.features[k].type

    def get_feature_value(self, k: str) -> Union[int, bool, float, str]:
        feature = self.features[k]
        return feature.get()

    def get_feature_options(self, k: str) -> List[str]:  # -> List[Union[str, float, int]]:
        feature = self.features[k]
        if feature.type == "enumerated":  # isinstance(feature, features.SDKEnum):
            return feature.options()
        else:
            raise ValueError(f"feature {feature} is of type {feature.type}.  No options.")

    def get_feature_limits(self, k: str) -> List[Union[float, int]]:
        feature = self.features[k]
        if feature.type in ["int", "float"]:
            return [feature.min(), feature.max()]
        raise ValueError(f"feature {feature} is of type {feature.type}.  No limits.")

    def close(self):
        self.sdk.close(self.hndl)

    def _set_feature_by_key(self, key, val):
        self._loop.create_task(self._aset_feature_by_key(key, val))

    async def _aset_feature_by_key(self, key, val):
        if self._busy:
            await asyncio.wait_for(self._not_busy_sig.wait())
        self.features[key].set(val)
        self._state[key] = self.features[key].get()

    def gen_setter(self, key):
        def setter(val: Any):
            self._set_feature_by_key(key, val)

        return setter

    def gen_getter(self, key):
        def getter() -> Any:
            return self._state[key]

        return getter

    def gen_limits_getter(self, key):
        def getter() -> List[float]:
            return [self.features[key].min(), self.features[key].max()]

        return getter

    def gen_options_getter(self, key):
        def getter() -> str:
            return self.features[key].options()

        return getter
