from collections import namedtuple

Spec = namedtuple("Feature", ["sdk_name", "type", "availability"])

# adapted from sdk3 manual
# note some table entries were omitted (focused on neo, simcam features)
# a = apogee, n = neo, s = simcam, z = zyla
specs = dict(
    acquisition_start = Spec("AcquisitionStart", "command", "nsz"),
    acquisition_stop = Spec("AcquisitionStop", "command", "nsz"),
    aoi_binning = Spec("AOIBinning", "enumerated", "nz"),
    aoi_hbin = Spec("AOIHBin", "int", "asz"),
    aoi_height = Spec("AOIHeight", "int", "ansz"),
    aoi_layout = Spec("AOILayout", "enumerated", "az"),
    aoi_left = Spec("AOILeft", "int", "ansz"),
    aoi_stride = Spec("AOIStride", "int", "nsz"),  # not in docs, but implemented for simcam
    aoi_top = Spec("AOITop", "int", "ansz"),
    aoi_vbin = Spec("AOIVBin", "int", "asz"),
    aoi_width = Spec("AOIWidth", "int", "ansz"),
    baseline = Spec("Baseline", "int", "nz"),
    bit_depth = Spec("BitDepth", "enumerated", "anz"),
    buffer_overflow_event = Spec("BufferOverflowEvent", "int", "nz"),
    bytes_per_pixel = Spec("BytesPerPixel", "float", "nz"),
    camera_acquiring = Spec("CameraAcquiring", "bool", "nsz"),
    camera_dump = Spec("CameraDump", "command", "nz"),
    camera_family = Spec("CameraFamily", "string", "a"),
    camera_memory = Spec("CameraMemory", "int", "a"),
    camera_model = Spec("CameraModel", "string", "nsz"),
    camera_name = Spec("CameraName", "string", "anz"),
    cycle_mode = Spec("CycleMode", "enumerated", "nsz"),
    electronic_shuttering_mode = Spec("ElectronicShutteringMode", "enumerated", "nsz"),
    event_enable = Spec("EventEnable", "bool", "nz"),
    events_missed_event = Spec("EventsMissedEvent", "int", "nz"),
    event_selector = Spec("EventSelector", "enumerated", "nz"),
    exposure_time = Spec("ExposureTime", "float", "nsz"),
    exposure_end_event = Spec("ExposureEndEvent", "int", "nz"),
    exposure_start_event = Spec("ExposureStartEvent", "int", "nz"),
    external_trigger_delay = Spec("ExposureTriggerDelay", "float", "z"),
    fan_speed = Spec("FanSpeed", "enumerated", "ansz"),
    # fast_aoi_frame_rate_enable = Spec("FastAOIFrameRateEnable", "bool", "nz"),
    firmware_version = Spec("FirmwareVersion", "string", "anz"),
    frame_count = Spec("FrameCount", "int", "ansz"),
    frame_rate = Spec("FrameInterval", "float", "ansz"),
    full_aoi_control = Spec("FullAOIControl", "bool", "nz"),
    image_size_bytes = Spec("ImageSizeBytes", "int", "nsz"),
    interface_type = Spec("InterfaceType", "string", "anz"),
    max_interface_transfer_rate = Spec("MaxInterfaceTransferRate", "float", "nz"),
    metadata_enable = Spec("MetadataEnable", "bool", "nz"),
    metadata_frame = Spec("MetadataFrame", "bool", "nz"),
    metadata_timestamp = Spec("MetadataTimestamp", "bool", "nz"),
    overlap = Spec("Overlap", "bool", "anz"),
    pixel_correction = Spec("PixelCorrection", "enumerated", "s"),
    pixel_encoding = Spec("PixelEncoding", "enumerated", "nsz"),
    pixel_height = Spec("PixelHeight", "enumerated", "nsz"),
    pixel_readout_rate = Spec("PixelReadoutRate", "enumerated", "ansz"),
    pixel_width = Spec("PixelWidth", "float", "ansz"),
    readout_time = Spec("ReadoutTime", "float", "nz"),
    sensor_cooling = Spec("SensorCooling", "bool", "ansz"),
    sensor_height = Spec("SensorHeight", "int", "ansz"),
    sensor_readout_mode = Spec("SensorReadoutMode", "enumerated", "z"),
    sensor_temperature = Spec("SensorTemperature", "float", "ansz"),
    sensor_width = Spec("SensorWidth", "int", "ansz"),
    serial_number = Spec("SerialNumber", "string", "snz"),
    simple_preamp_gain_control = Spec("SimplePreampGainControl", "enumerated", "nz"),
    software_trigger = Spec("SoftwareTrigger", "command", "nz"),
    spurious_noise_filter = Spec("SpuriousNoiseFilter", "bool", "nz"),
    static_blemish_correction = Spec("StaticBlemishCorrection", "bool", "nz"),
    temperature_control = Spec("TemperatureControl", "enumerated", "n"),
    temperature_status = Spec("TemperatureStatus", "enumerated", "anz"),
    timestamp_clock = Spec("TimestampClock", "int", "nz"),
    timestamp_clock_frequency = Spec("TimestampClockFrequency", "int", "nz"),
    timestamp_clock_reset = Spec("TimestampClockReset", "command", "nz"),
    trigger_mode = Spec("TriggerMode", "enumerated", "ansz"),
    vertically_center_aoi = Spec("VerticallyCentreAOI", "bool", "nz")
)


class Feature:
    def __init__(self, sdk, hndl, spec):
        """
        command:  feature()
        get: feature.get()
        set: feature.set(value)
        """
        self.type = spec.type
        self.sdk_name = spec.sdk_name
        self.sdk = sdk
        self.hndl = hndl
        # feature implemented?
        self.is_implemented = bool(self.sdk.is_implemented(self.hndl, self.sdk_name))
        if self.is_implemented:
                self.get = self._get
        else:
            # DDK: not implemented = never available?
            raise NotImplementedError(f"feature {self.sdk_name} is not implemented")
        # is read only?
        self.is_readonly = bool(self.sdk.is_readonly(self.hndl, self.sdk_name))
        if not self.is_readonly:
            self._set_call = f"set_{self.type}"
            self.set = self._set
        self._get_call = f"get_{self.type}"

    def _get(self):
        call = self._get_call
        if self.sdk.is_readable(self.hndl, self.sdk_name):
            return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)
        else:
            raise TypeError(f"call {call} is not readable")

    def _set(self, value):
        call = self._set_call
        if self.sdk.is_writable(self.hndl, self.sdk_name):
            return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name, value)
        else:
            raise ValueError(f"call {call} is not currently writable")    


class SDKString(Feature):
    def __init__(self, *args):
        super().__init__(*args)


class SDKFloat(Feature):
    def __init__(self, *args):
        super().__init__(*args)

    def max(self) -> float:
        call = self.get_call + "_max"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)

    def min(self) -> float:
        call = self.get_call + "_min"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)


class SDKInt(Feature):
    def __init__(self, *args):
        super().__init__(*args)
        
    def max(self) -> int:
        call = self._get_call + "_max"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)

    def min(self) -> int:
        call = self._get_call + "_min"
        return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)


class SDKBool(Feature):
    def __init__(self, *args):
        super().__init__(*args)


class SDKCommand(Feature):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self):
        return self.sdk.command(self.hndl, self.sdk_name)


class SDKEnum(Feature):
    def __init__(self, *args):
        super().__init__(*args)
        self._get_call = self._get_call + "_string"
        if not self.is_readonly:
            self._set_call = self._set_call + "_string"

    def options(self):
        """query available feature string options
        """
        return self.sdk.get_enumerated_string_options(self.hndl, self.sdk_name)


def obj_from_spec(sdk, hndl, spec):
    if spec.type == "command":
        return SDKCommand(sdk, hndl, spec)
    elif spec.type == "bool":
        return SDKBool(sdk, hndl, spec)
    elif spec.type == "enumerated":
        return SDKEnum(sdk, hndl, spec)
    elif spec.type == "float":
        return SDKFloat(sdk, hndl, spec)
    elif spec.type == "int":
        return SDKInt(sdk, hndl, spec)
    elif spec.type == "string":
        return SDKString(sdk, hndl, spec)
    else:
        print(f"failed to find valid type for {spec}.")
        pass