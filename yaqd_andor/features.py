from collections import namedtuple

FeatureSpec = namedtuple("Feature", ["sdk_name", "type", "availability"])

# adapted from sdk3 manual
# note some table entries were omitted (focused on neo, simcam features)
# a = apogee, n = neo, s = simcam, z = zyla
feature_specs = dict(
    acquisition_start = FeatureSpec("AcquisitionStart", "command", "nsz"),
    acquisition_stop = FeatureSpec("AcquisitionStop", "command", "nsz"),
    aoi_binning = FeatureSpec("AOIBinning", "enumerated", "nz"),
    aoi_hbin = FeatureSpec("AOIHBin", "integer", "asz"),
    aoi_height = FeatureSpec("AOIHeight", "integer", "ansz"),
    aoi_layout = FeatureSpec("AOILayout", "enumerated", "az"),
    aoi_left = FeatureSpec("AOILeft", "integer", "ansz"),
    aoi_stride = FeatureSpec("AOIStride", "integer", "nz"),
    aoi_top = FeatureSpec("AOITop", "integer", "ansz"),
    aoi_vbin = FeatureSpec("AOIVBin", "integer", "asz"),
    aoi_width = FeatureSpec("AOIWidth", "integer", "ansz"),
    baseline = FeatureSpec("Baseline", "integer", "nz"),
    bit_depth = FeatureSpec("BitDepth", "enumerated", "anz"),
    buffer_overflow_event = FeatureSpec("BufferOverflowEvent", "integer", "nz"),
    bytes_per_pixel = FeatureSpec("BytesPerPixel", "float", "nz"),
    camera_acquiring = FeatureSpec("CameraAcquiring", "boolean", "nsz"),
    camera_dump = FeatureSpec("CameraDump", "command", "nz"),
    camera_family = FeatureSpec("CameraFamily", "string", "a"),
    camera_memory = FeatureSpec("CameraMemory", "integer", "a"),
    camera_model = FeatureSpec("CameraModel", "string", "nsz"),
    camera_name = FeatureSpec("CameraName", "string", "anz"),
    cycle_mode = FeatureSpec("CycleMode", "enumerated", "nsz"),
    electronic_shuttering_mode = FeatureSpec("ElectronicShutteringMode", "enumerated", "nsz"),
    event_enable = FeatureSpec("EventEnable", "boolean", "nz"),
    events_missed_event = FeatureSpec("EventsMissedEvent", "integer", "nz"),
    event_selector = FeatureSpec("EventSelector", "enumerated", "nz"),
    exposure_time = FeatureSpec("ExposureTime", "float", "nsz"),
    exposure_end_event = FeatureSpec("ExposureEndEvent", "integer", "nz"),
    exposure_start_event = FeatureSpec("ExposureStartEvent", "integer", "nz"),
    external_trigger_delay = FeatureSpec("ExposureTriggerDelay", "float", "z"),
    fan_speed = FeatureSpec("FanSpeed", "enumerated", "ansz"),
    # fast_aoi_frame_rate_enable = FeatureSpec("FastAOIFrameRateEnable", "boolean", "nz"),
    firmware_version = FeatureSpec("FirmwareVersion", "string", "anz"),
    frame_count = FeatureSpec("FrameCount", "integer", "ansz"),
    frame_rate = FeatureSpec("FrameInterval", "float", "ansz"),
    full_aoi_control = FeatureSpec("FullAOIControl", "boolean", "nz"),
    image_size_bytes = FeatureSpec("ImageSizeBytes", "integer", "nsz"),
    interface_type = FeatureSpec("InterfaceType", "string", "anz"),
    max_interface_transfer_rate = FeatureSpec("MaxInterfaceTransferRate", "float", "nz"),
    metadata_enable = FeatureSpec("MetadataEnable", "boolean", "nz"),
    metadata_frame = FeatureSpec("MetadataFrame", "boolean", "nz"),
    metadata_timestamp = FeatureSpec("MetadataTimestamp", "boolean", "nz"),
    overlap = FeatureSpec("Overlap", "boolean", "anz"),
    pixel_correction = FeatureSpec("PixelCorrection", "enumerated", "s"),
    pixel_encoding = FeatureSpec("PixelEncoding", "enumerated", "nsz"),
    pixel_height = FeatureSpec("PixelHeight", "enumerated", "nsz"),
    pixel_readout_rate = FeatureSpec("PixelReadoutRate", "enumerated", "ansz"),
    pixel_width = FeatureSpec("PixelWidth", "float", "ansz"),
    readout_time = FeatureSpec("ReadoutTime", "float", "nz"),
    sensor_cooling = FeatureSpec("SensorCooling", "boolean", "ansz"),
    sensor_height = FeatureSpec("SensorHeight", "integer", "ansz"),
    sensor_readout_mode = FeatureSpec("SensorReadoutMode", "enumerated", "z"),
    sensor_temperature = FeatureSpec("SensorTemperature", "float", "ansz"),
    sensor_width = FeatureSpec("SensorWidth", "integer", "ansz"),
    serial_number = FeatureSpec("SerialNumber", "string", "snz"),
    simple_preamp_gain_control = FeatureSpec("SimplePreampGainControl", "enumerated", "nz"),
    software_trigger = FeatureSpec("SoftwareTrigger", "command", "nz"),
    spurious_noise_filter = FeatureSpec("SpuriousNoiseFilter", "boolean", "nz"),
    temperature_control = FeatureSpec("TemperatureControl", "enumerated", "n"),
    temperature_status = FeatureSpec("TemperatureStatus", "enumerated", "anz"),
    timestamp_clock = FeatureSpec("TimestampClock", "integer", "nz"),
    timestamp_clock_frequency = FeatureSpec("TimestampClockFrequency", "integer", "nz"),
    timestamp_clock_reset = FeatureSpec("TimestampClockReset", "command", "nz"),
    trigger_mode = FeatureSpec("TriggerMode", "enumerated", "ansz"),
    vertically_center_aoi = FeatureSpec("VerticallyCentreAOI", "boolean", "nz")
)

shared_features = [k for k,v in feature_specs.items() if \
    ("n" in v.availability) and ("s" in v.availability) and ("z" in v.availability)]
simcam_features = [k for k,v in feature_specs.items() if "s" in v.availability]
neo_features = [k for k,v in feature_specs.items() if "n" in v.availability]

if False:
    def generate_getter(self, call:str):
        def getter(self):
            if self.sdk.is_readable(self.hndl, self.sdk_name):
                return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name)
            else:
                raise ValueError(f"call {call} is not currently readable")
        return getter

    def generate_setter(self, call:str):
        def setter(self, value):
            if self.sdk.is_writable(self.hndl, self.sdk_name):
                return self.sdk.__getattribute__(call)(self.hndl, self.sdk_name, value)
            else:
                raise ValueError(f"call {call} is not currently writable")
        return setter


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
        self.method = r"{}" + f"_{type}"
        # feature implemented?
        self.is_implemented = self.sdk.is_implemented(self.hndl, self.sdk_name)
        if self.is_implemented:
                self.get = self._get
        else:
            raise NotImplementedError(f"feature {self.sdk_name} is not implemented")
        # is read only?
        self.is_readonly = self.sdk.is_readonly(self.hndl, self.sdk_name)
        if not self.is_readonly:
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
        del self._set
        del self._get

    def __call__(self):
        return self.sdk.command(self.hndl, self.sdk_name)


class SDKEnum(Feature):
    def __init__(self, *args):
        super().__init__(*args)
        self._get_call = self._get_call + "_string"
        self._set_call = self._set_call + "_string"

    def options(self):
        """query available feature string options
        """
        return self.sdk.get_enumerated_string_options(self.hndl, self.sdk_name)

