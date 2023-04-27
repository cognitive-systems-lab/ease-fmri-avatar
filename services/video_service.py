"""
Specification:
--------------
 - lsl stream server side
 - lsl stream client side
 - video recording client side
 - video recording server side
"""

import multiprocessing
from pathlib import Path
from itertools import count
import logging
import enum

import matplotlib.pyplot as plt
import numpy as np
import qrcode
from cv2 import QRCodeDetector

from vidgear.gears import NetGear, CamGear
import cv2
import pylsl
from labc import(
    register,
    register_service,
    network,
)
log = logging.Logger(__name__)


def create_clapperboard(frame, corners = [1, 43, 97], id = 0, block_size = 16):
    frame[0:block_size,0:block_size,:] = corners[0]
    frame[-block_size:, 0:block_size, :] = corners[1]
    frame[-block_size:, -block_size:, :] = corners[2]
    qr = qrcode.QRCode(
        version=1,
        box_size=2,
        border=2)
    qr.add_data(f"{id}")
    qr.make(fit=True)
    img = qr.make_image(fill='black', back_color='white')
    frame[0:50, -50:, :] = np.asarray(img.convert("RGB"))
    return frame


def check_clapperboard(frame, corners = [1, 43, 97], block_size = 8, qr_size = 50, tolerance_std = 0.1, tolerance_mean= 5):
    if np.mean(np.abs(frame[0:block_size,0:block_size,:] - corners[0]))  > tolerance_mean and np.std(np.abs(frame[0:block_size,0:block_size,:])) > tolerance_std:
        return -1
    if np.mean(np.abs(frame[-block_size:, 0:block_size, :] - corners[1])) > tolerance_mean and np.std(np.abs(frame[-block_size:, 0:block_size, :])) > tolerance_std:
        return -1
    if np.mean(np.abs(frame[-block_size:, -block_size:, :] - corners[2])) > tolerance_mean and np.std(np.abs(frame[-block_size:, -block_size:, :])) > tolerance_std:
        return -1
    qrCodeDetector = cv2.QRCodeDetector()
    decodedText, points, _ = qrCodeDetector.detectAndDecode(frame[0:qr_size, -qr_size:, :])
    qr_data = decodedText.split(',')
    return int(qr_data[0])


def parse_addr(addr):
    print(addr)
    if isinstance(addr, str):
        addr = addr.split(":")
        if len(addr) == 1:
            host = addr[0]
            port = 0
        else:
            host, port = addr
    elif isinstance(addr, (tuple, list)):
        host, port = addr
    port = int(port)
    assert port >= 0
    assert isinstance(host, str), f"'host' is not a string: {host} {type(host)}"
    return (host, port)


def create_filename_paths(config):
    print(config)
    print(config["video_file_extension"])
    for i in count():
        video_file_name = config["video_file_template"].format(
            iteration=i, type="data", file_extension=config["video_file_extension"], **config)
        data_path = Path(config["data_dir"]).expanduser().resolve()
        video_file_path = data_path / video_file_name
        if not video_file_path.exists():
            break
    video_ts_file_path = data_path / config["video_file_template"].format(
            iteration=i, type="ts", file_extension="csv", **config)
    video_ts_clapperboard_file_path = data_path / config["video_file_template"].format(
            iteration=i, type="ts_clapperboard", file_extension="csv", **config)
    return video_file_path, video_ts_file_path, video_ts_clapperboard_file_path


class Status(enum.IntEnum):
    STARTING = 0
    STARTED = 1
    STOPPING = 2
    STOPPED = 3


@register_service
class CameraStreamer:
    """Creates a stream from a camera source."""

    DEFAULT_CONF = {
        "camera_id": 0,
        "target_addr": "0.0.0.0",
        "port": None,
        "max_retries": 100,
        "request_timeout": 5,
    }

    def __init__(self):
        self.config = multiprocessing.Manager().dict(CameraStreamer.DEFAULT_CONF)
        self.streaming_status = multiprocessing.Value("i", Status.STOPPED)
        self.clapperboard = multiprocessing.Value("i", -1)

    @register
    def is_running(self):
        return self.streaming_status.value in (Status.STARTING, Status.STARTED)

    @register
    async def stop(self):
        self.streaming_status.value = Status.STOPPING

    @register
    def set_config(self, **kwargs):
        self.config.update(kwargs)

    @register
    def get_config(self):
        return dict(self.config)

    @register
    def start(self):
        """
        source: (integer) index of /dev/video
        """
        self.streaming_status.value = Status.STARTING
        process = multiprocessing.Process(
            name='video_streamer',
            target=self.run,
            args=(self.streaming_status, self.clapperboard, self.config),
        )
        process.start()

    @register
    async def clap(self, index):
        self.clapperboard.value = index

    @staticmethod
    def run(streaming_status, clapperboard, config):

        options = {
            "CAP_PROP_FRAME_WIDTH": config["width"],
            "CAP_PROP_FRAME_HEIGHT": config["height"],
            "CAP_PROP_FPS": int(config["framerate"]), # TODO: does this need to be integer ?
            "CAP_PROP_FOURCC": cv2.VideoWriter_fourcc(*config["codec"]),
        }

        video_capture = CamGear(source=int(config["camera_id"]), logging=False, **options).start() #Open any video stream
        host, port = parse_addr(config["target_addr"])
        video_streamer = NetGear(
            address=host,
            receive_mode=False,
            logging=False,
            max_retries=config["max_retries"],
            request_timeout=config["request_timeout"],
        )

        # LSL Streams
        stream_info_capture = pylsl.StreamInfo(
            name = 'camera_capture_events',
            type = 'timestamps',
            channel_count = 1,
            channel_format = 'float32', # timestamps are always float32
            source_id="CameraStreamer",
        )
        lsl_outlet_capture = pylsl.StreamOutlet(stream_info_capture)

        stream_info_clapperboard = pylsl.StreamInfo(
            name = 'camera_clapperboard_events',
            type = 'id',
            channel_count = 1,
            channel_format = 'int16',
            source_id="CameraStreamer",
        )
        lsl_outlet_clapperboard = pylsl.StreamOutlet(stream_info_clapperboard)

        log.info("Camera frame rate:", )

        streaming_status.value = Status.STARTED
        log.info(f"Camera streams to {host}:{port}.")
        try:
            while streaming_status.value not in (Status.STOPPED, Status.STOPPING):

                frame = video_capture.read()
                ts = pylsl.local_clock()

                if frame is None:
                    log.error("No Frame received from Camera. Cancel capturing!")
                    break

                # check for clapperboard
                if clapperboard.value >= 0:
                    frame = create_clapperboard(frame, id = clapperboard.value)
                    lsl_outlet_clapperboard.push_sample((clapperboard.value,), timestamp=ts)
                    clapperboard.value = -1

                # TODO: this should be dependent on whether the stream is consumed
                video_streamer.send(frame)
                lsl_outlet_capture.push_sample((ts,), timestamp=ts)

        finally:
            video_capture.stop()
            video_streamer.close()
            streaming_status.value = Status.STOPPED
            log.info("Camera stream stopped.")
            # TODO: close lsl-stream here ???


@register_service
class StreamViewer:
    """Consumes a camera stream and plays it on a screen"""

    DEFAULT_CONF = {
        "addr": "0.0.0.0",
        "max_retries": 100,
        "request_timeout": 5,
        "data_dir": "~/data/lablink_video_recording",
        "recording_name": "experiment",
        "video_file_extension": "mkv",
        "video_file_template": "headcam.{recording_name}.{type}.{iteration}.{file_extension}",
        "codec": "MJPG",
        "record_fps": 30,
        "diagnostics": False,
    }

    def __init__(self):
        self.addr = "0.0.0.0"
        self.stop_event = multiprocessing.Event()
        self.config = multiprocessing.Manager().dict(StreamViewer.DEFAULT_CONF)
        self.recording_status = multiprocessing.Value("i", Status.STOPPED)

    @register
    async def set_config(self, **kwargs):
        self.config.update(kwargs)

    @register
    async def get_config(self):
        return dict(self.config)

    @register
    def is_recording(self):
        return self.recording_status.value in (Status.STARTING, Status.STARTED)

    @register
    async def start_recording(self):
        self.recording_status.value = Status.STARTING

    @register
    async def stop_recording(self):
        self.recording_status.value = Status.STOPPING

    @register
    async def start(self):
        """
        source: (integer) index of /dev/video
        """
        process = multiprocessing.Process(
            name='video_viewer',
            target=StreamViewer.run_viewing,
            args=(self.addr, self.stop_event, self.recording_status, self.config)
        )
        process.start()

    @register
    async def get_address(self):
        ...

    @register
    async def stop(self):
        self.stop_event.set()

    @register
    async def is_running(self):
        return self.stop_event.is_set()

    @staticmethod
    def run_viewing(addr, stop_event, recording_status, config):
        log.info(f"Listening to video stream on {config['addr']}")
        client = NetGear(
            address=config["addr"],
            receive_mode=True,
            logging=True,
            max_retries=config["max_retries"],
            request_timeout=config["request_timeout"],
        )
        stream_info_receive = pylsl.StreamInfo(
            name='stream_receive_events',
            type='timestamps',
            channel_count=1,
            channel_format='float32',  # timestamps are always float32
            source_id="StreamViewer",
        )
        lsl_outlet_receive = pylsl.StreamOutlet(stream_info_receive)

        stream_info_clapperboard_receive = pylsl.StreamInfo(
            name = 'camera_clapperboard_receive_events',
            type = 'id',
            channel_count = 1,
            channel_format = 'int16',
            source_id="CameraStreamer",
        )
        lsl_outlet_clapperboard = pylsl.StreamOutlet(stream_info_clapperboard_receive)

        if config["diagnostics"]:
            info = pylsl.StreamInfo('LSL_Diagnostics_2', 'Markers', 1, 0, 'string', "StreamViewer")
            marker_outlet = pylsl.StreamOutlet(info)

        try:
            while not stop_event.is_set():
                # receive frames from network
                frame = client.recv()
                ts = pylsl.local_clock()
                lsl_outlet_receive.push_sample((ts,), timestamp=ts)

                # check if frame is None
                if frame is None:
                    break

                clap_id = check_clapperboard(frame)
                if clap_id >= 0:
                    lsl_outlet_clapperboard.push_sample((clap_id,), timestamp=ts)

                # Show output window
                cv2.imshow("Output Frame", frame)

                key = cv2.waitKey(1) & 0xFF

                # check for 'q' key-press
                if key == ord("q"):
                    #if 'q' key-pressed break out
                    break
                elif key == ord("r"):
                    network.sound_service.play << ("right", "de")
                    if config["diagnostics"]:
                        marker_outlet.push_sample(("response_stim",), timestamp=ts)
                    log.info("Pressed right button.")
                elif key == ord("l") and config["diagnostics"]:
                    network.sound_service.play << ("left", "de")
                    if config["diagnostics"]:
                        marker_outlet.push_sample(("response_stim",), timestamp=ts)
                    log.info("Pressed left button.")
                else:
                    log.info("Pressed invalid button.")

                # Video Recording
                if recording_status.value == Status.STARTING:
                    # setup video recorder
                    video_file_path, ts_file_path, video_ts_clapperboard_file_path = create_filename_paths(config)
                    Path(config["data_dir"]).expanduser().mkdir(parents=True, exist_ok=True)
                    fourcc = cv2.VideoWriter_fourcc(*config["codec"])
                    res = (frame.shape[1], frame.shape[0])
                    print("resolution", res)
                    video_writer = cv2.VideoWriter(str(video_file_path), fourcc, config["framerate"], res)
                    ts_file = open(ts_file_path, 'w')
                    ts_file_clapperboard = open(video_ts_clapperboard_file_path, 'w')
                    recording_status.value = Status.STARTED

                if recording_status.value == Status.STOPPING:
                    ts_file.close()
                    ts_file_clapperboard.close()
                    video_writer.release()
                    recording_status.value = Status.STOPPED

                # save timestamp and frame locally
                if recording_status.value == Status.STARTED:
                    ts_file.write(str(ts) + '\n')
                    if clap_id >= 0:
                        ts_file_clapperboard.write(str(ts) + '\t' + str(clap_id) + '\n')
                    video_writer.write(frame)
        finally:
            cv2.destroyAllWindows()
            client.close()
            stop_event.clear()
            if recording_status.value != Status.STOPPED:
                recording_status.value = Status.STOPPED
            if "ts_file" in locals():
                ts_file.close()
            if "video_writer" in locals():
                video_writer.release()
            log.info("Close StreamViewer.")
