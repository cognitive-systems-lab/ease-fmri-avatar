#! /usr/bin/python3
#
from labc import network
import logging
import time
from pathlib import Path

log = logging.Logger(__name__)

print("Possible Actions: start / stop | stream / record")

video_client_ip = "127.0.0.1"

# tell video to stream to Computer in Skyra-Lab
network.camera_streamer.set_config(
    target_addr=video_client_ip,
    camera_id=4,
)

"""
if 'subject' not in params:
    params['subject'] = 'subject0'
if 'trial' not in params:
    params['trial'] = 'trial0'
if 'experiment' not in params:
    params['experiment'] = 'test'
if 'path' not in params:
    params['path'] = "/home/mome/data/recordings/{experiment}/{subject}/{trial}".format(**params)
"""

recording_path = Path("~/data/lablink_stream_recording").expanduser()

network.lsl_recorder.set_params(
    path=str(recording_path)
)

clap_idx = 0
while True:
    cmd = input("Enter next action:")

    if cmd in ("q", "quit"):
        break

    try:
        first, second = cmd.split()
    except:
        print("Invalid command!")
        continue
    if first == "start":

        if second in ("s", "str", "stream", "streaming"):
            log.info("Start video stream.")
            network.camera_streamer.start()
            network.stream_viewer.start()

        elif second in ("r", "rec", "record", "recording"):
            log.info("Start recording!")
            network.stream_viewer.start_recording()
            #time.sleep(3)
            #network.lsl_recorder.start()

        elif second in ("clap"):
            network.camera_streamer.clap(clap_idx)
            clap_idx += 1

        else:
            print("Invalid second argument:", second)

    elif first == "stop":

        if second in ("s", "str", "stream", "streaming"):
            log.info("Stopvideo stream.")
            network.camera_streamer.stop()
            network.stream_viewer.stop()

        elif second in ("r", "rec", "record", "recording"):
            log.info("Stop recording!")
            #network.lsl_recorder.stop()
            #time.sleep(3)
            network.stream_viewer.stop_recording()

        else:
            print("Invalid second argument:", second)

    else:
        print("Invalid first argument:", first)

