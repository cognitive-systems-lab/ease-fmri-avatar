import multiprocessing as mp
import time
import logging
from pathlib import Path
import random

import pylsl
import labc



log = logging.getLogger(__name__)

BUTTON_LEFT_LEFT = 1 # 254
BUTTON_LEFT_MIDDLE = 2 # 253
BUTTON_LEFT_RIGHT = 4 # 251
BUTTON_RIGHT_LEFT = 8 # 247
BUTTON_RIGHT_MIDDLE = 16 # 239
BUTTON_RIGHT_RIGHT = 32 # 223

MINIMAL_BUTTON_PRESS_TIME = 0.1 # sconds



# Create Mockup Classes

class Port:
    def set_data_dir(self, d):
        self.data_dir = d

    def getInAcknowledge(self):
        return random.randint(0, 1)

    def getData(self):
        return random.randint(0, 2**32)


class parallel:
    class ParallelPort:
        def __init__(self, address):
            self.address = address
            self.port = Port()


@labc.register_service
class TriggerAndButtonsSimulator:

    def __init__(self, address=0, sample_rate=2000.0):
        self.start_event = mp.Event() # if set service is started
        self.stop_event = mp.Event() # if set service is in the process of stopping
        self.sample_rate = sample_rate # Hz
        self.address = address

    @labc.register
    def start(self):
        if self.start_event.is_set():
            log.warning("Service already started !!")
            return
        self.start_event.set()
        p = mp.Process(
            target = self.run,
            args = (self.stop_event, self.sample_rate, self.address),
        )
        p.start()

    @labc.register
    def stop(self):
        self.stop_event.set()
        self.stop_event.wait()
        self.start_event.clear()

    @labc.register
    def set_recording_parameters(self, path, file_prefix):
        self.recording_path = path
        self.recording_file_prefix = file_prefix

    @staticmethod
    def run(stop_event, sample_rate, address):

        sample_duration = 1 / sample_rate

        log.info(f"Opening parallel port {address}.")
        pp = parallel.ParallelPort(address)
        pp.port.setDataDir(0) # this is requiered because Peter said so

        press_time = {k: 0 for k in "ll lm lr rl rm rr".split()}

        button_stream_info = pylsl.StreamInfo(
            name='button_events',
            type='Marker',
            channel_count=3,
            channel_format='string',
        )
        button_outlet = pylsl.StreamOutlet(button_stream_info)

        trigger_stream_info = pylsl.StreamInfo(
            name='mrt_trigger_events',
            type='Marker',
            channel_count=1,
            channel_format='string',
        )
        trigger_outlet = pylsl.StreamOutlet(trigger_stream_info)

        # local trigger recording
        recording_file_prefix = "local-trigger"
        filepath = Path("~/data/").expanduser()
        filename = filepath / (recording_file_prefix + "_" + str(time.time()) + ".csv")
        local_recording_file = open(filename, "a")

        old_ack = pp.port.getInAcknowledge()
        old_data = pp.port.getData()

        while not stop_event.is_set():
            tic = pylsl.local_clock() # relative time ascending ts in seconds

            # handle button input
            data = pp.port.getData()
            press = (data ^ old_data) & ~data
            release = (data ^ old_data) & ~old_data

            if press & BUTTON_LEFT_LEFT:
                if press_time["ll"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["press", "left", "left"], tic)
                    labc.network.sound_service.play << "left"
                    press_time["ll"] = tic

            if press & BUTTON_LEFT_MIDDLE:
                if press_time["lm"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["press", "left", "middle"], tic)
                    labc.network.sound_service.play << "left"
                    press_time["lm"] = tic

            if press & BUTTON_LEFT_RIGHT:

                if press_time["lr"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["press", "left", "right"], tic)
                    labc.network.sound_service.play << "left"
                    press_time["lr"] = tic

            if press & BUTTON_RIGHT_LEFT:
                if press_time["rl"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["press", "right", "left"], tic)
                    labc.network.sound_service.play << "right"
                    press_time["rl"] = tic

            if press & BUTTON_RIGHT_MIDDLE:
                if press_time["rm"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["press", "right", "middle"], tic)
                    labc.network.sound_service.play << "left"
                    press_time["rm"] = tic

            if press & BUTTON_RIGHT_RIGHT:
                if press_time["rr"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["press", "right", "right"], tic)
                    labc.network.sound_service.play << "right"
                    press_time["rr"] = tic

            if release & BUTTON_LEFT_LEFT:
                if press_time["ll"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["release", "left", "left"], tic)
                    press_time["ll"] = tic

            if release & BUTTON_LEFT_MIDDLE:
                if press_time["lm"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["release", "left", "middle"], tic)
                    press_time["lm"] = tic

            if release & BUTTON_LEFT_RIGHT:
                if press_time["lr"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["release", "left", "right"], tic)
                    press_time["lr"] = tic

            if release & BUTTON_RIGHT_LEFT:
                if press_time["rl"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["release", "right", "left"], tic)
                    press_time["rl"] = tic

            if release & BUTTON_RIGHT_MIDDLE:
                if press_time["rm"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["release", "right", "middle"], tic)
                    press_time["rm"] = tic

            if release & BUTTON_RIGHT_RIGHT:
                if press_time["rr"] + MINIMAL_BUTTON_PRESS_TIME < tic:
                    button_outlet.push_sample(["release", "right", "right"], tic)
                    press_time["rr"] = tic

            old_data = data

            # handle button input
            ack = pp.port.getInAcknowledge()
            if ack ^ old_ack:
                if ack:
                    trigger_outlet.push_sample(["on"])
                    local_recording_file.write(str(tic) + ',on\n')
                else:
                    trigger_outlet.push_sample(["off"])
                    local_recording_file.write(str(tic) + ',off\n')

            old_ack = ack

            # sleep till next circle
            toc = time.perf_counter()
            sleep_time = max(sample_duration - (toc - tic), 0.0)
            time.sleep(sleep_time)

        local_recording_file.close()
        stop_event.clear()
