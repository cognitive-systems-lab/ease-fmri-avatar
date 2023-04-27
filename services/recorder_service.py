from os import environ
import time
import logging
from pathlib import Path

from lsl_tools.recording import Recorder
from lsl_tools import configuration as c

from labc import register, subscribe, register_service

log = logging.getLogger(__name__)

module_path = Path(__file__).parent


@register_service
class LslRecorder:
    """
    An application component providing procedures with different kinds
    of arguments.
    """
    def __init__(self):
        self.recorder = None
        self.params = {}

    @register
    def set_params(self, **kwargs):
        self.params.update(kwargs)

    @register
    def start(self):
        assert self.recorder is None, "There is already a recorder running."
        c.add_path(str(module_path / "recorder_conf.yaml"))
        c.reload_conf()
        config = c.get('recorder')
        print(config)
        self.recorder = Recorder(config=config, extra_params=self.params)
        self.recorder.configure()
        print("CONFIGURE DONE")
        self.recorder.start()
        log.info('Recorder started. !!!!')

    @register
    def stop(self):
        self.recorder.stop()
        log.info('Recorder stopped.')
        self.recorder = None
