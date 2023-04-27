from labc import register_service, subscribe
from pylsl import StreamInfo, StreamOutlet
import uuid

@register_service
class LslLogger:

    def __init__(self):
        self.uid = uuid.uuid1()
        self.info = StreamInfo(
            name='logging',
            type='Marker',
            channel_count=1,
            nominal_srate=0.0,
            channel_format='string',
            source_id=str(self.uid),
        )
        self.outlet = StreamOutlet(self.info)

    @subscribe(topic="")
    def rerout_log(self, record):
        print(record)
