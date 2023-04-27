
# sound sample source: wiktionary

from pathlib import Path
import asyncio

from labc import register_service, subscribe
from playsound import playsound


audio_path = Path(__file__).parent / 'audio_files'


files = {
    "de" : {
        "right" : "De-ja.ogg",
        "left" : "De-nein.ogg",
        "login": "login.ogg",
        "logout": "logout.ogg",
    },
    "en" : {
        "right" : "En-uk-yes.ogg",
        "left" : "En-uk-no.ogg",
        "login": "login.ogg",
        "logout": "logout.ogg",
    },
    "default": "quack.wav",
}


@register_service
class SoundService:
    def __init__(self, lang="en"):
        self.lang = lang

    @subscribe
    async def play(self, name=None, lang=None):
        lang = lang or self.lang
        do_default = False
        try:
            path = audio_path / files[lang][name]
            if not path.exists():
                do_default = True
        except Exception as e:
            print("Cannot select soundfile:", name, 'for', lang)
            do_default = True
        if do_default:
            path = audio_path / files['default']
        asyncio.get_event_loop().run_in_executor(None, playsound, str(path))

