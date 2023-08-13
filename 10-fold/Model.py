from torcheeg.models import TSCeption
from torcheeg.datasets.constants.emotion_recognition.dreamer import DREAMER_CHANNEL_LOCATION_DICT

def core_model(config):
    if "Name" in config.keys():
        return "TSCeption"
    return TSCeption(num_electrodes=len(DREAMER_CHANNEL_LOCATION_DICT),
                                    num_classes=2,
                                    num_T=15,
                                    num_S=15,
                                    in_channels=1,
                                    hid_channels=32,
                                    sampling_rate=128,
                                    dropout=0.5)