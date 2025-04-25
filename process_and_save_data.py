from datasets import CWRU, UORED, Hust, Paderborn
from src.data_processing.preprocessing import (
    PreprocessingPipeline, ResampleTransform, NormalizeTransform, 
    OutlierRemovalTransform, DetrendTransform, ZeroMeanTransform
)

if __name__ == "__main__":

    filters = {
        "cwru": {"label": ["N", "I", "O", "B"], "sensor_position": ["no", "centered"], "sampling_rate": "48000", "bearing_type": "6205"},
        "cwru48_7": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "007"},
        "cwru48_14": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "48000", "bearing_type": "6205", "extent_damage": "014"},
        "cwru48_21": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "021"},
        "cwru12_7": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "007"},
        "cwru12_14": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "014"},
        "cwru12_21": {"label": ["N", "I", "O", "B"], 
                 "sensor_position": ["no", "centered"], 
                 "sampling_rate": "12000", "bearing_type": "6205", "extent_damage": "021"},
        "cwru_ref": {"label": "N", "bearing_type": "6205"},
        "uored": {"label": ["N", "I", "O", "B"], "condition_bearing_health": ["faulty", "healthy"]},
        "hust": {"label": ["N", "I", "O", "B"], "bearing_type": ["6204", "6205", "6206", "6207", "6208"]},
        "hust_4": {"label": ["N", "I", "O", "B"], "bearing_type": "6204"},
        "hust_5": {"label": ["N", "I", "O", "B"], "bearing_type": "6205"},
        "hust_6": {"label": ["N", "I", "O", "B"], "bearing_type": "6206"},
        "hust_7": {"label": ["N", "I", "O", "B"], "bearing_type": "6207"},
        "hust_8": {"label": ["N", "I", "O", "B"], "bearing_type": "6208"},
        "paderborn": [
            "K001", "K002", "K003", "K004", "K005", "K006",
            "KA01", "KA03", "KA05", "KA06", "KA07", "KA09", 
            "KI01", "KI03", "KI07", "KI08"
        ],
        "su": {"label": ["N", "I", "O", "B"], "condition_bearing_health": ["faulty"]},
    }

    # Define the parameters
    dataset = Paderborn()
    segment_size = 4096
    max_size = None
    original_sr = 64000
    target_sr = 12000

    # Define the preprocessing pipeline
    pipeline = PreprocessingPipeline([
        DetrendTransform(),
        OutlierRemovalTransform(threshold=3.0),
        ZeroMeanTransform(),
        NormalizeTransform(),
        ResampleTransform(target_fs=target_sr)
    ])

    dataset.process_and_save_signal(
        output_root=None,   #"data/processed/cwru_ref",
        filter=filters["paderborn"],
        segment_size= int(segment_size * original_sr / target_sr),
        max_size= max_size,
        pipeline_transforms=pipeline
    )