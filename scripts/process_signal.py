from datasets import CWRU, UORED, Hust, Paderborn


def process_signal(dataset_name, segment_size=4200, target_sr=42000):

    filters = {
        "cwru": {"label": ["I", "O", "B"], "sensor_position": ["no", "centered"], "sampling_rate": "48000", "bearing_type": "6205"},
        "uored": {"label": ["I", "O", "B"], "condition_bearing_health": ["faulty"]},
        "hust": {"label": ["I", "O", "B"], "bearing_type": "6205"},
        "paderborn": ["KA01", "KA03", "KA05", "KA06", "KA07", "KA09", "KI01", "KI03", "KI07", "KI08"]
    }
    if dataset_name.lower() == "cwru":
        dataset = CWRU()
        dataset.save_signal(f"data/processed/cwru", filters["cwru"], segment_size, target_sr, class_names=["I", "O", "B"])
    elif dataset_name.lower() == "paderborn":
        dataset = Paderborn()
        dataset.save_signal(f"data/processed/paderborn", filters["paderborn"], segment_size, target_sr, class_names=["I", "O"])
    elif dataset_name.lower() == "hust":
        dataset = Hust()
        dataset.save_signal(f"data/processed/hust", filters["hust"], segment_size, target_sr, class_names=["I", "O", "B"])
    elif dataset_name.lower() == "uored":
        dataset = UORED()
        dataset.save_signal(f"data/processed/uored", filters["uored"], segment_size, target_sr, class_names=["I", "O", "B"])
    else:
        print("Please provide a valid dataset.") 
    