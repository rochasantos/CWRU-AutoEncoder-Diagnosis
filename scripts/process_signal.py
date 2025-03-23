from datasets import CWRU, UORED, Hust, Paderborn, SU


def process_signal(dataset_name, segment_size=None, target_sr=42000):

    filters = {
        "cwru": {"label": ["N", "I", "O", "B"], "sensor_position": ["no", "centered"], "sampling_rate": "48000", "bearing_type": "6205"},
        "uored": {"label": ["N", "I", "O", "B"], "condition_bearing_health": ["faulty", "healthy"]},
        "hust": {"label": ["N", "I", "O", "B"], "bearing_type": ["6204", "6205", "6206", "6207", "6208"]},
        "hust_4": {"label": ["N", "I", "O", "B"], "bearing_type": "6204"},
        "hust_5": {"label": ["N", "I", "O", "B"], "bearing_type": "6205"},
        "hust_6": {"label": ["N", "I", "O", "B"], "bearing_type": "6206"},
        "hust_7": {"label": ["N", "I", "O", "B"], "bearing_type": "6207"},
        "hust_8": {"label": ["N", "I", "O", "B"], "bearing_type": "6208"},
        "paderborn": ["KA01", "KA03", "KA05", "KA06", "KA07", "KA09", "KI01", "KI03", "KI07", "KI08"],
        "su": {"label": ["N", "I", "O", "B"], "condition_bearing_health": ["faulty"]},
    }
    if dataset_name.lower() == "cwru":
        dataset = CWRU()
        dataset.save_signal(f"data/processed/cwru", filters["cwru"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "paderborn":
        dataset = Paderborn()
        dataset.save_signal(f"data/processed/paderborn", filters["paderborn"], segment_size, target_sr, class_names=["I", "O"])
    elif dataset_name.lower() == "hust":
        dataset = Hust()
        dataset.save_signal(f"data/processed/hust", filters["hust"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "hust_4":
        dataset = Hust()
        dataset.save_signal(f"data/processed/hust", filters["hust_4"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "hust_5":
        dataset = Hust()
        dataset.save_signal(f"data/processed/hust", filters["hust_5"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "hust_6":
        dataset = Hust()
        dataset.save_signal(f"data/processed/hust", filters["hust_6"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "hust_7":
        dataset = Hust()
        dataset.save_signal(f"data/processed/hust", filters["hust_7"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "hust_8":
        dataset = Hust()
        dataset.save_signal(f"data/processed/hust", filters["hust_8"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "uored":
        dataset = UORED()
        dataset.save_signal(f"data/processed/uored", filters["uored"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    elif dataset_name.lower() == "su":
        dataset = SU()
        dataset.save_signal(f"data/processed/su", filters["su"], segment_size, target_sr, class_names=["N", "I", "O", "B"])
    else:
        print("Please provide a valid dataset.") 
    