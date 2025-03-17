from datasets import CWRU, UORED, Hust, Paderborn, SU

def download(dataset_name):
    if dataset_name.lower() == "cwru":
        CWRU().download()
    elif dataset_name.lower() == "paderborn":
        Paderborn().download()
    elif dataset_name.lower() == "hust":
        Hust().download()
    elif dataset_name.lower() == "uored":
        UORED().download()
    elif dataset_name.lower() == "su":
        SU().download()
    else:
        print("Please provide a valid dataset.")     