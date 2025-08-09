import numpy as np
import matplotlib.pyplot as plt
import pywt
from scipy.stats import kurtosis
from ewtpy import EWT1D
import os
import numpy as np
from datasets import CWRU


def remove_outliers(data, threshold=4):
    mean = np.mean(data)
    std = np.std(data)
    z_scores = [(x - mean) / std for x in data]
    filtered_data = [data[i] for i, z in enumerate(z_scores) if abs(z) < threshold]
    return np.array(filtered_data)

def apply_ewt_and_cwt(signal, save_path, filename, cmap='inferno'):
    # Step 1: EWT decomposition
    components, mfb, boundaries = EWT1D(signal)

    # Step 2: Select component with highest kurtosis (can also use energy)
    kurt_values = [kurtosis(comp) for comp in components]
    best_idx = np.argmax(kurt_values)
    selected_component = components[best_idx]

    # Step 3: Apply Continuous Wavelet Transform (CWT)
    scales = np.arange(1, 129)  # adjust depending on your freq resolution
    coef, freqs = pywt.cwt(selected_component, scales, 'morl')

    # Step 4: Save the CWT as image
    plt.figure(figsize=(3, 3))
    plt.imshow(np.abs(coef), extent=[0, 1, 1, 128], cmap=cmap, aspect='auto', origin='lower')
    plt.axis('off')
    plt.tight_layout()
    
    full_path = os.path.join(save_path, f"{filename}.png")
    plt.savefig(full_path, bbox_inches='tight', pad_inches=0)
    plt.close()



ds = CWRU()

root_dir = 'data/processed/io/cwru'

metainfo = ds.load_data_f({"label": ["I", "O"], "sampling_rate": "48000", "bearing_type": "6205"})

# motor_load_train = ['1', '2', '3']
# motor_load_val = ['0']
# fold_map = {
#     'fold1': {'train': {'extent_damage': ['014', '021'], 'hp': motor_load_train},
#               'val': {'extent_damage': ['014', '021'], 'hp': motor_load_val},
#               'test': {'extent_damage': ['007'], 'hp': motor_load_train}},
#     'fold2': {'train': {'extent_damage': ['007', '021'], 'hp': motor_load_train},
#               'val': {'extent_damage': ['007', '021'], 'hp': motor_load_val},
#               'test': {'extent_damage': ['014'], 'hp': motor_load_train}},
#     'fold3': {'train': {'extent_damage': ['007', '014'], 'hp': motor_load_train},
#               'val': {'extent_damage': ['007', '014'], 'hp': motor_load_val},
#               'test': {'extent_damage': ['021'], 'hp': motor_load_train}}   
# }

for i, item in enumerate(metainfo):
    signal = item[0].flatten()
    label = item[1]['label']
    basename = item[1]['filename']
    severity = item[1]['extent_damage']
    motor_load = item[1]['hp']
    print(f"Signal {i}: {basename}, Label: {label}, Length: {len(signal)}")

    segment_length = 4096
    num_segments = len(signal) // segment_length
    output_dir = f'{root_dir}/{severity}/{label}'
    if motor_load == '0':
        output_dir = f'{root_dir}_val/{severity}/{label}'
    os.makedirs(output_dir, exist_ok=True)

    for seg_idx in range(num_segments):
        start = seg_idx * segment_length
        end = start + segment_length
        segment = signal[start:end]      

        filename = f'{basename}_{i}_seg{seg_idx}_{label}'
        # filepath = os.path.join(output_dir, filename)
    
        # apply_ewt_and_cwt(segment, save_path=output_dir, filename=filename, cmap='inferno')
        np.save(os.path.join(output_dir, f"{filename}.npy"), segment)

