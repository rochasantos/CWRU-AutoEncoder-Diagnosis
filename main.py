# logging
import sys
import logging
from src.utils import LoggerWriter

from experiments import tune_svm_wp, tune_lr_wp, tune_rf_wp, tune_dt_wp, tune_castboot_wp, wp_experimenter

if __name__ == "__main__":

    experiment_key = "experiment"
    # sys.stdout = LoggerWriter(logging.info, experiment_key)
   
    # tune_svm_wp()
    # tune_lr_wp()
    # tune_rf_wp()
    tune_dt_wp()
    # tune_castboot_wp(task_type="CPU", n_trials=50)
    # wp_experimenter()
   