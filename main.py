import sys
import logging
from src.utils import LoggerWriter
import os
from datasets import CWRU, UORED
from src.models import ModelFactory
from src.models import CNN7, CNN8, CNN9, CNN10, CNN11, CNN12, CNN13, CNN14
from config.experiment_configs import ExperimentKFoldConfig, ExperimentPretrainConfig
from experiments.kfold_runner import kfold_runner


def create_directory_structure():
    root_dir = "data/spectrograms"
    for severity in ["007", "014", "021"]:
        for label in ["N", "I", "O", "B"]:
            dir_path = os.path.join(root_dir, severity, label)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    if not os.path.exists("saved_models"):
        os.makedirs("saved_models")

def main(configs):
    for idx, config in enumerate(configs, 1):
        print(f"\n{'-'*50}")
        # show config.title of experiment
        print(f"{config.title}")
        print(f"\n{'-'*50}")
        print(f"\n[Experiment {idx}] Starting with parameters:")
        for field, value in config.__dict__.items():
            print(f"  {field}: {value}")
        kfold_runner(
            config.root_dir,
            config.learning_rate,
            config.num_epochs,
            config.batch_size,
            config.num_repetitions,
            config.model_factory,
            config.experiment_key,
            config.title
        )
        print(f"[Experiment {idx}] Finished.\n" + "-"*50)


if __name__ == "__main__":
    # create_directory_structure()
    # CWRU().download()

    # Redirect console output to logger
    experiment_key = f"experiments2"
    root_dir = "data/processed/cwru/bi"    
    sys.stdout = LoggerWriter(logging.info, experiment_key)

    # Parameters
    num_repetitions = 10
    num_epochs = 50
   
    experiment_configs = [
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN7, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_lr-001",
                         title="Simple 1D CNN with 3 Conv + Pool layers and 2 FC layers"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN8, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with BatchNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN9, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with LayerNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN10, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN11, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN12, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, LayerNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_ln_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm between layers and LayerNorm at the beginnig, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN7, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_lr-0001",
                         title="Simple 1D CNN with 3 Conv + Pool layers and 2 FC layers"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN8, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with BatchNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN9, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with LayerNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN10, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN11, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN12, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, LayerNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_ln_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm between layers and LayerNorm at the beginnig, final Sigmoid"),
    
    ]

    experiment_configs_pretrain = [
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_lr-001",
                         title="Simple 1D CNN with 3 Conv + Pool layers and 2 FC layers"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with BatchNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with LayerNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, LayerNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_ln_sig_lr-001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm between layers and LayerNorm at the beginnig, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_lr-0001",
                         title="Simple 1D CNN with 3 Conv + Pool layers and 2 FC layers"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with BatchNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, with LayerNorm"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, LayerNorm, final Sigmoid"),
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN14, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_bn_ln_sig_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm between layers and LayerNorm at the beginnig, final Sigmoid"),
    
    ]

    main([
        ExperimentKFoldConfig(root_dir=root_dir, learning_rate=0.0001, 
                         num_epochs=num_epochs, batch_size=32,
                         num_repetitions=num_repetitions, 
                         model_factory=ModelFactory(model_class=CNN13, input_length=9600 , num_classes=2), 
                         experiment_key="cnn_ln_bn_lr-0001",
                         title="Simple 1D CNN: 3 Conv + Pool layers, 2 FC layers, BatchNorm, final Sigmoid"),
    ])