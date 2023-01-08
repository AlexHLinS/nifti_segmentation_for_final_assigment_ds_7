# %%
import pandas as pd
import numpy as np
from os import cpu_count
import os.path
from typing import Any, Dict, Optional, Union, List, Tuple

import json
from pprint import pprint

# %%
import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT, EPOCH_OUTPUT
from torch.nn import Module, Conv2d, ConvTranspose2d, MaxPool2d, ReLU
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

import segmentation_models_pytorch as smp

import wandb

# %%
from nifti_tools import get_data_from_nifti_file

# %%
class NiftiDataset(Dataset):
    def normalize_data(self, x:torch.Tensor, y:torch.Tensor) -> Union[torch.Tensor, torch.Tensor]:
        x = x /2000
        y = y.int()
        return (x, y)

    def save_layers_to_files(self, target_dir: str, layers:np.ndarray, set_name: str) -> list:
        i = 0
        paths = []
        for layer in layers:
            filename = target_dir + set_name + '_' + str(i)
            torch.save(torch.Tensor(layer), filename)
            paths.append(filename)
            i += 1
        return paths

    def __init__(self, data_paths: pd.DataFrame, cache_dir: str, image_converter=None, mask_converter=None,
                 data_mutator=None):
        """
        Args:
            data_paths: Pandas DataFrame with the following columns 'image' and 'label', where exists paths to images and labels respectively
            cache_dir: directory to extract and store tensors from nifti dataset (DIR MAST BE EXIST!)
            image_converter: function applied to images when it loads from file
            mask_converter: function applied to images when it loads from file
            data_mutator: function applied to data when it loads from file (for example it will be normalizer)
        """
        self.cache_dir = cache_dir
        if os.path.isdir(self.cache_dir) is None:
            try:
                os.mkdir(self.cache_dir)
            except OSError:
                pprint('Cannot create cache directory!')
        self.image_paths = []
        self.mask_paths = []
        self.image_converter = image_converter
        self.mask_converter = mask_converter
        self.mutator = data_mutator

        for i in range(data_paths.shape[0]):
            image_data = get_data_from_nifti_file(data_paths['image'][i], transponded=True)
            mask_data = get_data_from_nifti_file(data_paths['label'][i], transponded=True)
            image_set_name = str(data_paths['image'][i]).split('/')[-1].split('.')[0] + '.image'
            mask_set_name = str(data_paths['label'][i]).split('/')[-1].split('.')[0] + '.label'
            self.image_paths += self.save_layers_to_files(cache_dir, image_data, image_set_name)
            self.mask_paths += self.save_layers_to_files(cache_dir, mask_data, mask_set_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index) -> Union[torch.Tensor, torch.Tensor]:
        image_filename = self.image_paths[index]
        mask_filename = self.mask_paths[index]
        image = torch.load(image_filename)
        mask = torch.load(mask_filename)
        return self.normalize_data(image, mask)

    '''def __del__(self):
        for file in self.image_paths:
            os.remove(file)
        for file in self.mask_paths:
            os.remove(file)'''

# %%


# %%
# %%

# %%
class BaseElement(Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.conv_1 = Conv2d(in_channels=input_size, out_channels=hidden_size, kernel_size=3, stride=1, padding=1)
        self.conv_2 = Conv2d(in_channels=hidden_size, out_channels=output_size, kernel_size=3, stride=1, padding=1)
        self.pool = MaxPool2d(kernel_size=2, stride=2)
        self.relu = ReLU()

    def forward(self, x):
        x = self.conv_1(x)
        x = self.relu(x)
        x = self.conv_2(x)
        x = self.relu(x)
        return self.pool(x), x

# %%
class UNet(Module):
    def __init__(self):
        super().__init__()
        self.encoder_1 = BaseElement(input_size=1, hidden_size=64, output_size=64)
        self.encoder_2 = BaseElement(input_size=64, hidden_size=128, output_size=128)
        self.encoder_3 = BaseElement(input_size=128, hidden_size=256, output_size=256)
        self.encoder_4 = BaseElement(input_size=256, hidden_size=512, output_size=512)
        self.encoder_5 = BaseElement(input_size=512, hidden_size=1024, output_size=512)

        self.decoder_1 = BaseElement(input_size=128, hidden_size=64, output_size=64)
        self.decoder_2 = BaseElement(input_size=256, hidden_size=128, output_size=64)
        self.decoder_3 = BaseElement(input_size=512, hidden_size=256, output_size=128)
        self.decoder_4 = BaseElement(input_size=1024, hidden_size=512, output_size=256)

        self.up_2_1 = ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2)
        self.up_3_2 = ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2)
        self.up_4_3 = ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2)
        self.up_5_4 = ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2)

        self.final = Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1)

    def forward(self, x):
        # x [batch_size, 1, 320, 320]
        #print(f'Enter point X shape:{x.shape} \n')
        x, encoded_1 = self.encoder_1(x)  # спускаем x, пробрасываем encoded_1
        # x [batch_size, 64, 160, 160] ,encoded_1 [batch_size, 64, 320, 320]
        #print(f'Down X:{x.shape}, after Encoder1:{encoded_1.shape}\n')
        x, encoded_2 = self.encoder_2(x)  # спускаем x, пробрасываем encoded_2
        # x [batch_size, 128, 80, 80] ,encoded_2 [batch_size, 128, 160, 160]
        #print(f'Down X:{x.shape}, after Encoder2:{encoded_2.shape}\n')
        x, encoded_3 = self.encoder_3(x)  # спускаем x, пробрасываем encoded_3
        # x [batch_size, 256, 40, 40] ,encoded_3 [batch_size, 256, 80, 80]
        #print(f'Down X:{x.shape}, after Encoder3:{encoded_3.shape}\n')
        x, encoded_4 = self.encoder_4(x)  # спускаем x, пробрасываем encoded_4
        # x [batch_size, 512, 20, 20] ,encoded_4 [batch_size, 512, 40, 40]
        #print(f'Down X:{x.shape}, after Encoder4:{encoded_4.shape}\n')
        _, x = self.encoder_5(x)  # получаем x "внизу"
        # x [batch_size, 512, 20, 20]
        #print(f'after Encoder5 X:{x.shape}\n')
        x = self.up_5_4(x)  # поднимаем x
        # x [batch_size, 512, 40, 40]
        # encoded_4 [batch_size, 512, 40, 40]
        #print(f'up X: {x.shape}, after Encoder4:{encoded_4.shape}\n')
        x = torch.cat([x, encoded_4])  # объединяем с проброшенным encoded_4
        # x [batch_size, 1024, 40, 40]
        #print(f'merged X: {x.shape}\n')
        _, x = self.decoder_4(x)  # получаем x на выходе 4 декодера
        # x [batch_size, 256, 40, 40]


        x = self.up_4_3(x)  # поднимаем x
        # x [batch_size, 256, 80, 80]
        #encoded_3 [batch_size, 256, 80, 80]
        x = torch.cat([x, encoded_3])  # объединяем с проброшенным encoded_3
        # x [batch_size, 512, 80, 80]
        _, x = self.decoder_3(x)  # получаем x на выходе 3 декодера
        # x [batch_size, 128, 80, 80]
        x = self.up_3_2(x)  # поднимаем x
        # x [batch_size, 128, 160, 160]
        # encoded_2 [batch_size, 128, 160, 160]
        x = torch.cat([x, encoded_2])  # объединяем с проброшенным encoded_2
        # x [batch_size, 256, 160, 160]
        _, x = self.decoder_2(x)  # получаем x на выходе 2 декодера
        # x [batch_size, 64, 160, 160]
        x = self.up_2_1(x)  # поднимаем x
        # x [batch_size, 64, 320, 320]
        # encoded_1 [batch_size, 64, 320, 320]
        x = torch.cat([x, encoded_1])  # объединяем с проброшенным encoded_1
        # x [batch_size, 128, 320, 320]
        _, x = self.decoder_1(x)  # получаем x на выходе 1 декодера
        # x [batch_size, 64, 320, 320]

        x = self.final(x)  # еще один слой на выходе
        # x [batch_size, 1, 320, 320]
        x = x.sigmoid()

        return x

# %%
class UNetModel1(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = UNet()
        self.loss = smp.losses.DiceLoss(smp.losses.BINARY_MODE,
                                        from_logits=True)

    def after_step(self, batch, stage) -> Dict:
        """
        After each step calculate and returns metrics
        Args:
            batch:
            stage:

        Returns:

        """
        result = dict()
        result['stage'] = stage
        x, y = batch
        y_hat = self(x)
        #print(f'X: {x.shape}, y: {y.shape}, y_hat: {y_hat.shape}')
        result['loss'] = self.loss(y_hat, y)
        y_prob = y_hat.sigmoid()
        y_pred = (y_prob > 0.5).float()
        result['true_positive'], result['false_positive'], \
            result['false_negative'], result['true_negative'] = smp.metrics.get_stats(y_pred.long(),
                                                                                      y.long(),
                                                                                      mode='binary')
        return result

    def after_epoch(self, outputs, stage):
        """
        Calculate loss and metrics, then save them to log
        Args:
            outputs:
            stage:

        Returns:

        """
        true_positive = torch.cat([x["true_positive"] for x in outputs])
        false_positive = torch.cat([x["false_positive"] for x in outputs])
        false_negative = torch.cat([x["false_negative"] for x in outputs])
        true_negative = torch.cat([x["true_negative"] for x in outputs])

        total_loss = 0

        for i in range(len(outputs)): total_loss += outputs[i].get(f'{stage}_loss',0)

        metrics_result = dict()

        metrics_result[f'{stage}_recall'] = smp.metrics.recall(true_positive, false_positive, false_negative, true_negative, reduction="micro")
        metrics_result[f'{stage}_precision'] = smp.metrics.precision(true_positive, false_positive, false_negative, true_negative,
                                          reduction="micro")
        metrics_result[f'{stage}_f1_score'] = smp.metrics.f1_score(true_positive, false_positive, false_negative, true_negative, reduction="micro")
        metrics_result[f'{stage}_accuracy'] = smp.metrics.accuracy(true_positive, false_positive, false_negative, true_negative, reduction="macro")

        self.log_dict(dictionary=metrics_result, prog_bar=True)


    def forward(self, x) -> Any:
        y = self.model(x)
        return y

    def training_step(self, batch, _) -> STEP_OUTPUT:
        return self.after_step(batch=batch, stage='training')

    def training_epoch_end(self, outputs: EPOCH_OUTPUT) -> None:
        return self.after_epoch(outputs=outputs, stage='training')

    def validation_step(self, batch, _) -> Optional[STEP_OUTPUT]:
        return self.after_step(batch=batch, stage='validation')

    def validation_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        return self.after_epoch(outputs=outputs, stage='validation')

    def test_step(self, batch, _) -> Optional[STEP_OUTPUT]:
        return self.after_step(batch=batch, stage='test')

    def test_epoch_end(self, outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]]) -> None:
        return self.after_epoch(outputs=outputs, stage='test')

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters(), lr=0.02)

# %%

def main():
    avalible_cpu_core_cout = cpu_count() - 1
    print(f'Avalible CPU cores: {avalible_cpu_core_cout+1}')
    DATASET_DIR = 'datasets/Task02_Heart/'
    CACHE_DIR = 'datasets/Task02_Heart/cache/'

    dataset_info = json.load(open(DATASET_DIR + 'dataset.json', 'r'))
    dataset_data = pd.dataset_data = pd.DataFrame(data=dataset_info['training'])
    dataset_data['image'] = dataset_data['image'].str.replace('./', DATASET_DIR, regex=False)
    dataset_data['label'] = dataset_data['label'].str.replace('./', DATASET_DIR, regex=False)
    pprint(dataset_data.head(2))
    nds = NiftiDataset(dataset_data, CACHE_DIR)

    train_ds_size = int(len(nds) * 0.7)
    test_ds_size = int(len(nds)) - train_ds_size
    train_ds, test_ds = data.random_split(nds, [train_ds_size, test_ds_size], generator=torch.Generator().manual_seed(1234))

    train_data = DataLoader(dataset=train_ds, shuffle=False, num_workers=avalible_cpu_core_cout, persistent_workers=True)
    test_data = DataLoader(dataset=test_ds, shuffle=False, num_workers=avalible_cpu_core_cout, persistent_workers=True)

    callback = ModelCheckpoint(monitor="validation_f1_score", mode='max', filename='best_valid_f1_score_model', save_top_k=1, save_weights_only=True)
    early_stop = EarlyStopping(monitor= "validation_f1_score", min_delta=0.00, patience = 20, verbose=True, mode="max")
    SWA = StochasticWeightAveraging(swa_epoch_start=0.8, swa_lrs=0.001, annealing_epochs=5, annealing_strategy='cos')

    with open('tokens/wandb.token', 'r') as tokenfile:
        wandb_token = tokenfile.read().replace('/n', '')
    try:
        wandb.finish()
    except Exception as e:
        pprint(str(e))
    wandb.login(key=wandb_token)
    wandb_logger = WandbLogger(project='HSE_DS7_Final', name='Notebook')
    trainer = Trainer(logger=wandb_logger,
                    enable_checkpointing=True,
                    callbacks=[callback,early_stop, SWA],
                    max_epochs=1,
                    accelerator='gpu',
                    devices=[1], 
                    enable_progress_bar=True,
                    amp_backend="apex",
                    deterministic=True
                    )

    trainer.fit(model=UNetModel1(), train_dataloaders=train_data, val_dataloaders=test_data)

    # %%

    

if __name__ == "__main__":
    main()
# %%
