import argparse
import os
import pathlib
from typing import Any, Dict, Literal

import torch
from pytorch_lightning.lite import LightningLite
from torch import Tensor
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchvision.transforms as T
import numpy as np

from utils.classifiers.binary_models import BinaryTrainedVGG, BinaryVGG
from utils.datasets import BinaryDataset, SequenceDataset
from utils.attacks import PGDL0, PGDL2, PGDLinf
from utils.utils import freeze, set_seed
from utils.gdpr_cfe import GDPR_CFE
from utils.scfe import APG0_CFE
from utils.utils import ModelWithNormalization

from wilds import get_dataset
from uuid import uuid4

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
UUID = uuid4()

def calc_loss(outs: Tensor, labels: Tensor) -> Tensor:
    assert outs.shape == labels.shape
    assert len(labels.shape) == 1
    return (- outs * labels.cuda()).exp()
    
 
class BinaryPGDL0(PGDL0):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDL2(PGDL2):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

class BinaryPGDLinf(PGDLinf):
    def _calc_loss(self, outs: Tensor, targets: Tensor) -> Tensor:
        return calc_loss(outs, targets)
    

def fine_tune(classifier, dataloader, train=False, save_model=True) -> float: # type: ignore
    
    epochs = 200
    optim = Adam(classifier.parameters(), lr=3e-4) if not train else SGD(classifier.parameters(),lr=0.01, momentum=0.9, weight_decay=5e-4, nesterov=True)
    scheduler = ReduceLROnPlateau(optim)
    
    classifier.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        for _, (imgs, labels) in enumerate(dataloader):
            outs = classifier(imgs.cuda())
            losses = calc_loss(outs, labels.cuda())
            loss = losses.mean()

            optim.zero_grad(True)
            loss.backward()
            optim.step()
            running_loss += loss.item() * imgs.size(0)

        scheduler.step(running_loss/ len(dataloader.dataset))
        
        if epoch % 20 == 0:
            print(f'Running loss: {running_loss / len(dataloader.dataset):.2f}')
        if epoch == epochs - 1:
            if save_model:
                torch.save(classifier.state_dict(), f'./models/waterbirds_clean_{UUID}.pt')
            return loss.item()

    
        

@torch.no_grad()
def test(classifier, dataloader) -> Tensor:
    
    num_samples = 0
    num_correct = 0
    
    for _, (imgs, labels) in enumerate(dataloader):
        num_landbird = 0 # Number of correct for english spanieel in a batch
        num_waterbird = 0 # Number of correct for cassette player in a batch
        assert len(labels.shape) == 1
        num_samples += len(labels)

        output = classifier(imgs.cuda())
        num_correct += ((output * labels.cuda()) > 0).count_nonzero().item()
        num_landbird += ((output * labels.cuda()) > 0).logical_and(labels.cuda() == -1).count_nonzero().item()
        num_waterbird += ((output * labels.cuda()) > 0).logical_and(labels.cuda() == 1).count_nonzero().item()
        #print(f'num correct for batch {_} is {num_correct} and num_landbird is {num_landbird}, num_waterbird {num_waterbird}')
    return num_correct / num_samples

@torch.no_grad()
def get_attack_succ_rate(classifier, dataloader):
    num_succ_attacks = 0
    for _, (adv_img, adv_labels) in enumerate(dataloader):
        pred = classifier(adv_img.cuda())
        num_succ_attacks += ((pred * adv_labels.cuda()) > 0).count_nonzero().item()

    return num_succ_attacks / len(dataloader.dataset)

def to_cpu(d: Dict[str, Any]) -> Dict[str, Any]:
    for k, v in d.items():
        if isinstance(v, Tensor):
            d[k] = v.cpu()
        elif isinstance(v, torch.nn.Module):
            d[k] = v.cpu().state_dict()
    return d

def generate_adv_labels(n: int, device: torch.device) -> Tensor:
    return 2 * torch.randint(0, 2, (n,), device=device) - 1

class Main(LightningLite):
    def run(self,
        norm: Literal['L0', 'L2', 'Linf'],
        mode: Literal['det'],
        seed: int,
    ) -> None:

        print(f'UUID: {UUID}')

        root = '/home/htc/kchitranshi/SCRATCH/CFE_datasets'
        os.makedirs(root, exist_ok=True)

        fname = f'WaterBirds_{norm}_{mode}'
        path = os.path.join(root, fname)

        if os.path.exists(path):
            print(f'already exist: {path}')
            return
        else:
            pathlib.Path(path).touch()

        fine_tune_model = True
        set_seed(seed)

        # Defining transformations
        train_transform = T.Compose([
            T.Resize(256),
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor()
        ])

        adv_train_transform = T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
        ])

        val_transform = test_transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor()
        ])

        dataset = get_dataset(dataset="waterbirds", download=False,root_dir='../SCRATCH/')
        train_data = dataset.get_subset("train", transform=train_transform)
        val_data = dataset.get_subset("val", transform=val_transform)
        test_data = dataset.get_subset("test", transform=test_transform)

        if fine_tune_model:
            train_data = BinaryDataset(train_data, which_dataset='waterbirds')
            train_dataloader = torch.utils.data.DataLoader(train_data, 
                                                        batch_size=32, 
                                                        shuffle=True,
                                                        num_workers=3,
                                                        pin_memory=True
                        )

        val_data = BinaryDataset(val_data, which_dataset='waterbirds')
        test_data = BinaryDataset(test_data, which_dataset='waterbirds')
        
        val_dataloader = torch.utils.data.DataLoader(val_data, 
                                                        batch_size=128, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        test_dataloader = torch.utils.data.DataLoader(test_data,
                                                      batch_size=128,
                                                      shuffle=False,
                                                      num_workers=3,
                                                      pin_memory=True
                        )

        model = BinaryTrainedVGG(vgg_type='vgg16')
        if fine_tune_model:
            model.train()
        else:
            print(f"Loading Pre-trained Model.")
            state_dict = torch.load("./models/waterbirds_clean.pt", map_location='cpu')
            #state_dict = OrderedDict((k.replace('resnet.', '',1), v) for k, v in state_dict.items())
        classifier = ModelWithNormalization(model,
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]
                    )
        if not fine_tune_model:
            classifier.load_state_dict(state_dict)

        classifier = self.setup(classifier)
        loss = fine_tune(classifier, train_dataloader) if fine_tune_model else None
        freeze(classifier)
        classifier.eval()

        val_acc = test(classifier, val_dataloader)
        test_acc = test(classifier, test_dataloader)

        print('-'*60)
        print(f'Accuracy of trained model on clean validation data: {val_acc * 100:.2f}%')
        print(f'Accuracy of trained model on clean test data: {test_acc * 100:.2f}%')

        data_range = (0,1)
        steps = 100

        if norm == 'GDPR_CFE':
            atk = GDPR_CFE(
                    model=classifier,
                    max_image_range = 1.0,
                    min_image_range = 0.0, 
                    optimizer = torch.optim.Adam, 
                    iters=steps, 
                    lamb=0.11,
                    lamb_cf=0.0027,
                    mode="artificial",
                    device= 'cuda:0',
                )
        elif norm == 'SCFE':
            atk = APG0_CFE
        elif norm == 'L2':
            atk = BinaryPGDL2(classifier=classifier, 
                              steps=steps, 
                              eps=3, 
                              data_range=data_range
                )
        elif norm == 'Linf':
            atk = BinaryPGDLinf(classifier=classifier, 
                                steps=steps, 
                                eps=0.03, 
                                data_range=data_range
                )
        else:
            raise ValueError(norm)
        
        adv_dataset = {'imgs': [], 'labels': []}
        
        adv_attack_data = dataset.get_subset("train", transform=val_transform)
        adv_attack_data = BinaryDataset(adv_attack_data, which_dataset='waterbirds')

        adv_attack_loader = torch.utils.data.DataLoader(adv_attack_data, 
                                                        batch_size=64, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                            )
        avg_L2_norms = []
        for _, (data, _) in tqdm(enumerate(adv_attack_loader)):

            labels = generate_adv_labels(data.shape[0], "cuda:0")

            if norm in ['L2','Linf']:
                adv_data = atk(data, labels)
            elif norm == 'SCFE':
                data_flattened = data.view(data.size(0), -1)

                maxs = torch.tensor(data_range[1]).repeat(data_flattened.size(-1))
                mins = torch.tensor(data_range[0]).repeat(data_flattened.size(-1))

                del data_flattened

                cfe_atk = atk(model=classifier,
                           range_min=None,
                           range_max=None,
                           numclasses=2,
                           scale_model=False,
                           iters=steps,
                           maxs=maxs,
                           mins=mins,
                           lam_steps=5
                )
                adv_data = cfe_atk.get_CFs_artificial(data.cuda(), labels.unsqueeze(1).cuda())

                del maxs
                del mins
            else:
                adv_data = atk.get_perturbations(data, labels.unsqueeze(1))
                
            avg_L2_norms.append((adv_data.cpu() - data).norm(p=2, dim=(1,2,3)).mean().item())
            adv_dataset['imgs'].append(adv_data.cpu())
            adv_dataset['labels'].append(labels.cpu())
        
        adv_dataset['imgs'] = torch.cat(adv_dataset['imgs'])
        adv_dataset['labels'] = torch.cat(adv_dataset['labels'])

        adv_data = SequenceDataset(adv_dataset['imgs'], adv_dataset['labels'],x_transform=adv_train_transform)
        adv_asr_check_data = SequenceDataset(adv_dataset['imgs'], adv_dataset['labels'],x_transform=None)

        adv_dataloader = torch.utils.data.DataLoader(adv_data, 
                                                        batch_size=32, 
                                                        shuffle=True,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        adv_asr_check_dataloader = torch.utils.data.DataLoader(adv_asr_check_data, 
                                                        batch_size=128, 
                                                        shuffle=False,
                                                        num_workers=3,
                                                        pin_memory=True
                        )
        
        print('-'*60)
        attack_succ_rate = get_attack_succ_rate(classifier=classifier, dataloader=adv_asr_check_dataloader)
        print(f'The attack success rate is {attack_succ_rate * 100:.2f} with an avg L2 norm of {np.mean(avg_L2_norms).item():.2f}')
        print('-'*60)

        adv_model = BinaryVGG(vgg_type='vgg16')
        adv_model.train()
        adv_classifier = ModelWithNormalization(adv_model,
                                            mean=[0.485, 0.456, 0.406], 
                                            std=[0.229, 0.224, 0.225]
                    )
        adv_classifier = self.setup(adv_classifier)
        adv_loss = fine_tune(adv_classifier, adv_dataloader, train=True, save_model=False)
        freeze(adv_classifier)
        adv_classifier.eval()

        adv_acc_for_natural_val = test(adv_classifier, val_dataloader)
        adv_acc_for_natural_test = test(adv_classifier, test_dataloader)
        
        print(f'Accuracy of Adversarially trained model on clean validation data: {adv_acc_for_natural_val * 100:.2f}%')
        print(f'Accuracy of Adversarially trained model on clean test data: {adv_acc_for_natural_test * 100:.2f}%')
        
        print('='*60)
        print('='*60)

        save_data = {
            'classifier': classifier,
            'adv_data': adv_dataset['imgs'],
            'adv_labels': adv_dataset['labels'],
            'clean_fine_tune_loss': loss if fine_tune_model else None,
            'clean_val_acc': val_acc,
            'clean_test_acc': test_acc,
            'adv_data': adv_data,
            'adv_classifier': adv_classifier,
            'adv_train_loss': adv_loss,
            'adv_acc_for_natural_val': adv_acc_for_natural_val,
            'adv_acc_for_natural_test': adv_acc_for_natural_test,
            'avg_L2_norms': avg_L2_norms
        }
        to_cpu(save_data)
        torch.save(save_data, path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('norm', choices=('GDPR_CFE', 'SCFE', 'L2', 'Linf'))
    parser.add_argument('mode', choices=('det'))
    parser.add_argument('seed', type=int)
    parser.add_argument('devices', nargs='+', type=int)
    args = parser.parse_args()

    lite_kwargs = {
        'accelerator': 'gpu',
        'strategy': 'ddp_find_unused_parameters_false',
        'devices': args.devices,
        'precision': 16,
    }
    
    Main(**lite_kwargs).run(
        args.norm,
        args.mode,
        args.seed,
    )
