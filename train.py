import argparse
import os
import copy
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from model import *
from dataclass import PatchedDatasetTensor
from utils import AverageMeter, calc_psnr
import wandb
# wandb.login()

def log_wandb(loss_train, loss_valid, psnr_train, psnr_valid, learning_rate):
    wandb.log({"Training Loss" : loss_train, 
            "Validation Loss" : loss_valid, 
            "Train PSNR" : psnr_train, 
            "Valid PSNR" : psnr_valid, 
            "Learning rate" : learning_rate})

def init_wandb(project_name, config_wandb):
    run = wandb.init(project=project_name, entity="asrimanth")
    run.save()
    wandb.config = config_wandb
    return run.name

def sr_train(model, train_dataloader, valid_dataloader, device, args, is_wandb=True):
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_psnr = 0.0
    
    logs_dict = {"Train Loss": [], "Valid Loss": [], "Train PSNR": [], "Valid PSNR": []}
    
    if is_wandb:
        project_name = "DSDMSR"
        config_wandb = dict(
            epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            dataset="ETH3D",
            architecture="DSDMSR"
        )
        run_name = init_wandb(project_name, config_wandb)
        wandb.watch(model, log="all", log_freq=10)
    else:
        run_name = "Sample-dry-run"
    
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()
        loss_train = 0
        psnr_train = 0
        
        # Train Loop
        with tqdm(total=(len(train_dataloader) * args.batch_size)) as t:
            t.set_description(f'Train epoch: {epoch}/{args.num_epochs - 1}')

            for data in train_dataloader:
                optimizer.zero_grad()
                x1_image, x2_image, x4_image, x8_image, x16_image = data

                x1_image = x1_image.to(device)
                x2_image = x2_image.to(device)
                x4_image = x4_image.to(device)
                x8_image = x8_image.to(device)
                x16_image = x16_image.to(device)
                
                out_x2, out_x4, out_x8, out_x16, out_msf = model(x1_image)
                loss_x2 = loss_function(out_x2, x2_image)
                loss_x4 = loss_function(out_x4, x4_image)
                loss_x8 = loss_function(out_x8, x8_image)
                loss_x16 = loss_function(out_x16, x16_image)
                loss_msf = loss_function(out_msf, x16_image)

                loss = loss_x2 + loss_x4 + loss_x8 + loss_x16 + loss_msf

                epoch_losses.update(loss.item(), len(x1_image))

                loss.backward()
                optimizer.step()
                
                loss_train += loss.item()
                psnr_train += calc_psnr(out_msf.cpu(), x16_image.cpu()).item()

                t.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                t.update(len(x1_image))
            
            loss_train /= len(train_dataloader)
            psnr_train /= len(train_dataloader)
            

        model.eval()
        epoch_psnr = AverageMeter()
        
        # Validation Loop
        loss_valid = 0
        psnr_valid = 0
        for data in tqdm(valid_dataloader):
            x1_image, x2_image, x4_image, x8_image, x16_image = data

            x1_image = x1_image.to(device)
            x2_image = x2_image.to(device)
            x4_image = x4_image.to(device)
            x8_image = x8_image.to(device)
            x16_image = x16_image.to(device)
            
            with torch.no_grad():
                out_x2, out_x4, out_x8, out_x16, out_msf = model(x1_image).clamp(0.0, 1.0)
                loss_x2 = loss_function(out_x2, x2_image)
                loss_x4 = loss_function(out_x4, x4_image)
                loss_x8 = loss_function(out_x8, x8_image)
                loss_x16 = loss_function(out_x16, x16_image)
                loss_msf = loss_function(out_msf, x16_image)
                loss = loss_x2 + loss_x4 + loss_x8 + loss_x16 + loss_msf
                loss_valid += loss.item()
            psnr_valid += calc_psnr(out_msf.cpu(), x16_image.cpu()).item()
            epoch_psnr.update(calc_psnr(out_msf, x16_image), len(x16_image))

        print('Eval psnr: {:.2f}'.format(epoch_psnr.avg))
        
        loss_valid /= len(valid_dataloader)
        psnr_valid /= len(valid_dataloader)
        
        print("-"*10, "STATUS AT EPOCH NO.", epoch, "-"*10)
        print(f"Train PSNR : {psnr_train}, Train loss {loss_train}")
        print(f"Valid PSNR : {psnr_valid}, Valid loss {loss_valid}")
        
        logs_dict["Train Loss"].append(loss_train)
        logs_dict["Valid Loss"].append(loss_valid)
        logs_dict["Train PSNR"].append(psnr_train)
        logs_dict["Valid PSNR"].append(psnr_valid)
        logs = pd.DataFrame(logs_dict)
        logs.to_csv(os.path.join(args.outputs_dir, f"model_report_{run_name}.csv"))
        
        if is_wandb:
            log_wandb(loss_train, loss_valid, psnr_train, psnr_valid)

        if epoch_psnr.avg > best_psnr:
            best_epoch = epoch
            best_psnr = epoch_psnr.avg
            best_weights = copy.deepcopy(model.state_dict())

        print('Best epoch: {}, psnr: {:.2f}'.format(best_epoch, best_psnr))
        torch.save(best_weights, os.path.join(args.outputs_dir, f'best_model_{run_name}.pth'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--scale', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-epochs', type=int, default=40)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=123)
    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = DSDMSR().to(device)
    
    train_dataset = PatchedDatasetTensor(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    valid_dataset = PatchedDatasetTensor(args.eval_file)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)

    sr_train(model, train_dataloader, valid_dataloader, device, args, is_wandb=True)