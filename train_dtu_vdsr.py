import argparse
import os
import copy
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data.dataloader import DataLoader


import torchvision.transforms.functional as tvf
from torchvision.utils import make_grid

from tqdm import tqdm

from model import DSDMSR_VDSR_x8
from dataclass import Patched_DTU_Tensor
from utils import AverageMeter, calc_psnr
import wandb
# wandb.login()


def init_wandb(project_name, config_wandb):
    run = wandb.init(project=project_name, entity="asrimanth")
    run.save()
    wandb.config = config_wandb
    return run.name

def sr_train(model, train_dataloader, valid_dataloader, device, args):
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = optim.SGD(params=model.parameters(),
                            lr=args.lr,
                            momentum=0.9,
                            weight_decay=0.0001)
    optim_scheduler = optim.lr_scheduler.StepLR(optimizer,
                            step_size=20,
                            gamma=0.1)

    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_loss = 9999

    # To plot n ramdom images from the validation set
    n_valid_plot = 6
    rand_indices_for_plotting = [np.random.randint(0, len(valid_dataloader)) for i in range(n_valid_plot)]

    logs_dict = {
        "Train Loss X2": [], "Train Loss X4": [], "Train Loss X8": [], "Train Loss MSF": [], "Train Loss Average": [],
        "Valid Loss X2": [], "Valid Loss X4": [], "Valid Loss X8": [], "Valid Loss MSF": [], "Valid Loss Average": [],
        "Train PSNR X2": [], "Train PSNR X4": [], "Train PSNR X8": [], "Train PSNR MSF": [], "Train PSNR Average": [],
        "Valid PSNR X2": [], "Valid PSNR X4": [], "Valid PSNR X8": [], "Valid PSNR MSF": [], "Valid PSNR Average": [],
    }

    project_name = "DSDMSR-VDSR"
    config_wandb = dict(
        epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        dataset="DTU",
        architecture="DSDMSR"
    )
    run_name = init_wandb(project_name, config_wandb)
    wandb.watch(model, log="all", log_freq=10)

    output_dir = f"{args.outputs_dir}/{run_name}/"
    if not os.path.exists(output_dir):
        print(f"Save path: {output_dir}")
        os.makedirs(output_dir)


    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = AverageMeter()

        loss_train = 0
        loss_train_by_2 = 0
        loss_train_by_4 = 0
        loss_train_by_8 = 0
        loss_train_msf = 0

        psnr_train = 0
        psnr_train_by_2 = 0
        psnr_train_by_4 = 0
        psnr_train_by_8 = 0
        psnr_train_msf = 0

        # Train Loop
        current_learning_rate = optim_scheduler.get_last_lr()[-1]
        with tqdm(total=(len(train_dataloader) * args.batch_size)) as progress_bar:
            progress_bar.set_description(f'Train epoch: {epoch}/{args.num_epochs - 1}')

            for data in train_dataloader:
                optimizer.zero_grad()
                x64_image, x128_image, x256_image, x512_image, _ = data
                batch_size = len(x64_image)

                x64_image = x64_image.to(device)
                x128_image = x128_image.to(device)
                x256_image = x256_image.to(device)
                x512_image = x512_image.to(device)

                out_x128, out_x256, out_x512, out_msf = model(x64_image)

                loss_x128 = loss_function(out_x128, x128_image)
                loss_x256 = loss_function(out_x256, x256_image)
                loss_x512 = loss_function(out_x512, x512_image)
                loss_msf = loss_function(out_msf, x512_image)

                loss = loss_x128 + loss_x256 + loss_x512 + loss_msf

                epoch_losses.update(loss.item(), batch_size)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               max_norm=1.0/current_learning_rate,
                                               norm_type=1.0)
                optimizer.step()

                loss_train_by_2 += loss_x128.item()
                loss_train_by_4 += loss_x256.item()
                loss_train_by_8 += loss_x512.item()
                loss_train_msf += loss_msf.item()

                psnr_train_by_2 += calc_psnr(out_x128.cpu(), x128_image.cpu()).item()
                psnr_train_by_4 += calc_psnr(out_x256.cpu(), x256_image.cpu()).item()
                psnr_train_by_8 += calc_psnr(out_x512.cpu(), x512_image.cpu()).item()
                psnr_train_msf += calc_psnr(out_msf.cpu(), x512_image.cpu()).item()

                progress_bar.set_postfix(loss='{:.6f}'.format(epoch_losses.avg))
                progress_bar.update(batch_size)

            loss_train_by_2 /= len(train_dataloader)
            loss_train_by_4 /= len(train_dataloader)
            loss_train_by_8 /= len(train_dataloader)
            loss_train_msf /= len(train_dataloader)
            loss_train = (loss_train_by_2 + loss_train_by_4 + loss_train_by_8 + loss_train_msf) / 4

            psnr_train_by_2 /= len(train_dataloader)
            psnr_train_by_4 /= len(train_dataloader)
            psnr_train_by_8 /= len(train_dataloader)
            psnr_train_msf /= len(train_dataloader)
            psnr_train = (psnr_train_by_2 + psnr_train_by_4 + psnr_train_by_8 + psnr_train_msf) / 4


        model.eval()
        epoch_psnr = AverageMeter()

        # Validation Loop
        loss_valid = 0
        psnr_valid = 0

        loss_valid_by_2 = 0
        loss_valid_by_4 = 0
        loss_valid_by_8 = 0
        loss_valid_msf = 0

        psnr_valid_by_2 = 0
        psnr_valid_by_4 = 0
        psnr_valid_by_8 = 0
        psnr_valid_msf = 0

        example = torch.Tensor([])
        examples = []
        valid_idx = 0

        for data in tqdm(valid_dataloader):
            x64_image, x128_image, x256_image, x512_image, _ = data
            batch_size = len(x64_image)
            x64_image = x64_image.to(device)
            x128_image = x128_image.to(device)
            x256_image = x256_image.to(device)
            x512_image = x512_image.to(device)

            with torch.no_grad():
                out_x128, out_x256, out_x512, out_msf = model(x64_image) #.clamp(0.0, 1.0)
                loss_x128 = loss_function(out_x128, x128_image)
                loss_x256 = loss_function(out_x256, x256_image)
                loss_x512 = loss_function(out_x512, x512_image)
                loss_msf = loss_function(out_msf, x512_image)
                loss = loss_x128 + loss_x256 + loss_x512 + loss_msf
                loss_valid += loss.item()

            loss_valid_by_2 += loss_x128.item()
            loss_valid_by_4 += loss_x256.item()
            loss_valid_by_8 += loss_x512.item()
            loss_valid_msf += loss_msf.item()

            psnr_valid_by_2 += calc_psnr(out_x128.cpu(), x128_image.cpu()).item()
            psnr_valid_by_4 += calc_psnr(out_x256.cpu(), x256_image.cpu()).item()
            psnr_valid_by_8 += calc_psnr(out_x512.cpu(), x512_image.cpu()).item()
            psnr_valid_msf += calc_psnr(out_msf.cpu(), x512_image.cpu()).item()

            # Plot selected images
            if valid_idx in rand_indices_for_plotting:
                x1_input = x64_image.squeeze(0)
                x2_gt = x128_image.squeeze(0)
                x4_gt = x256_image.squeeze(0)
                x8_gt = x512_image.squeeze(0)
                x1_input = F.pad(input=x1_input, pad=(280, 280, 224, 224), mode='constant', value=1) # For 64x80
                x2_gt = F.pad(input=x2_gt, pad=(240, 240, 192, 192), mode='constant', value=1) # For 128x160
                x4_gt = F.pad(input=x4_gt, pad=(160, 160, 128, 128), mode='constant', value=1) # For 256x320

                x2_pred = out_x128.squeeze(0)
                x4_pred = out_x256.squeeze(0)
                x2_pred = F.pad(input=x2_pred, pad=(240, 240, 192, 192), mode='constant', value=1) # For 128x160
                x4_pred = F.pad(input=x4_pred, pad=(160, 160, 128, 128), mode='constant', value=1) # For 256x320

                x8_pred = out_x512.squeeze(0)
                msf_pred = out_msf.squeeze(0)

                example = make_grid(
                    [x2_pred, x2_gt, x4_pred, x4_gt, x8_pred, msf_pred, x8_gt, x1_input],
                    nrow=2
                )
                caption = f""
                caption += f"Top Left: x128 Pred, Top Right: x128 GT, "
                caption += f"2nd row Left: x256 Pred, 2nd row Right: x256 GT, "
                caption += f"3rd row Left: x512 Pred, 3rd row Right: x512 MSF, "
                caption += f"Last row Left: x512 GT, Last row Right: x64 Input "
                caption += f"--- At index : {valid_idx}."
                example = wandb.Image(example, caption=caption)
                examples.append(example)

            epoch_psnr.update(calc_psnr(out_msf, x512_image), batch_size)
            valid_idx += 1

        print('Eval psnr: {:.2f}'.format(epoch_psnr.avg))

        loss_valid_by_2 /= len(valid_dataloader)
        loss_valid_by_4 /= len(valid_dataloader)
        loss_valid_by_8 /= len(valid_dataloader)
        loss_valid_msf /= len(valid_dataloader)

        psnr_valid_by_2 /= len(valid_dataloader)
        psnr_valid_by_4 /= len(valid_dataloader)
        psnr_valid_by_8 /= len(valid_dataloader)
        psnr_valid_msf /= len(valid_dataloader)

        loss_valid = (loss_valid_by_2 + loss_valid_by_4 + loss_valid_by_8 + loss_valid_msf) / 4
        psnr_valid = (psnr_valid_by_2 + psnr_valid_by_4 + psnr_valid_by_8 + psnr_valid_msf) / 4

        print("-"*10, "STATUS AT EPOCH NO.", epoch, "-"*10)
        print(f"Train PSNR X2: {psnr_train_by_2}, Train PSNR X4: {psnr_train_by_4},\nTrain PSNR X8: {psnr_train_by_8}, Train PSNR MSF: {psnr_train_msf},\nTrain PSNR Avge. : {psnr_train}")
        print(f"Valid PSNR X2: {psnr_valid_by_2}, Valid PSNR X4: {psnr_valid_by_4},\nValid PSNR X8: {psnr_valid_by_8}, Valid PSNR MSF: {psnr_valid_msf},\nValid PSNR Avge. : {psnr_valid}")
        print(f"Train Loss X2: {loss_train_by_2}, Train Loss X4: {loss_train_by_4},\nTrain Loss X8: {loss_train_by_8}, Train Loss MSF: {loss_train_msf},\nTrain Loss Avge.: {loss_train}")
        print(f"Valid Loss X2: {loss_valid_by_2}, Valid Loss X4: {loss_valid_by_4},\nValid Loss X8: {loss_valid_by_8}, Valid Loss MSF: {loss_valid_msf},\nValid Loss Avge.: {loss_valid}")
        print("-"*25)

        logs_dict["Train Loss X2"].append(loss_train_by_2)
        logs_dict["Train Loss X4"].append(loss_train_by_4)
        logs_dict["Train Loss X8"].append(loss_train_by_8)
        logs_dict["Train Loss MSF"].append(loss_train_msf)
        logs_dict["Train Loss Average"].append(loss_train)

        logs_dict["Valid Loss X2"].append(loss_valid_by_2)
        logs_dict["Valid Loss X4"].append(loss_valid_by_4)
        logs_dict["Valid Loss X8"].append(loss_valid_by_8)
        logs_dict["Valid Loss MSF"].append(loss_valid_msf)
        logs_dict["Valid Loss Average"].append(loss_valid)

        logs_dict["Train PSNR X2"].append(psnr_train_by_2)
        logs_dict["Train PSNR X4"].append(psnr_train_by_4)
        logs_dict["Train PSNR X8"].append(psnr_train_by_8)
        logs_dict["Train PSNR MSF"].append(psnr_train_msf)
        logs_dict["Train PSNR Average"].append(psnr_train)

        logs_dict["Valid PSNR X2"].append(psnr_valid_by_2)
        logs_dict["Valid PSNR X4"].append(psnr_valid_by_4)
        logs_dict["Valid PSNR X8"].append(psnr_valid_by_8)
        logs_dict["Valid PSNR MSF"].append(psnr_valid_msf)
        logs_dict["Valid PSNR Average"].append(psnr_valid)

        logs = pd.DataFrame(logs_dict)
        logs.to_csv(os.path.join(output_dir, f"model_report_{run_name}.csv"))


        # We know that the batch size is 1.
        # Show the last image in the unshuffled batch.
        example_0, example_1, example_2, example_3, example_4, example_5 = examples
        wandb.log({
            "Train Loss X2" : loss_train_by_2,
            "Valid Loss X2" : loss_valid_by_2,
            "Train Loss X4" : loss_train_by_4,
            "Valid Loss X4" : loss_valid_by_4,
            "Train Loss X8" : loss_train_by_8,
            "Valid Loss X8" : loss_valid_by_8,
            "Train Loss MSF" : loss_train_msf,
            "Valid Loss MSF" : loss_valid_msf,
            "Train Loss Average" : loss_train,
            "Valid Loss Average" : loss_valid,

            "Train PSNR X2" : psnr_train_by_2,
            "Valid PSNR X2" : psnr_valid_by_2,
            "Train PSNR X4" : psnr_train_by_4,
            "Valid PSNR X4" : psnr_valid_by_4,
            "Train PSNR X8" : psnr_train_by_8,
            "Valid PSNR X8" : psnr_valid_by_8,
            "Train PSNR MSF" : psnr_train_msf,
            "Valid PSNR MSF" : psnr_valid_msf,

            "Train PSNR Average" : psnr_train,
            "Valid PSNR Average" : psnr_valid,
            "Learning Rate": current_learning_rate,
            f"Example 0": example_0,
            f"Example 1": example_1,
            f"Example 2": example_2,
            f"Example 3": example_3,
            f"Example 4": example_4,
            f"Example 5": example_5,
        })

        if loss_valid < best_loss:
            best_loss = loss_valid
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            torch.save(best_weights,
                   os.path.join(output_dir, f"best_model_{run_name}.pth"))
        print(f"Best epoch: {best_epoch}, Best Loss: {best_loss}")

        # Save last 10 epochs
        if args.num_epochs - 10 < epoch < args.num_epochs:
            current_weights = copy.deepcopy(model.state_dict())
            torch.save(current_weights,
                   os.path.join(output_dir, f"{epoch}_{run_name}.pth"))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file', type=str, required=True)
    parser.add_argument('--eval-file', type=str, required=True)
    parser.add_argument('--outputs-dir', type=str, required=True)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-epochs', type=int, default=80)
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    cudnn.benchmark = True
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = DSDMSR_VDSR_x8(device).to(device)
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    train_dataset = Patched_DTU_Tensor(args.train_file)
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=True)
    valid_dataset = Patched_DTU_Tensor(args.eval_file)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1)

    sr_train(model, train_dataloader, valid_dataloader, device, args)
