# -*- coding:utf-8 -*-
import sys
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F

RAND_MODE = False


def train_epoch(deep_model, optimizer, train_load, epoch, config, log_file):
    deep_model.train()
    loss_ce_ave = Averager()
    loss_l1_ave = Averager()
    acc_ave = Averager()
    loop = tqdm(total=len(train_load))

    for name, module in deep_model.named_modules():
        if isinstance(module, MaskConv):
            optimizer.update_mask(module.weight, module.mask)
    infer_imgs = []
    for i, data in enumerate(train_load):
        img, label = data
        img, label = img.cuda(), label.cuda()
        selected_ind = random.randint(0, img.size(0) - 3)
        infer_imgs.append(img[selected_ind:selected_ind+2, :, :, :].detach().clone())

        optimizer.zero_grad()

        outs = deep_model(img)
        loss_ce = F.cross_entropy(outs, label)

        loss_ce_ave.add(loss_ce.item())
        loss_ce.backward()
        optimizer.step()

        predicts = torch.argmax(outs, dim=1)
        acc = torch.sum(predicts == label) / label.size(0)
        acc_ave.add(acc.item())

        loop.update(1)
        loop.set_description(f"[epoch {epoch}]")
        loop.set_postfix(loss=loss_ce_ave.item(), acc=acc_ave.item())
    loop.close()
    log_file.info(f"[training accumulation epoch {epoch}]")
    log_file.info(f"[loss_ce {loss_ce_ave.item():.4f}][loss_l1 {loss_l1_ave.item():.4f}]")
    log_file.info(f"[accuracy {acc_ave.item() * 100:.2f}%]")

    if not RAND_MODE and epoch % config.update_freq == 0:
        update_mask(deep_model, torch.cat(infer_imgs, dim=0)[:config.image_cnt, :, :, :], log_file)
        for name, module in deep_model.named_modules():
            if isinstance(module, MaskConv):
                print(np.random.choice(torch.sum(module.mask.squeeze(-1).squeeze(-1), dim=0).detach().cpu().numpy(),20))
    del infer_imgs
    torch.cuda.empty_cache()


def update_mask(deep_model, infer_imgs, log_file):
    for name, module in deep_model.named_modules():
        if isinstance(module, MaskConv):
            module.set_mask_update_enable(True)

    deep_model.eval()
    with torch.no_grad():
        deep_model(infer_imgs)

    disconnect_ave, disuse_ave = Averager(), Averager()
    msg = ""
    for name, module in deep_model.named_modules():
        if isinstance(module, MaskConv):
            if len(module.finished_disuse) < module.out_planes:
                disuse = module.update_use_disuse()
                disuse_ave.add(disuse)
            msg += f"{len(module.finished_disuse)}/{module.out_planes}  "

    log_file.info(msg)
    torch.cuda.empty_cache()
    for name, module in deep_model.named_modules():
        if isinstance(module, MaskConv):
            module.set_mask_update_enable(False)


def test_epoch(deep_model, test_load, epoch, log_file):
    print_ratio = 20
    if epoch % print_ratio == 0:
        mask_list = []
        name_list = []
        for name, module in deep_model.named_modules():
            if isinstance(module, MaskConv):
                mask = torch.where(module.mask.squeeze(-1).squeeze(-1) > 0, 1, 0).cpu().detach().numpy()
                out_c, in_c = mask.shape
                log_file.info(f"[name {name}][mask link ratio {np.sum(mask) / (out_c * in_c)}]")
                mask_list.append(mask)
                name_list.append(name)

    deep_model.eval()
    with torch.no_grad():
        acc_ave = Averager()
        loop = tqdm(enumerate(test_load), total=len(test_load))
        for i, data in loop:
            img, label = data
            img, label = img.cuda(), label.cuda()
            outs = deep_model(img)

            predicts = torch.argmax(outs, dim=1)
            acc = torch.sum(predicts == label) / label.size(0)
            acc_ave.add(acc.item())

        log_file.info(f"[test {epoch}][accuracy {acc_ave.item()*100:.2f}%]")
        torch.cuda.empty_cache()
        return acc_ave.item() * 100


def parse_parameters():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cfg", type=str, default="./config/vgg_cifar.yaml",
                        choices=["./config/vgg_cifar.yaml", "./config/resnet_cifar.yaml"])
    parser.add_argument("-g", "--gpu-id", type=str, default="0",
                        choices=["0", "1", "2", "3", "1,2,3,0"],
                        help="GPU ID")
    parser.add_argument("--release-mode", type=bool, default=True)
    parser.add_argument("--eval", type=bool, default=False)

    parser.add_argument("--auto-resume", type=bool, default=True)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--optimizer", type=str, default="SGD")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--start-epoch", type=int)
    parser.add_argument("--batch-size", type=int)

    param = parser.parse_args()
    config = get_config(param)
    global CLASS_CNT
    CLASS_CNT = config.MODEL.NUM_CLASSES
    return config


def main(config):
    log_file = init_logger(config.OUTPUT, config.DATA.NAME, config.RELEASE_MODE)
    recorder = init_logger(config.OUTPUT, "recorder", config.RELEASE_MODE, file_name="recorder.txt")
    log_file.info(config)
    recorder.info("***********************************")
    recorder.info(f"[SEED {config.SEED}]]")
    recorder.info(f"[update_freq {config.update_freq}]]")
    recorder.info(f"[image_cnt {config.image_cnt}]]")
    recorder.info(f"[sparsity {config.sparsity}]]")
    recorder.info(f"[update_epoch {config.update_epoch}]]")

    setup_seed(config.SEED)

    deep_model = build_model(config)
    log_file.info(deep_model)
    train_load, test_load, _ = build_dataset(config)

    optimizer = build_optimizer(deep_model, config)
    lr_scheduler = build_scheduler(optimizer, config)

    for name, module in deep_model.named_modules():
        if isinstance(module, MaskConv):
            if module.in_planes <= 10:
                module.connect_cnt = module.in_planes
            else:
                module.connect_cnt = int(module.in_planes * config.sparsity)
            module.out_link = (module.out_planes * module.connect_cnt) / module.in_planes
            update_ratio = 1.0 / (config.update_epoch / config.update_freq)
            module.per_disuse_cnt = max(1, math.ceil((module.in_planes - module.connect_cnt) * update_ratio))
            if RAND_MODE:
                prob = torch.rand(size=(module.out_planes, module.in_planes, 1, 1), dtype=torch.float32)
                topK, _ = torch.topk(prob, k=module.connect_cnt, dim=1)
                threshold = topK[:, -1, :, :].view(module.out_planes, 1, 1, 1).expand_as(prob)
                module.mask.data[:, :, :, :] = torch.where(prob >= threshold, 1.0, 0.0)[:, :, :, :]

    max_acc = 0
    start_epoch = config.TRAIN.START_EPOCH

    log_file.info("Start Training!!!")
    for epoch in range(start_epoch, config.TRAIN.EPOCHS + 1):
        log_file.info("=" * 30)
        lr_scheduler.step(epoch)
        lr = optimizer.param_groups[0]['lr']
        log_file.info(f"[training epoch {epoch}/{config.TRAIN.EPOCHS}][learning_rate {lr:.6f}]")

        train_epoch(deep_model, optimizer, train_load, epoch, config, log_file)

        if epoch % 1 == 0:
            accuracy = test_epoch(deep_model, test_load, epoch, log_file)
            if max_acc <= accuracy:
                max_acc = accuracy
                config.RELEASE_MODE and save_checkpoint(
                    epoch, deep_model, optimizer, lr_scheduler, max_accuracy=max_acc,
                    folder=config.OUTPUT, logger=log_file, save_type="best")
        log_file.info(f"[max_acc {max_acc:.4f}]")
        if epoch % 10 == 0:
            config.RELEASE_MODE and save_checkpoint(
                epoch, deep_model, optimizer, lr_scheduler, max_accuracy=max_acc,
                folder=config.OUTPUT, logger=log_file, save_type="last")
    log_file.info("Finish Training!!!")

    recorder.info(f"[accuracy {accuracy:.4f}]][max_acc {max_acc:.4f}]]")
    recorder.info("***********************************")


if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..')))
    sys.path.append(os.path.abspath(os.path.join(sys.path[0], '..', "..")))
    sys.path.append(sys.path[0])
    from scripts.utils import *
    from scripts.build import *
    from scripts.config_base import get_config
    from scripts.deep_model.mask_conv import MaskConv
    config = parse_parameters()
    config.defrost()
    config.SEED = 48
    config.sparsity = 0.5
    config.update_epoch = 120
    config.update_freq = 3
    config.image_cnt = 128
    config.OUTPUT = os.path.join(config.ROOT_PATH, config.MODEL.NAME, config.DATA.NAME, config.TAG)
    if not os.path.exists(config.OUTPUT):
        os.makedirs(config.OUTPUT)
    config.freeze()
    main(config)
