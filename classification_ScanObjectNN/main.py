"""
for training with resume functions.
Usage:
python main.py --model PointNet --msg demo
or
CUDA_VISIBLE_DEVICES=0 nohup python main.py --model PointNet --msg demo > nohup/PointNet_demo.out &
"""
import argparse
import copy
import os
import logging
import datetime
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
import models as models
from models.pointmlp import Model
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from ScanObjectNN import ScanObjectNN
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np


def parse_args():
    """Parameters"""
    parser = argparse.ArgumentParser('training')
    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')
    parser.add_argument('--dual_net', type=bool, default=False, help='enable dual network')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='PointNet', help='model name [default: pointnet_cls]')
    parser.add_argument('--num_classes', default=15, type=int, help='default value for classes of ScanObjectNN')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--num_points', type=int, default=1024, help='Point Number')
    parser.add_argument('--min_lr', default=0.005, type=float, help='min lr')
    parser.add_argument('--gpu', type=int, default=0, help='device selection')
    parser.add_argument('--learning_rate', default=0.01, type=float, help='learning rate in training')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--smoothing', action='store_true', default=False, help='loss smoothing')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=4, type=int, help='workers')
    parser.add_argument('--last_layer_concat', default='concat', type=str, help='last layer concatenation')
    parser.add_argument('--add_factor', default=1.0, type=float, help='last layer concatenation')

    # sparse net parameters
    parser.add_argument('--num_points_low', type=int, default=1024, help='Point Number')
    parser.add_argument('--neighbours_low', type=int, default=32, help='Point Number')

    # dense net parameteres
    parser.add_argument('--num_points_high', type=int, default=2048, help='Point Number')
    parser.add_argument('--neighbours_high', type=int, default=32, help='Point Number')
    parser.add_argument('--num_channel', type=int, default=32, help='Point Number')

    return parser.parse_args()



def main():
    args = parse_args()
    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        device = 'cuda'
        if args.seed is not None:
            torch.cuda.manual_seed(args.seed)
    else:
        device = 'cpu'
    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        message = time_str
    else:
        message = "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + message + '-' + str(args.seed)
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    # Model
    printf(f"args: {args}")
    printf('==> Building model..')
    sparse_net = Model(points=args.num_points_low, k_neighbors=[args.neighbours_low] * 4, parser_args=args)
    if args.dual_net and args.last_layer_concat == 'concat':
        sparse_net.classifier[0] = torch.nn.Linear(1280, 512)
    dense_net = Model(points=args.num_points_high, class_num= args.num_classes, embed_dim=args.num_channel, groups=1,
                      res_expansion=1.0,
                      activation="relu", bias=False, use_xyz=False, normalize="anchor",
                      dim_expansion=[2, 2, 2, 2], pre_blocks=[2, 2, 2, 2], pos_blocks=[2, 2, 2, 2],
                      k_neighbors=[args.neighbours_high] * 4, reducers=[2, 2, 2, 2], parser_args=args)

    total_param = sum(p.numel() for p in sparse_net.parameters() if p.requires_grad) + sum(
        p.numel() for p in dense_net.parameters() if p.requires_grad)
    printf(f"total param: {total_param / 1e6:.2f}M")

    criterion = cal_loss
    sparse_net = sparse_net.to(device)
    dense_net = dense_net.to(device)
    # criterion = criterion.to(device)
    # if device == 'cuda':
    #     sparse_net = torch.nn.DataParallel(sparse_net)
    #     dense_net = torch.nn.DataParallel(dense_net)
    #     cudnn.benchmark = True

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        sparse_net.load_state_dict(checkpoint['sparse_net'])
        dense_net.load_state_dict(checkpoint['dense_net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    train_loader = DataLoader(ScanObjectNN(partition='training', num_points=args.num_points_high, dual_net=args.dual_net,
                                          num_points_low=args.num_points_low), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ScanObjectNN(partition='test', num_points=args.num_points_high, dual_net=args.dual_net,
                                        num_points_low=args.num_points_low), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    try:
        from thop import profile, clever_format
        data = next(iter(train_loader))
        data, data2, label = data


        data, data2 = data.permute(0, 2, 1), data2.permute(0, 2, 1)
        data, data2 = data.to(device), data2.to(device)

        # dense net
        flops, params = profile(copy.deepcopy(dense_net), inputs=(data,))
        flops2, params2 = clever_format([flops, params])
        dense_logits, inter_x = dense_net(data)
        print('\n# Model Params: {} FLOPs: {}'.format(params2, flops2))

        # sparse net
        flops, params = profile(copy.deepcopy(sparse_net), inputs=(data2, inter_x,))
        flops1, params1 = clever_format([flops, params])
        print('\n# Model Params: {} FLOPs: {}'.format(params1, flops1))

    except Exception as e:
        print(e)
        print("could not calculate flops")

    optimizer = torch.optim.SGD(list(sparse_net.parameters()) + list(dense_net.parameters()), lr=args.learning_rate,
                                momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.learning_rate / 100, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(sparse_net, dense_net, train_loader, optimizer, criterion,
                          device, epoch, args)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(sparse_net, dense_net, test_loader, criterion, device, epoch, args)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            sparse_net, dense_net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # best test accuracy
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


def train(sparse_net, dense_net, trainloader, optimizer, criterion, device, epoch, args):
    sparse_net.train()
    dense_net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, data in enumerate(trainloader):
        if len(data) == 2:
            data, label = data
            data = data.permute(0, 2, 1)
            data, label = data.to(device), label.to(device).squeeze()
        elif len(data) == 3:
            data, data2, label = data
            data = data.permute(0, 2, 1)
            data2 = data2.permute(0, 2, 1)
            data, data2, label = data.to(device), data2.to(device), label.to(device).squeeze()

        optimizer.zero_grad()
        if batch_idx == 0 and epoch == 0:
            debug = True
        else:
            debug = False

        if args.dual_net:
            dense_logits, inter_x = dense_net(data, debug=debug)
            logits, _ = sparse_net(data2, inter_x, debug=debug)
        else:
            logits, _ = sparse_net(data, None, debug=debug)

        loss = criterion(logits, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        preds = logits.max(dim=1)[1]

        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())

        total += label.size(0)
        correct += preds.eq(label).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }


def validate(sparse_net, dense_net, testloader, criterion, device, epoch, args):
    sparse_net.eval()
    dense_net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, data in enumerate(testloader):
            if len(data) == 2:
                data, label = data
                data = data.permute(0, 2, 1)
                data, label = data.to(device), label.to(device).squeeze()
            elif len(data) == 3:
                data, data2, label = data
                data = data.permute(0, 2, 1)
                data2 = data2.permute(0, 2, 1)
                data, data2, label = data.to(device), data2.to(device), label.to(device).squeeze()

            if args.dual_net:
                dense_logits, inter_x = dense_net(data, debug=False)
                logits, _ = sparse_net(data2, inter_x, debug=False)
            else:
                logits, _ = sparse_net(data, None, debug=False)

            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }


if __name__ == '__main__':
    main()

    try:
        print("Finish Training. Canceling job...")
        os.system('scancel %s' % os.environ["SLURM_ARRAY_JOB_ID"])
    except:
        print("Finish Training...")
