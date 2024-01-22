import time

from loss.distortion import *
from test import test
from utils import *


def train(args, config, train_loader, test_loader, net, ):
    net.train()
    model_params = [{'params': net.parameters(), 'lr': 0.0001}]
    optimizer = torch.optim.Adam(model_params, lr=config.learning_rate)
    elapsed, losses, psnrs, msssims, cbrs, snrs = [AverageMeter() for _ in range(6)]
    metrics = [elapsed, losses, psnrs, msssims, cbrs, snrs]
    if torch.cuda.is_available():
        if args.trainset == 'CIFAR10':
            CalcuSSIM = MS_SSIM(window_size=3, data_range=1., levels=4, channel=3).to(config.device)
        else:
            CalcuSSIM = MS_SSIM(data_range=1., levels=4, channel=3).to(config.device)
    else:
        config.logger.info("No support for CPU, please install CUDA.")
        return

    for epoch in range(config.tot_epoch):

        if args.trainset == 'CIFAR10':
            for batch_idx, (input, label) in enumerate(train_loader):
                start_time = time.time()
                input = input.to(config.device)
                recon_image, CBR, SNR, mse, loss = net(input)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                elapsed.update(time.time() - start_time)
                losses.update(loss.item())
                cbrs.update(CBR)
                snrs.update(SNR)
                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - CalcuSSIM(input, recon_image.clamp(0., 1.)).mean().item()
                    msssims.update(msssim)
                else:
                    psnr, msssim = 100, 100
                    psnrs.update(100)
                    msssims.update(100)

                if ((batch_idx + 1) % config.print_step) == 0 or batch_idx + 1 == len(train_loader):
                    process = ((batch_idx + 1) % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{(batch_idx + 1) % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {config.learning_rate}',
                    ]))
                    config.logger.info(log)
                    for i in metrics:
                        i.clear()
        else:
            for batch_idx, (input, label) in enumerate(train_loader):
                start_time = time.time()
                input = input.to(config.device)
                recon_image, CBR, SNR, mse, loss = net(input)

                if mse.item() > 0:
                    psnr = 10 * (torch.log(255. * 255. / mse) / np.log(10))
                    psnrs.update(psnr.item())
                    msssim = 1 - loss
                    msssims.update(msssim)

                else:
                    psnrs.update(100)
                    msssims.update(100)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                elapsed.update(time.time() - start_time)
                losses.update(loss.item())
                cbrs.update(CBR)
                snrs.update(SNR)

                if ((batch_idx + 1) % config.print_step) == 0:
                    process = ((batch_idx + 1) % train_loader.__len__()) / (train_loader.__len__()) * 100.0
                    log = (' | '.join([
                        f'Epoch {epoch}',
                        f'Step [{(batch_idx + 1) % train_loader.__len__()}/{train_loader.__len__()}={process:.2f}%]',
                        f'Time {elapsed.val:.3f}',
                        f'Loss {losses.val:.3f} ({losses.avg:.3f})',
                        f'CBR {cbrs.val:.4f} ({cbrs.avg:.4f})',
                        f'SNR {snrs.val:.1f} ({snrs.avg:.1f})',
                        f'PSNR {psnrs.val:.3f} ({psnrs.avg:.3f})',
                        f'MSSSIM {msssims.val:.3f} ({msssims.avg:.3f})',
                        f'Lr {config.learning_rate}',
                    ]))
                    config.logger.info(log)
                    for i in metrics:
                        i.clear()

        for i in metrics:
            i.clear()

        if (epoch + 1) % config.save_model_freq == 0:
            print(os.path.join(config.models, '{}_EP{}.model'.format(config.filename, epoch + 1)))
            save_model(net, save_path=os.path.join(config.models, '{}_EP{}.model'.format(config.filename, epoch + 1)))
            test(args, config, test_loader, net)
