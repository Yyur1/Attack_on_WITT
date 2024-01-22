import os

import torch
import torch.nn as nn
import torchvision.models as models
from torch import optim

from utils import save_model


# Mainly ResNet 18
def get_classifier(name: str, num_class):
    classifier = None
    if name.lower() == 'resnet18':
        classifier = models.resnet18(True)
        classifier.fc = nn.Linear(classifier.fc.in_features, num_class)

    # You can add more classifiers
    return classifier


def downstream(config, train: bool, net, classifier, backdoor_dataloader, non_backdoor_dataloader, task: str):
    if task.lower() == "classify":
        if train:
            optimizer = optim.Adam(classifier.parameters(), lr=config.downstream_learning_rate, )
            criterion = nn.CrossEntropyLoss()
            for epoch in range(config.downstream_epoch):
                sum_loss, correct, total = 0.0, 0, 0

                for batch_idx, (input, label) in enumerate(backdoor_dataloader):
                    input, label = input.to(config.device), label.to(config.device)
                    with torch.no_grad():
                        recon_image, CBR, SNR, mse, loss_G = net(input)

                    optimizer.zero_grad()
                    output = classifier(recon_image)
                    loss = criterion(output, label)
                    loss.backward()
                    optimizer.step()

                    # calculate the precision of backdoor images and normal images
                    sum_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()
                    if ((batch_idx + 1) % config.print_step) == 0 or batch_idx + 1 == len(backdoor_dataloader):
                        config.logger.info(
                            'Train [epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% | Correct: %d | SNR: %d' % (
                                epoch + 1, (batch_idx + 1 + epoch * len(backdoor_dataloader)),
                                sum_loss / (batch_idx + 1),
                                (100. * correct / total), predicted.eq(label.data).cpu().sum(), SNR))

                if (epoch + 1) % 100 == 0:
                    save_model(classifier, save_path=os.path.join(config.models,
                                                                  f"%{config.portion * 100}backdoor_{epoch + 1}epoch_classifier.model"))

        else:  # we need to synthesize
            correct, total = 0, 0
            with torch.no_grad():
                sample_list = list(backdoor_dataloader.batch_sampler)
                backdoor_sum = [0, 0, 0, 0, 0]
                backdoor_correct = [0, 0, 0, 0, 0]
                non_backdoor_sum = [0, 0, 0, 0, 0]
                non_backdoor_correct = [0, 0, 0, 0, 0]
                non_backdoor_iter = iter(non_backdoor_dataloader)
                for batch_idx, (input, label) in enumerate(backdoor_dataloader):
                    # calculate the precision of backdoor images and normal images
                    is_backdoor_list = [1 if x in backdoor_dataloader.perm else 0 for x in set(sample_list[batch_idx])]
                    input, label = input.to(config.device), label.to(config.device)
                    (non_backdoor_input, non_backdoor_label) = next(non_backdoor_iter)
                    non_backdoor_input = non_backdoor_input.to(config.device)
                    recon_image, CBR, SNR, mse, loss_G = net(input, non_backdoor_input_image=non_backdoor_input)
                    output = classifier(recon_image)
                    _, predicted = torch.max(output.data, 1)
                    total += label.size(0)
                    correct += predicted.eq(label.data).cpu().sum()
                    sum_backdoor = sum(is_backdoor_list)
                    backdoor_sum[(SNR - 1) // 3] += sum_backdoor
                    non_backdoor_sum[(SNR - 1) // 3] += backdoor_dataloader.batch_size - sum_backdoor

                    backdoor_correct_temp, non_backdoor_correct_temp = 0, 0
                    # calculate the precision in backdoor images and non-backdoor images
                    for i in range(backdoor_dataloader.batch_size):
                        if predicted.eq(label.data).cpu()[i] and is_backdoor_list[i] == 1:
                            backdoor_correct_temp += 1
                        elif predicted.eq(label.data).cpu()[i] and is_backdoor_list[i] == 0:
                            non_backdoor_correct_temp += 1

                    config.logger.info(
                        'Test [iter:%d] Acc: %.3f%% | Correct: %d | SNR: %d | Backdoor Correct: %d | Non-Backdoor Correct: %d' % (
                            (batch_idx + 1), (100. * correct / total), predicted.eq(label.data).cpu().sum(), SNR,
                            backdoor_correct_temp, non_backdoor_correct_temp))

                    backdoor_correct[(SNR - 1) // 3] += backdoor_correct_temp
                    non_backdoor_correct[(SNR - 1) // 3] += non_backdoor_correct_temp

            backdoor_recall, non_backdoor_recall = [], []
            for snr in range(5):
                backdoor_recall.append(backdoor_correct[snr] / backdoor_sum[snr])
                non_backdoor_recall.append(non_backdoor_correct[snr] / non_backdoor_sum[snr])
            config.logger.info(f"backdoor_recall:{backdoor_recall}, non_backdoor_recall:{non_backdoor_recall}")
