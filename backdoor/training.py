import time
import random
import yaml
import torch
import torch.nn as nn
import torch.functional as f
import math

import log

logger = log.Logger(r'../log.txt').logger


def train(helper, epoch, train_data_sets, local_model, target_model, is_poison):
    """

    :param helper:
    :param epoch:
    :param train_data_sets:
    :param local_model:
    :param target_model:
    :param is_poison:
    :return:
    """

    weight_accumulator = dict()
    for name, data in target_model.state_dict().items():
        weight_accumulator[name] = torch.zeros_like(data)

    target_params_variables = dict()
    for name, param in target_model.named_parameters():
        target_params_variables[name] = target_model.state_dict()[name].clone().detach().requires_grad_(False)

    current_number_of_adversaries = 0
    for model_id, _ in train_data_sets:
        if model_id == -1 or model_id in helper.params['adversary_list']:
            current_number_of_adversaries += 1
    logger.info(f'There are {current_number_of_adversaries} adversaries in the training.')

    for model_id in range(helper.params['num_models']):

        model = local_model
        model.copy_params(target_model.state_dict())
        optimizer = torch.optim.SGD(model.parameters(), lr=helper.params['lr'],
                                    momentum=helper.params['momentum'],
                                    weight_decay=helper.params['decay'])
        model.train()

        start_time = time.time()
        _, (current_data_model, train_data) = train_data_sets[model_id]

        if current_data_model == -1:
            continue
        if is_poison and current_data_model in helper.params['adversary_list'] and \
                (epoch in helper.params['poison_epochs'] or helper.params['random_compromise']):
            logger.info('poison now...')
            poisoned_data = helper.poisoned_data_for_train

            _, acc_p = test_poison(helper=helper, epoch=epoch,
                                   data_source=helper.test_data_poison,
                                   model=model, is_poison=True)
            _, acc_initial = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                  model=model, is_poison=False)
            logger.info(acc_p)
            poison_lr = helper.params['poison_lr']
            if not helper.params['baseline']:
                if acc_p > 20:
                    poison_lr /= 50
                if acc_p > 60:
                    poison_lr /= 100

            retain_num_times = helper.params['retrain_poison']
            step_lr = helper.params['poison_step_lr']

            poison_optimizer = torch.optim.SGD(model.parameters(), lr=poison_lr, momentum=helper.params['momentum'],
                                               weight_decay=helper.params['decay'])
            scheduler = torch.optim.lr_scheduler.MultiStepLR(poison_optimizer,
                                                             milestones=[0.2 * retain_num_times,
                                                                         0.8 * retain_num_times],
                                                             gamma=0.1)

            try:

                for internal_epoch in range(1, retain_num_times + 1):
                    if step_lr:
                        scheduler.step()
                        logger.info(f'Current lr: {scheduler.get_lr()}')
                    data_iterator = poisoned_data

                    for batch_id, batch in enumerate(data_iterator):
                        for i in range(helper.params['poisoning_per_batch']):
                            for pos, image in enumerate(helper.params['poison_images']):
                                poison_pos = len(helper.params['poison_images']) * i + pos

                                batch[0][poison_pos] = helper.train_dataset[image][0]
                                batch[0][poison_pos].add_(torch.FloatTensor(batch[0][poison_pos].shape).normal_(0, helper.params['noise_level']))

                                batch[1][poison_pos] = helper.params['poison_label_swap']

                        data, targets = helper.get_batch(poisoned_data, batch, False)

                        poison_optimizer.zero_grad()

                        output = model(data)
                        class_loss = nn.CrossEntropyLoss(output, targets)

                        distance_loss = helper.model_dist_norm_var(model, target_params_variables)

                        loss = helper.params['alpha_loss'] * class_loss + (1 - helper.params['alpha_loss']) * distance_loss
                        loss.backward()

                        if helper.params['diff_privacy']:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), helper.params['clip'])
                            poison_optimizer.step()

                            model_norm = helper.model_dist_norm(model, target_params_variables)
                            if model_norm > helper.params['s_norm']:
                                logger.info(
                                    f'The limit reached for distance: '
                                    f'{helper.model_dist_norm(model, target_params_variables)}')
                                norm_scale = helper.params['s_norm'] / model_norm
                                for name, layer in model.named_parameters():
                                    clipped_difference = norm_scale * (
                                            layer.data - target_model.state_dict()[name])
                                    layer.data.copy_(
                                        target_model.state_dict()[name] + clipped_difference)

                        else:
                            poison_optimizer.step()
                    loss, acc = test(helper=helper, epoch=epoch, data_source=helper.test_data,
                                     model=model, is_poison=False)
                    loss_p, acc_p = test_poison(helper=helper, epoch=internal_epoch,
                                                data_source=helper.test_data_poison,
                                                model=model, is_poison=True)

                    if loss_p <= 0.0001:
                        if helper.params['type'] == 'image' and acc < acc_initial:
                            if step_lr:
                                scheduler.step()
                            continue

                        raise ValueError()
                    logger.error(
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')
            except ValueError:
                logger.info('Converged earlier')

            logger.info(f'Global model norm: {helper.model_global_norm(target_model)}.')
            logger.info(f'Norm before scaling: {helper.model_global_norm(model)}. '
                        f'Distance: {helper.model_dist_norm(model, target_params_variables)}')

            # Adversary wants to scale his weights. Baseline model doesn't do this
            if not helper.params['baseline']:
                # We scale data according to formula: L = 100*X-99*G = G + (100*X- 100*G).
                clip_rate = helper.params['scale_weights'] / current_number_of_adversaries
                logger.info(f"Scaling by  {clip_rate}")
                for key, value in model.state_dict().items():
                    # don't scale tied weights:
                    if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                        continue
                    target_value = target_model.state_dict()[key]
                    new_value = target_value + (value - target_value) * clip_rate

                    model.state_dict()[key].copy_(new_value)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            if helper.params['diff_privacy']:
                model_norm = helper.model_dist_norm(model, target_params_variables)

                if model_norm > helper.params['s_norm']:
                    norm_scale = helper.params['s_norm'] / model_norm
                    for name, layer in model.named_parameters():
                        # don't scale tied weights:
                        if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                            continue
                        clipped_difference = norm_scale * (layer.data - target_model.state_dict()[name])
                        layer.data.copy_(target_model.state_dict()[name] + clipped_difference)
                distance = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'Scaled Norm after poisoning and clipping: '
                    f'{helper.model_global_norm(model)}, distance: {distance}')

            for key, value in model.state_dict().items():
                # don't scale tied weights:
                if helper.params.get('tied', False) and key == 'decoder.weight' or '__' in key:
                    continue
                target_value = target_model.state_dict()[key]
                new_value = target_value + (value - target_value) * current_number_of_adversaries
                model.state_dict()[key].copy_(new_value)
            distance = helper.model_dist_norm(model, target_params_variables)
            logger.info(f"Total norm for {current_number_of_adversaries} "
                        f"adversaries is: {helper.model_global_norm(model)}. distance: {distance}")

        else:
            # we will load helper.params later
            if helper.params['fake_participants_load']:
                continue

            for internal_epoch in range(1, helper.params['retrain_no_times'] + 1):
                total_loss = 0.

                data_iterator = train_data
                for batch_id, batch in enumerate(data_iterator):
                    optimizer.zero_grad()
                    data, targets = helper.get_batch(train_data, batch, evaluation=False)

                    output = model(data)
                    loss = f.cross_entropy(output, targets)

                    loss.backward()

                    if helper.params['diff_privacy']:
                        optimizer.step()
                        model_norm = helper.model_dist_norm(model, target_params_variables)

                        if model_norm > helper.params['s_norm']:
                            norm_scale = helper.params['s_norm'] / model_norm
                            for name, layer in model.named_parameters():
                                # don't scale tied weights:
                                if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                                    continue
                                clipped_difference = norm_scale * (
                                        layer.data - target_model.state_dict()[name])
                                layer.data.copy_(
                                    target_model.state_dict()[name] + clipped_difference)
                    else:
                        optimizer.step()

                    total_loss += loss.data

                    if helper.params["report_train_loss"] and batch % helper.params['log_interval'] == 0 and batch > 0:
                        cur_loss = total_loss.item() / helper.params['log_interval']
                        elapsed = time.time() - start_time
                        logger.info('model {} | epoch {:3d} | internal_epoch {:3d} '
                                    '| {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                                    'loss {:5.2f} | ppl {:8.2f}'
                                    .format(model_id, epoch, internal_epoch,
                                            batch, train_data.size(0) // helper.params['bptt'],
                                            helper.params['lr'],
                                            elapsed * 1000 / helper.params['log_interval'],
                                            cur_loss,
                                            math.exp(cur_loss) if cur_loss < 30 else -1.))
                        total_loss = 0
                        start_time = time.time()

            if helper.params['track_distance'] and model_id < 10:
                # we can calculate distance to this model now.
                distance_to_global_model = helper.model_dist_norm(model, target_params_variables)
                logger.info(
                    f'MODEL {model_id}. P-norm is {helper.model_global_norm(model):.4f}. '
                    f'Distance to the global model: {distance_to_global_model:.4f}. '
                    f'Dataset size: {train_data.size(0)}')

        for name, data in model.state_dict().items():
            # don't scale tied weights:
            if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                continue
            weight_accumulator[name].add_(data - target_model.state_dict()[name])

        if helper.params["fake_participants_save"]:
            torch.save(weight_accumulator,
                       f"{helper.params['fake_participants_file']}_"
                       f"{helper.params['s_norm']}_{helper.params['no_models']}")
        elif helper.params["fake_participants_load"]:
            fake_models = helper.params['no_models'] - helper.params['number_of_adversaries']
            fake_weight_accumulator = torch.load(
                f"{helper.params['fake_participants_file']}_{helper.params['s_norm']}_{fake_models}")
            logger.info(f"Faking data for {fake_models}")
            for name in target_model.state_dict().keys():
                # don't scale tied weights:
                if helper.params.get('tied', False) and name == 'decoder.weight' or '__' in name:
                    continue
                weight_accumulator[name].add_(fake_weight_accumulator[name])

        return weight_accumulator


def test(helper, epoch, data_source, model, is_poison=False):
    model.eval()
    total_loss = 0
    correct = 0
    
    dataset_size = len(data_source.dataset)
    data_iterator = data_source

    for batch_id, batch in enumerate(data_iterator):
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        
        output = model(data)
        total_loss += f.cross_entropy(output, targets, reduction='sum').item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().item()

    acc = 100.0 * (float(correct) / float(dataset_size))
    total_l = total_loss / dataset_size

    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.4f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))
    model.train()
    return total_l, acc


def test_poison(helper, epoch, data_source, model, is_poison=False):
    model.eval()
    total_loss = 0.0
    correct = 0.0
    
    data_iterator = data_source
    dataset_size = 1000

    for batch_id, batch in enumerate(data_iterator):
        for pos in range(len(batch[0])):
            batch[0][pos] = helper.train_dataset[random.choice(helper.params['poison_images_test'])][0]

            batch[1][pos] = helper.params['poison_label_swap']
        data, targets = helper.get_batch(data_source, batch, evaluation=True)
        output = model(data)
        total_loss += f.cross_entropy(output, targets, reduction='sum').data.item()  # sum up batch loss
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(targets.data.view_as(pred)).cpu().sum().to(dtype=torch.float)

    acc = 100.0 * (correct / dataset_size)
    total_l = total_loss / dataset_size
    logger.info('___Test {} poisoned: {}, epoch: {}: Average loss: {:.4f}, '
                'Accuracy: {}/{} ({:.0f}%)'.format(model.name, is_poison, epoch,
                                                   total_l, correct, dataset_size,
                                                   acc))
    model.train()
    return total_l, acc
