from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch


def test(epoch, net, dataset, args):
    net.eval()
    test_loss = 0
    total = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.test_bs)

    with torch.no_grad():
        for img, lab in data_loader:
            img, lab = img.to(args.device), lab.to(args.device)
            total += lab.size(0)
            out = net(img)

            test_loss += f.cross_entropy(out, lab, reduction='sum').item()

            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(lab.data.view_as(pred)).long().cpu().sum()

    test_loss /= total
    accuracy = correct / total
    args.logger.info('Epoch: {}\taverage loss: {:.4f}\tacc: {}/{} ({:.2f})'
                     .format(epoch, test_loss, correct, total, accuracy))
    return accuracy, test_loss
