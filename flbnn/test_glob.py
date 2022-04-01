from torch.utils.data import DataLoader
import torch.nn.functional as f


def test(net, dataset, args):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=args.test_bs)

    for img, lab in data_loader:
        img, lab = img.to(args.device), lab.to(args.device)
        out = net(img)

        test_loss += f.cross_entropy(out, lab, reduction='sum').item()

        pred = out.data.max(1, keepdim=True)[1]
        correct += pred.eq(lab.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)
    args.logger.info('Average loss: {:.4f}  acc: {}/{} ({:.2f}%)'
                     .format(test_loss, correct, len(data_loader.dataset), accuracy))
    return accuracy, test_loss
