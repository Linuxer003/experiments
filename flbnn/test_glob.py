from torch.utils.data import DataLoader
import torch.nn.functional as f
import torch


def test(net, dataset, params):
    net.eval()
    test_loss = 0
    correct = 0
    data_loader = DataLoader(dataset, batch_size=params['test_bs'])

    with torch.no_grad():
        for img, lab in data_loader:

            img, lab = img.to(torch.device(params['device'])), lab.to(torch.device(params['device']))

            out = net(img)

            test_loss += f.cross_entropy(out, lab, reduction='sum').item()

            pred = out.data.max(1, keepdim=True)[1]
            correct += pred.eq(lab.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(data_loader.dataset)
    accuracy = 100.00 * correct / len(data_loader.dataset)

    return accuracy, test_loss
