import yaml
from torchvision import datasets
import log


if __name__ == '__main__':
    with open(r'./backdoor/params.yaml', 'r', encoding='utf-8') as f:
        params = yaml.safe_load(f)

    logger = log.Logger(params['log_file']).logger
    logger.info('Start training...')

    pass
