from Dataset import *
from NN import *
import random
import scipy.linalg as linalg
import sys
import numpy as np

image_size = 512
batch_size = 50

transform = transforms.Compose(
    [transforms.ToPILImage(),
     transforms.Resize([image_size, image_size]),
     transforms.ToTensor()])


def data_augment(x):
    # x is a 600 x 600 x nc ndarray
    input_size = 600
    output_size = 512
    th, tw = output_size, output_size
    if input_size == tw and input_size == th:
        i, j = 0, 0
    else:
        i = random.randint(0, input_size - th)
        j = random.randint(0, input_size - tw)
    return x[i:(i + tw), j:(j + th), :]


with torch.no_grad():
    device = torch.device("cuda:0")
    netC = InceptionDiscriminator(9).to(device)
    netCparam = torch.load("models\\ClassifierWithunet2.nn")
    netC.load_state_dict(netCparam)
    netC.eval()

    # Real data Gaussian empiric parameter
    dataSet = SatelliteDataset(sys.argv[2], transform=transform, data_augment=data_augment)
    dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=False, num_workers=0)

    sum = np.zeros(512)
    for data in dataloader:
        data, label = data
        data = data.to(device)

        netC(data)
        features = netC.feature_layer.cpu().numpy()
        sum += np.sum(features, axis=0)
    mu_1 = sum / sum.shape[0]

    sum_sq = np.zeros((512, 512))
    for data in dataloader:
        data, label = data
        data = data.to(device)

        netC(data)
        features = netC.feature_layer.cpu().numpy() - mu_1
        sum_sq += np.matmul(np.transpose(features), features)
    cov_1 = sum_sq/sum_sq.shape[0]

    # Fake data Gaussian empiric parameter
    dataSet = SatelliteDataset(sys.argv[3], transform=transform, data_augment=data_augment)
    dataloader = torch.utils.data.DataLoader(dataSet, batch_size=batch_size, shuffle=False, num_workers=0)

    sum = np.zeros(512)
    for data in dataloader:
        data, label = data
        data = data.to(device)

        netC(data)
        features = netC.feature_layer.cpu().numpy()
        sum += np.sum(features, axis=0)
    mu_2 = sum / sum.shape[0]

    sum_sq = np.zeros((512, 512))
    for data in dataloader:
        data, label = data
        data = data.to(device)

        netC(data)
        features = netC.feature_layer.cpu().numpy() - mu_1
        sum_sq += np.matmul(np.transpose(features), features)
    cov_2 = sum_sq / sum_sq.shape[0]

    cov_product = linalg.sqrtm(np.dot(cov_1, cov_2))
    if not np.isfinite(cov_product).all():
        offset = np.eye(cov_1.shape[0]) * 1e-6
        cov_product = linalg.sqrtm((cov_1 + offset).dot(cov_2 + offset))
    
    if np.iscomplexobj(cov_product):
        cov_product = cov_product.real

    delta_mu = mu_1 - mu_2
    one = np.dot(delta_mu, delta_mu)
    two = np.trace(cov_1) + np.trace(cov_2) - 2*np.trace(cov_product)

    fid = one + two
    print(fid)
