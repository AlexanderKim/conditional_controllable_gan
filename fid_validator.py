import numpy as np
import scipy

import torch
from torchvision.models import Inception3
from torch.distributions import MultivariateNormal

import seaborn as sns # This is for visualization


def matrix_sqrt(x):
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)


def frechet_distance(mu_x, mu_y, sigma_x, sigma_y):
    mu = torch.norm(mu_x - mu_y)
    mu *= mu

    tr = sigma_x + sigma_y - 2 * matrix_sqrt(sigma_x @ sigma_y)
    tr = torch.trace(tr)

    return mu + tr


def preprocess(img):
    img = torch.nn.functional.interpolate(img, size=(299, 299), mode='bilinear', align_corners=False)
    return img


def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))

class FIDValidator(object):
    def __init__(self, inception_model: Inception3):
        self.inception_model = inception_model
        self.inception_model.fc = torch.nn.Identity()

    def validate(self, fake, real) -> float:
        real_features_list = []
        fake_features_list = []

        real_preprocessed = preprocess(real)
        real_features = self.inception_model(real_preprocessed).detach().to('cpu')
        real_features_list.append(real_features)

        fake_preprocessed = preprocess(fake)
        fake_features = self.inception_model(fake_preprocessed).detach().to('cpu')
        fake_features_list.append(fake_features)

        real_features_list_cat = torch.cat(real_features_list)
        fake_features_list_cat = torch.cat(fake_features_list)

        mu_fake = torch.mean(fake_features_list_cat, dim=0)
        mu_real = torch.mean(real_features_list_cat, dim=0)

        sigma_fake = get_covariance(fake_features_list_cat)
        sigma_real = get_covariance(real_features_list_cat)

        fid = frechet_distance(mu_real, mu_fake, sigma_real, sigma_fake).item()
        return fid

    def visualise(self, mu_fake, sigma_fake, mu_real, sigma_real):
        indices = [2, 4, 5]
        fake_dist = MultivariateNormal(mu_fake[indices], sigma_fake[indices][:, indices])
        fake_samples = fake_dist.sample((5000,))
        real_dist = MultivariateNormal(mu_real[indices], sigma_real[indices][:, indices])
        real_samples = real_dist.sample((5000,))

        import pandas as pd
        df_fake = pd.DataFrame(fake_samples.numpy(), columns=indices)
        df_real = pd.DataFrame(real_samples.numpy(), columns=indices)
        df_fake["is_real"] = "no"
        df_real["is_real"] = "yes"
        df = pd.concat([df_fake, df_real])
        sns.pairplot(df, plot_kws={'alpha': 0.1}, hue='is_real')

