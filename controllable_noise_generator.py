from typing import List

import torch

from generator import Generator
from n_classes_classifier import NClassesClassifier


class ContorllableNoiseGenerator(object):
    def __init__(self,
                 generator: Generator,
                 classifier: NClassesClassifier):
        self.generator = generator
        self.classifier = classifier

        self.optimizer = torch.optim.Adam(
            params=self.classifier.parameters(),
            lr=3e-4,
            betas=(0.5, 0.999))

    def _calculate_updated_noise(self, noise, weight):
        new_noise = noise + noise.grad * weight

        return new_noise

    def _get_score(self,
                   current_classifications,
                   original_classifications,
                   target_indices,
                   other_indices,
                   penalty_weight):
        other_class_penalty = - torch.mean(
            torch.norm(
                original_classifications[:, other_indices] - current_classifications[:, other_indices],
                dim=1)
        ) * penalty_weight

        target_score = torch.mean(current_classifications[:, target_indices])
        return target_score + other_class_penalty

    def generate_noise(self,
                       target_indices: List[int],
                       other_indices: List[int],
                       n_images: int,
                       grad_steps: int = 10,
                       device='cuda'):
        noise = self.generator.gen_noize(n_images, device=device)
        original_classifications = self.classifier(self.generator(noise))

        fake_history: List[torch.Tensor] = []

        for _ in range(grad_steps):
            self.optimizer.zero_grad()

            fake = self.generator(noise)
            fake_history += fake
            fake_score = self._get_score(
                current_classifications=self.classifier(fake),
                original_classifications=original_classifications,
                target_indices=target_indices,
                other_indices=other_indices,
                penalty_weight=0.1
            )

            fake_score.backward()
            noise.data = self._calculate_updated_noise(noise, 1 / grad_steps)

        return noise, fake_history
