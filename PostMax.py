




import torch
import numpy as np
import scipy.stats


class PostMax:
    def __init__(self, norm: int = 2):
        """
        PostMax scoring using Generalized Pareto Distribution (GPD).

        Parameters
        ----------
        norm : int
            Norm type (e.g. 1, 2, or np.inf) for feature normalization.
        """
        self.norm = norm

    def score(self, model: dict, norm_logits: np.ndarray):
        """
        Compute probabilities from normalized logits using a fitted GPD.

        Parameters
        ----------
        model : dict
            GPD parameters {'shape', 'loc', 'scale'}.
        norm_logits : np.ndarray
            Normalized logits for samples.

        Returns
        -------
        torch.Tensor
            Probabilities mapped via GPD CDF.
        """
        probs = scipy.stats.genpareto.cdf(norm_logits, shape=model['shape'], loc=model['loc'], scale=model['scale'])
        return torch.from_numpy(probs)

    def train(self, labels: torch.Tensor, features: torch.Tensor, logits: torch.Tensor):
        """
        Fit GPD to normalized logits.

        Parameters
        ----------
        labels : torch.Tensor
            Ground-truth labels [N].
        features : torch.Tensor
            Feature embeddings [N, D].
        logits : torch.Tensor
            Class logits [N, C].

        Returns
        -------
        dict
            Fitted GPD parameters.
        """
        assert labels.shape[0] == features.shape[0] == logits.shape[0], "Tensors must have the same batch dimension."

        mask = labels == torch.argmax(logits, dim=1)
        filt_labels = labels[mask]
        filt_features = features[mask]
        filt_logits = logits[mask]

        norm_logits = []
        for cls_id in range(filt_logits.shape[1]):
            cls_features = filt_features[filt_labels == cls_id]
            cls_logits = filt_logits[filt_labels == cls_id]

            if cls_features.numel() == 0:
                continue

            max_cls_logits = cls_logits[:, cls_id]
            norm_cls_logits = max_cls_logits / torch.norm(cls_features, p=self.norm, dim=1)
            norm_logits.append(norm_cls_logits)

        norm_logits = torch.cat(norm_logits, dim=0).numpy()
        shape, loc, scale = scipy.stats.genpareto.fit(norm_logits)

        return {'shape': shape, 'loc': loc, 'scale': scale}

    def evaluate(self, model: dict, labels: torch.Tensor, features: torch.Tensor, logits: torch.Tensor, pct: float = 1.0):
        """
        Evaluate model and compute scores.

        Parameters
        ----------
        model : dict
            GPD parameters.
        labels : torch.Tensor
            Ground-truth labels [N].
        features : torch.Tensor
            Feature embeddings [N, D].
        logits : torch.Tensor
            Class logits [N, C].
        pct : float, default=1.0
            Fraction of dataset to evaluate (e.g. 0.5 for 50%).

        Returns
        -------
        torch.Tensor
            Tensor of shape [M, 3] with (label, prediction, probability).
        """
        assert labels.shape[0] == features.shape[0] == logits.shape[0], "Tensors must have the same batch dimension."

        if pct < 1.0:
            num_imgs = int(labels.shape[0] * pct)
            labels, features, logits = labels[:num_imgs], features[:num_imgs], logits[:num_imgs]

        max_logits, preds = torch.max(logits, dim=1)
        norm_logits = max_logits / torch.norm(features, p=self.norm, dim=1)
        probs = self.score(model, norm_logits.numpy())

        # Stack results into [N, 3]
        results = torch.stack((labels, preds, probs), dim=1)
        
        return results
