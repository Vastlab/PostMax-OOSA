import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from cycler import cycler
import torch
import random

def OSA(gt: torch.Tensor, pred: torch.Tensor, prob: torch.Tensor, thresh: float, algo_name: str):
    """
    Compute Open-Set Accuracy (OSA) and Unknown Rejection Rate (URR).

    Parameters
    ----------
    gt : torch.Tensor
        Ground truth class labels (1D tensor).
    pred : torch.Tensor
        Predicted class labels (1D tensor).
    prob : torch.Tensor
        Prediction probabilities (1D tensor or 2D with shape [N, C]).
        If 2D, only the first column is used.
    thresh : float
        Operational Threshold.
        If provided, compute OSA at this threshold, otherwise compute operational threshold that maximizes OSA.
    algo_name : str
        Algorithm name.

    Returns
    -------
    If `thresh` is provided:
        tuple
            (total_accuracy, URR, algo_name, thresh_idx), osa_score
            - total_accuracy : torch.Tensor
                OSA across thresholds.
            - URR : torch.Tensor
                Unknown Rejection Rate across thresholds.
            - algo_name : str
                Algorithm name.
            - thresh_idx : int
                Index of the operational threshold.
            - osa_score : float
                OSA achieved at the given threshold.
    If `thresh` is not provided:
        float
            The operational threshold that maximizes OSA.
    """
    
    if len(prob.shape)!= 1:
        prob = prob[:,0]
        
    prob, indices = torch.sort(prob, descending=True)
    pred_class, gt = pred[indices], gt[indices]
    
    # Get unique consecutive values and their counts
    unique_prob, counts = torch.unique_consecutive(torch.flip(prob, [0]), return_counts=True)
    # Reverse them back to original order
    unique_prob, counts = torch.flip(unique_prob, [0]), torch.flip(counts, [0])
    
    # Get labels
    pred_class_unique = pred_class.flatten().unique()
    knowns_idxs = torch.isin(gt, pred_class_unique)
    unknowns_idxs = ~knowns_idxs
    
    # Get denominator for accuracy and Unknown Rejection Rate (URR)
    num_knowns = knowns_idxs.sum().float()
    num_unknowns = unknowns_idxs.sum().float()
    
    all_unknowns = torch.cumsum(unknowns_idxs, dim=-1).float()
    URR = all_unknowns / num_unknowns
    
    correct = torch.any(gt[:, None] == pred_class, dim=1)
    correct = torch.cumsum(correct, dim=-1)
    
    knowns_acc = correct / num_knowns
    threshold_indices = torch.cumsum(counts, dim=-1) - 1
    
    total_accuracy = ((knowns_acc[threshold_indices] * num_knowns) + ((1 - URR[threshold_indices]) * num_unknowns)) / (num_unknowns + num_knowns)
    URR = 1 - URR[threshold_indices]
    
    # If threshold is given, compute OSA and idx
    if thresh:
        thresh_mask = unique_prob < thresh
        thresh_idx = torch.argmax(thresh_mask)
        osa_score = total_accuracy[thresh_idx]
        print('Max OSA', title, osa_score.item())
        
        return (total_accuracy, URR, title, thresh_idx), osa_score.item()
    
    # If threshold is not known, compute and return it.
    else:
        total_accuracy_flipped = torch.flip(total_accuracy, [0])
        max_idx = (total_accuracy_flipped.shape[0]-1) - torch.argmax(total_accuracy_flipped)
        print('Max OSA achieved with threshold:', unique_prob[max_idx].item())
        
        return unique_prob[max_idx].item()
    

def plot_OSA(to_plot: List[Tuple[torch.Tensor, torch.Tensor, str, int]], log: bool = False, filename: Optional[str] = None, title: Optional[str] = None):
    """
    Plot Open-Set Accuracy curve.

    Parameters
    ----------
    to_plot : list of tuples
        Each tuple: (knowns_accuracy, URR, algo_name, thresh_idx)
    log : bool, default=False
        Use log scale on x-axis.
    filename : str, optional
        If provided, saves the figure (adds .pdf if no extension).
    title : str, optional
        Plot title.
    """

    # Cycling of colors + markers (tab10 colors + 7 distinct markers)
    prop_cycle = (cycler(color=plt.cm.tab10.colors) * cycler(marker=['o', 's', '^', 'D', 'v', 'x', '*']))

    fig, ax = plt.subplots()
    ax.set_prop_cycle(prop_cycle)

    if title:
        fig.suptitle(title, fontsize=20)

    for knowns_accuracy, URR, algo_name, thresh_idx in to_plot:
        knowns_acc_flipped = torch.flip(knowns_accuracy, [0])
        max_idx = (knowns_acc_flipped.shape[0] - 1) - torch.argmax(knowns_acc_flipped)
        
        # Draw curve (randomized marker placement)
        markevery = random.randint(int(URR.shape[0] * 0.1), int(URR.shape[0] * 0.2))
        line, = ax.plot(URR, knowns_accuracy, label=algo_name, markevery=markevery)
        
        # Extract the assigned color from this line
        color = line.get_color()

        # Operational threshold
        ax.plot(URR[thresh_idx], knowns_accuracy[thresh_idx],marker="*", markersize=15, markeredgecolor=color, markerfacecolor=color)

        # Oracle performance
        ax.plot(URR[max_idx], knowns_accuracy[max_idx], marker="D", markersize=7, markeredgecolor=color, markerfacecolor="None")

    # Log scale if desired
    if log:
        ax.set_xscale("log")

    ax.set_ylim([0, 1])
    ax.set_ylabel("Open-Set Accuracy", fontsize=18, labelpad=10)
    ax.set_xlabel("Unknown Rejection Rate", fontsize=18, labelpad=10)

    # Legends
    test_thresh = mlines.Line2D([], [], marker='D', linestyle='None', markeredgecolor='Black', markerfacecolor='None', markersize=7, label='Oracle Performance')
    val_thresh = mlines.Line2D([], [], color='Black', marker='*', linestyle='None',markersize=15, label='Operational Performance')

    # First legend: oracle/operational
    star_legend = ax.legend(handles=[test_thresh, val_thresh], loc="lower right", ncol=1, fontsize=10, frameon=False)
    ax.add_artist(star_legend)

    # Second legend: curve labels
    ax.legend(loc="lower left", ncol=1, fontsize=12, frameon=False)

    # Save plot
    if filename:
        if "." not in filename:
            filename = f"{filename}.pdf"
        fig.savefig(filename, bbox_inches="tight")

    plt.close()
