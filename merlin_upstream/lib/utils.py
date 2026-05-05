import logging
import torch
from lib.config import cfg
import numpy as np

import matplotlib.pyplot as plt
# import seaborn as sns
import os


def log(message, print_to_console=True, log_level=logging.DEBUG):
    if log_level == logging.INFO:
        logging.info(message)
    elif log_level == logging.DEBUG:
        logging.debug(message)
    elif log_level == logging.WARNING:
        logging.warning(message)
    elif log_level == logging.ERROR:
        logging.error(message)
    elif log_level == logging.CRITICAL:
        logging.critical(message)
    else:
        logging.debug(message)

    if print_to_console:
        print(message)


def compute_accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        result.append(correct_k.mul_(100.0 / batch_size))
    return result


class Metrics:
    def __init__(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count


# one-hot the layer index
def one_hot(data, max_value):
    ones = torch.sparse.torch.eye(max_value)
    return ones.index_select(0, data)


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def lr_linear(epoch):
    lr = cfg.kernels.learning_rate * np.minimum((-epoch) * 1. / (cfg.kernels.epochs) + 1, 1.)
    return max(0, lr)


def tonp(x):
    return x.cpu().detach().numpy()


def plot(samples, filename='samples'):
    try:
        d = 7 if '7x7' in cfg.model else 3
        f, axs = plt.subplots(nrows=5, ncols=5, figsize=(12, 10))
        for x, ax in zip(samples.reshape((-1, d, d)), axs.flat):
            # sns.heatmap(tonp(x), ax=ax, cmap='Greens')
            ax.axis('off')
        f.savefig(os.path.join(cfg.output_dir, filename), dpi=200)
        plt.close(f)
    except Exception as error:
        log('Exception occurred while plotting. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def plot_reconstructions(data, samples, filename='reconstructions'):
    try:
        d = 7 if '7x7' in cfg.model else 3
        f, axs = plt.subplots(nrows=5, ncols=5, figsize=(15, 7))
        for x, x_rec, ax in zip(data.reshape((-1, d, d)), samples.reshape((-1, d, d)), axs.flat):
            # sns.heatmap(np.concatenate((tonp(x), tonp(x_rec)), 1), ax=ax, cmap='Greens')
            ax.axis('off')
        f.savefig(os.path.join(cfg.output_dir, filename), dpi=200)
        plt.close(f)
    except Exception as error:
        log('Exception occurred while plotting. Ignoring.', log_level=logging.ERROR)
        log(error, log_level=logging.ERROR)


def get_glove_embedding(data):
    vectors = torch.load('./glove_embeddings.pkl')
    return torch.FloatTensor(vectors[str(data)])


def compute_offset(task):
    offset1 = task * cfg.continual.n_class_per_task
    offset2 = (task + 1) * cfg.continual.n_class_per_task
    return int(offset1), int(offset2)


def compute_forgetting(accuracies):
    if len(accuracies) <= 1:
        return 0.0

    rows = [np.asarray(acc, dtype=np.float32).reshape(-1) for acc in accuracies]
    num_rows = len(rows)

    same_width = all(row.shape[0] == rows[0].shape[0] for row in rows)
    if same_width:
        acc_matrix = np.stack(rows, axis=0)
        num_tasks = acc_matrix.shape[1]
        if num_tasks <= 1:
            return 0.0

        forgetness = 0.0
        for task_idx in range(num_tasks - 1):
            learned_row = min(task_idx, num_rows - 1)
            best_after_learning = np.max(acc_matrix[learned_row:num_rows - 1, task_idx]) if learned_row < (num_rows - 1) else acc_matrix[-1, task_idx]
            forgetness += float(best_after_learning - acc_matrix[-1, task_idx])
        return forgetness / float(num_tasks - 1)

    acc_matrix = []
    num_tasks = len(rows)
    for index, acc in enumerate(rows):
        acc_matrix.append(np.pad(acc, [(0, num_tasks - (index + 1))], mode='constant', constant_values=101))

    acc_matrix = np.array(acc_matrix)
    forgetness = 0.0
    for t in range(num_tasks - 1):
        forgetness += float(np.max(acc_matrix[t:num_tasks - 1, t] - acc_matrix[-1, t]))
    return forgetness / float(num_tasks - 1)


def confusion_matrix(result_a, log_dir, fname='merlin_eval.txt', row_labels=None, col_labels=None, diag_row_indices=None):
    os.makedirs(log_dir, exist_ok=True)
    ncols = result_a.size(1)
    nrows = result_a.size(0) - 1
    baseline = result_a[0]
    post_rows = result_a[1:, :]

    if diag_row_indices is None:
        diag_row_indices = list(range(ncols))

    diag_vec = torch.tensor(
        [post_rows[diag_row_indices[col], col].item() for col in range(ncols)],
        dtype=post_rows.dtype
    )
    fin = post_rows[-1]
    bwt = fin - diag_vec

    fwt = torch.zeros(ncols, dtype=post_rows.dtype)
    for col, row_idx in enumerate(diag_row_indices):
        if row_idx > 0:
            fwt[col] = post_rows[row_idx - 1, col] - baseline[col]

    retained_acc = []
    learned_acc = [float(v) for v in diag_vec.tolist()]
    future_acc = []
    for t in range(nrows):
        upto = min(t, ncols)
        retained_acc.append(post_rows[t, :upto].mean().item() if upto > 0 else 0.0)
        start = min(t + 1, ncols)
        future_acc.append(post_rows[t, start:].mean().item() if start < ncols else 0.0)

    path = os.path.join(log_dir, fname)
    with open(path, 'w') as handle:
        print(' '.join(['%.4f' % r for r in baseline.tolist()]), file=handle)
        if row_labels is not None:
            print('Row labels: ' + ' '.join(str(int(r)) for r in row_labels), file=handle)
        if col_labels is not None:
            print('Col labels: ' + ' '.join(str(int(c)) for c in col_labels), file=handle)
        if diag_row_indices is not None:
            print('Diag rows: ' + ' '.join(str(int(d)) for d in diag_row_indices), file=handle)
        print('|', file=handle)
        for row in range(post_rows.size(0)):
            print(' '.join(['%.4f' % r for r in post_rows[row].tolist()]), file=handle)
        print('', file=handle)
        print('Diagonal Accuracy: %.4f' % diag_vec.mean().item(), file=handle)
        print('Final Accuracy: %.4f' % fin.mean().item(), file=handle)
        print('Backward: %.4f' % bwt.mean().item(), file=handle)
        print('Forward:  %.4f' % fwt.mean().item(), file=handle)
        print('mean retained: %.4f' % (sum(retained_acc) / max(1, len(retained_acc))), file=handle)
        print('mean learned: %.4f' % (sum(learned_acc) / max(1, len(learned_acc))), file=handle)
        print('mean future: %.4f' % (sum(future_acc) / max(1, len(future_acc))), file=handle)

    return (
        diag_vec.mean().item(),
        fin.mean().item(),
        bwt.mean().item(),
        fwt.mean().item(),
        sum(retained_acc) / max(1, len(retained_acc)),
        sum(learned_acc) / max(1, len(learned_acc)),
        sum(future_acc) / max(1, len(future_acc)),
        path,
    )
