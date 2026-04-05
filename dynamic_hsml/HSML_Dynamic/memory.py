import numpy as np
from collections import Counter, defaultdict

class Memory(object):
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.x = None
        self.y = None
        self.task_id = None

    def add(self, images, labels):
        """Adds a batch of images and labels to memory."""
        if self.x is None:
            # First addition, allocate memory directly
            self.x = images
            self.y = labels
        else:
            # Subsequent additions, we need to concatenate
            self.x = np.concatenate((self.x, images), axis=0)
            self.y = np.concatenate((self.y, labels), axis=0)

        # Truncate to the most recent `memory_size` samples
        if self.x.shape[0] > self.memory_size:
            self.x = self.x[-self.memory_size:]
            self.y = self.y[-self.memory_size:]

    def is_empty(self):
        return self.x is None
    def size(self):
        return 0 if self.x is None else int(self.x.shape[0])


    def sample(self, n):
        """Randomly sample n items from memory."""
        if self.x is None or self.x.shape[0] == 0:
            return None, None
        n = min(n, self.x.shape[0])
        idx = np.random.choice(self.x.shape[0], size=n, replace=False)
        return self.x[idx], self.y[idx]


class TaskAwareMemory:
    """A memory system that holds a separate reservoir for each task."""
    def __init__(self, memory_size_per_task):
        self.memory_size_per_task = memory_size_per_task
        self.reservoirs = {}  # {task_id: ReservoirBuffer}
        self.task_count = 0

    def add_task(self):
        """Adds a new, empty reservoir for a new task."""
        self.task_count += 1
        new_task_id = self.task_count
        self.reservoirs[new_task_id] = Memory(memory_size=self.memory_size_per_task)
        print(f"Task-Aware Memory: Added buffer for new task ID: {new_task_id}")
        return new_task_id

    def add_samples(self, task_id, images, labels):
        """Adds samples to a specific task's reservoir."""
        if task_id not in self.reservoirs:
            print(f"Warning: Task ID {task_id} not found. Cannot add samples.")
            return
        self.reservoirs[task_id].add(images, labels)

    def sample_from_past(self, batch_size):
        """Samples a batch randomly from all past tasks combined."""
        if not self.reservoirs or self.task_count <= 1:
            return None, None # No past tasks to sample from

        past_task_ids = list(self.reservoirs.keys())
        if self.task_count in past_task_ids:
            past_task_ids.remove(self.task_count)
        
        if not past_task_ids:
            return None, None

        samples_per_task = max(1, batch_size // len(past_task_ids))
        
        all_x, all_y = [], []
        for task_id in past_task_ids:
            x, y = self.reservoirs[task_id].sample(samples_per_task)
            if x is not None:
                all_x.append(x)
                all_y.append(y)
        
        if not all_x:
            return None, None

        return np.concatenate(all_x), np.concatenate(all_y)
    def _sizes_by_task(self, exclude_task_id=None):
        """Return [(task_id, size)] and total size across (optionally excluding current)."""
        items, total = [], 0
        for tid, res in self.reservoirs.items():
            if exclude_task_id is not None and tid == exclude_task_id:
                continue
            sz = res.size()
            if sz > 0:
                items.append((tid, sz))
                total += sz
        return items, total

    def sample_uniform_across_tasks(self, n, exclude_task_id=None, return_task_ids=False):
        """
        Sample n items uniformly over ALL stored samples (i.e., per-sample fairness),
        optionally excluding the current task. Returns (X, Y) or (X, Y, TIDs)
        flattened over tasks. Shapes: [n, 784], [n], [n].
        """
        items, total = self._sizes_by_task(exclude_task_id)
        if total == 0:
            return (None, None, None) if return_task_ids else (None, None)

        # Build multinomial over tasks proportional to reservoir sizes (≈ per-sample uniform)
        task_ids = [tid for tid, _ in items]
        probs    = np.array([sz for _, sz in items], dtype=np.float64)
        probs    = probs / probs.sum()

        chosen_tids = np.random.choice(task_ids, size=n, replace=True, p=probs)

        Xs, Ys, TIDs = [], [], []
        # For efficiency, group asks per task, then sample that many from the task's reservoir
        from collections import Counter
        ask = Counter(chosen_tids)
        for tid, k in ask.items():
            xk, yk = self.reservoirs[tid].sample(k)  # should return [k,784], [k]
            if xk is None: 
                continue
            Xs.append(xk); Ys.append(yk); TIDs.append(np.full(len(yk), tid, dtype=np.int64))

        if not Xs:
            return (None, None, None) if return_task_ids else (None, None)

        X = np.concatenate(Xs, axis=0)
        Y = np.concatenate(Ys, axis=0)
        T = np.concatenate(TIDs, axis=0)

        # If exact n wasn’t met due to empty reservoirs/rounding, trim/pad
        if X.shape[0] > n:
            idx = np.random.choice(X.shape[0], size=n, replace=False)
            X, Y, T = X[idx], Y[idx], T[idx]
        elif X.shape[0] < n:
            # top up by random re-sampling (rare)
            need = n - X.shape[0]
            ridx = np.random.choice(X.shape[0], size=need, replace=True)
            X = np.concatenate([X, X[ridx]], axis=0)
            Y = np.concatenate([Y, Y[ridx]], axis=0)
            T = np.concatenate([T, T[ridx]], axis=0)

        return (X, Y, T) if return_task_ids else (X, Y)
# === NEW: Global FTML-style reservoir ===
class GlobalReservoir(object):
    """Reservoir buffer that enforces per-task minimum quotas."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.n_seen = 0
        self._x = []
        self._y = []
        self._t = []
        self._g = []  # global labels (for nonstationary buffer)
        self._d = []  # dataset ids (for nonstationary buffer)
        self.counts = Counter()
        self.min_quota = defaultdict(int)

    def __len__(self):
        return min(self.n_seen, self.capacity)

    def set_min_quota(self, task_ids):
        task_ids = [int(tid) for tid in task_ids]
        if not task_ids or self.capacity <= 0:
            self.min_quota.clear()
            return
        per_task = self.capacity // max(1, len(task_ids))
        for tid in task_ids:
            self.min_quota[tid] = per_task

    def _append_or_replace_one(self, xi, yi, ti, gi=None, di=None):
        ti = int(ti)
        if gi is None:
            gi = yi
        if di is None:
            di = ti
        if len(self._x) < self.capacity:
            self._x.append(np.asarray(xi))
            self._y.append(np.asarray(yi))
            self._t.append(ti)
            self._g.append(np.asarray(gi))
            self._d.append(int(di))
            self.counts[ti] += 1
            self.n_seen += 1
            return

        allowed_tasks = [tid for tid, cnt in self.counts.items()
                          if cnt > self.min_quota.get(tid, 0)]
        replace_idx = None
        attempts = 0
        max_attempts = 32
        while attempts < max_attempts:
            j = np.random.randint(0, self.n_seen)
            attempts += 1
            if j >= self.capacity:
                continue
            old_tid = self._t[j]
            if self.counts.get(old_tid, 0) > self.min_quota.get(old_tid, 0):
                replace_idx = j
                break

        if replace_idx is None and allowed_tasks:
            target_tid = max(allowed_tasks, key=lambda tid: self.counts[tid] - self.min_quota.get(tid, 0))
            for idx in range(self.capacity):
                if self._t[idx] == target_tid:
                    replace_idx = idx
                    break

        if replace_idx is None:
            self.n_seen += 1
            return

        old_tid = self._t[replace_idx]
        self.counts[old_tid] -= 1
        if self.counts[old_tid] <= 0:
            self.counts.pop(old_tid, None)

        self._x[replace_idx] = np.asarray(xi)
        self._y[replace_idx] = np.asarray(yi)
        self._t[replace_idx] = ti
        self._g[replace_idx] = np.asarray(gi)
        self._d[replace_idx] = int(di)
        self.counts[ti] += 1
        self.n_seen += 1

    def _arrays(self):
        if len(self._x) == 0:
            return None, None, None
        return (np.asarray(self._x), np.asarray(self._y), np.asarray(self._t))

    def add_samples(self, task_id, X, Y):
        X = np.asarray(X); Y = np.asarray(Y)
        assert X.shape[0] == Y.shape[0]
        for xi, yi in zip(X, Y):
            self._append_or_replace_one(xi, yi, task_id)

    def add_samples_with_dataset(self, task_id, dataset_id, X, Y_global):
        X = np.asarray(X); Y_global = np.asarray(Y_global)
        assert X.shape[0] == Y_global.shape[0]
        for xi, yi in zip(X, Y_global):
            self._append_or_replace_one(xi, yi, task_id, gi=yi, di=dataset_id)

    def tasks_present(self):
        return sorted(self.counts.keys())


    @property
    def total(self):
        return len(self._x)
    def counts_by_task(self):
        return dict(self.counts)

    def sample(self, k):
        X, Y, T = self._arrays()
        if X is None: return None, None, None
        k = min(int(k), len(X))
        idx = np.random.choice(len(X), size=k, replace=False)
        return X[idx], Y[idx], T[idx]

    def sample_task(self, tid, k):
        X, Y, T = self._arrays()
        if X is None: return None, None
        idx = np.where(T == int(tid))[0]
        if idx.size == 0:
            return None, None
        take = np.random.choice(idx, size=int(k), replace=(idx.size < k))
        return X[take], Y[take]

    def sample_dataset_class(self, dataset_id, class_id, k):
        X, Y, T = self._arrays()
        if X is None: return None, None
        d = np.asarray(self._d)
        g = np.asarray(self._g)
        idx = np.where((d == int(dataset_id)) & (g == int(class_id)))[0]
        if idx.size == 0:
            return None, None
        take = np.random.choice(idx, size=int(k), replace=(idx.size < k))
        return X[take], g[take]

    def sample_grouped(self, n_eps, K, Q, exclude_tid=None):
        X, Y, T = self._arrays()
        if X is None: return []
        counts = self.counts_by_task()
        tids = [tid for tid in counts.keys() if tid != exclude_tid]
        if len(tids) == 0:
            return []
        probs = np.asarray([counts[tid] for tid in tids], dtype=np.float64)
        probs = probs / probs.sum()
        out = []
        for _ in range(int(n_eps)):
            tid = int(np.random.choice(tids, p=probs))
            xs, ys = self.sample_task(tid, K)
            xq, yq = self.sample_task(tid, Q)
            if xs is None or xq is None:
                alt = [t for t in tids if t != tid]
                if not alt:
                    continue
                weights = np.asarray([counts[a] for a in alt], dtype=np.float64)
                weights = weights / weights.sum()
                tid2 = int(np.random.choice(alt, p=weights))
                xs, ys = self.sample_task(tid2, K)
                xq, yq = self.sample_task(tid2, Q)
                if xs is None or xq is None:
                    continue
                tid = tid2
            out.append((tid, xs, ys, xq, yq))
        return out
