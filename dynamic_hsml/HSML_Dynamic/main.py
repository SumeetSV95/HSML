import csv
import json
import pickle
import random

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()      # TF1-style graph
tf.disable_v2_behavior() 

tf.set_random_seed(0)
from data_generator import DataGenerator
from maml import MAML
from memory import GlobalReservoir, Memory, TaskAwareMemory
from tensorflow.python.platform import flags
import random
import os, csv, collections, time
import numpy as np

## Dataset/method options
flags.DEFINE_string('datasource', 'sinusoid', 'sinusoid or omniglot or miniimagenet or mixture or multidataset')
flags.DEFINE_integer('test_dataset', -1,
                     'which dataset to be test: 0: bird, 1: texture, 2: aircraft, 3: fungi, -1 is test all')
flags.DEFINE_integer('num_classes', 5, 'number of classes used in classification (e.g. 5-way classification).')
flags.DEFINE_integer('num_test_task', 1000, 'number of test tasks.')
flags.DEFINE_integer('test_epoch', -1, 'test epoch, only work when test start')
flags.DEFINE_string('nonstat_data_root', '', 'Root directory containing prepared nonstationary .pt files.')
flags.DEFINE_integer('nonstat_epochs_per_dataset', 50, 'Epochs to train on each dataset in nonstationary benchmark.')
flags.DEFINE_integer('nonstat_episodes_per_epoch', 32, 'Episodes sampled per epoch for nonstationary benchmark.')
flags.DEFINE_integer('nonstat_eval_tasks', 100, 'Evaluation episodes per dataset boundary in nonstationary setting.')
flags.DEFINE_integer('nonstat_eval_tasks_interval', 20, 'Evaluation episodes per dataset for interval eval in nonstationary setting.')
flags.DEFINE_integer('nonstat_n_way', 5, 'Way (number of classes) per episode for nonstationary mode.')
flags.DEFINE_integer('nonstat_k_shot', 5, 'Support shots per class for nonstationary mode.')
flags.DEFINE_integer('nonstat_q_query', 15, 'Query samples per class for nonstationary mode.')
flags.DEFINE_string('nonstat_test_split', 'test', 'Which split to use for evaluation episodes (train/test/val).')
flags.DEFINE_integer('nonstat_eval_interval', 100, 'How often (in iterations) to run test evaluation in nonstationary mode.')
flags.DEFINE_bool('nonstat_add_cluster_on_boundary', True, 'If True, add a cluster when moving to the next dataset stage.')
flags.DEFINE_string('stationary_dataset', 'vggflowers', 'Dataset key to isolate for stationary single-dataset mode.')
flags.DEFINE_integer('stationary_iterations', 10000, 'Training iterations for the stationary single-dataset setting.')
flags.DEFINE_integer('stationary_eval_interval', 500, 'How often to run evaluation during stationary mode training.')
flags.DEFINE_float('stationary_dropout', 0.1, 'Dropout probability for the stationary VERSA encoder blocks.')

## Training options
flags.DEFINE_integer('pretrain_iterations', 0, 'number of pre-training iterations.')
flags.DEFINE_integer('metatrain_iterations', 15000,
                     'number of metatraining iterations.')  # 15k for omniglot, 50k for sinusoid
flags.DEFINE_integer('meta_batch_size', 25, 'number of tasks sampled per meta-update')
flags.DEFINE_float('meta_lr', 0.001, 'the base learning rate of the generator')
flags.DEFINE_integer('update_batch_size', 5,
                     'number of examples used for inner gradient update (K for K-shot learning).')
flags.DEFINE_integer('update_batch_size_eval', 100,
                     'number of examples used for inner gradient test (K for K-shot learning).')
flags.DEFINE_float('update_lr', 0.001, 'step size alpha for inner gradient update.')  # 0.1 for omniglot
flags.DEFINE_integer('num_updates', 3, 'number of inner gradient updates during training.')
flags.DEFINE_integer('num_groups', 1, 'number of groups.')
flags.DEFINE_integer('fix_embedding_sample', -1,
                     'if the fix_embedding sample is -1, all samples are used for embedding. Otherwise, specific samples are used')

## Model options
flags.DEFINE_string('norm', 'batch_norm', 'batch_norm, layer_norm, or None')
flags.DEFINE_integer('hidden_dim', 40, 'output dimension of task embedding')
flags.DEFINE_integer('num_filters', 64, 'number of filters for conv nets -- 32 for miniimagenet, 64 for omiglot.')
flags.DEFINE_bool('conv', False, 'whether or not to use a convolutional network, only applicable in some cases')
flags.DEFINE_bool('max_pool', False, 'Whether or not to use max pooling rather than strided convolutions')
flags.DEFINE_bool('stop_grad', False, 'if True, do not use second derivatives in meta-optimization (for speed)')
flags.DEFINE_float('emb_loss_weight', 0.1, 'the weight of autoencoder')
flags.DEFINE_string('emb_type', 'sigmoid', 'sigmoid embedding')
flags.DEFINE_bool('no_val', False, 'if true, there are no validation set of Omniglot dataset')
flags.DEFINE_integer('tree_type', 1, 'select the tree type: 1 or 2')
flags.DEFINE_integer('task_embedding_num_filters', 32, 'number of filters for task embedding')
flags.DEFINE_string('task_embedding_type', 'rnn', 'rnn or mean')

## clustering information
flags.DEFINE_integer('cluster_layer_0', 4, 'number of clusters in the first layer')
flags.DEFINE_integer('cluster_layer_1', 2, 'number of clusters in the second layer')

flags.DEFINE_integer('seed', 0, 'Random seed for numpy/random/tensorflow RNGs.')

## Online version
flags.DEFINE_bool('online_training', False, 'whether to train online or not')
flags.DEFINE_float('online_threshold', 1.5, 'the threshold of each method')
flags.DEFINE_integer('online_change', 8000, 'keep stable training')


## Logging, saving, and testing options
flags.DEFINE_bool('log', True, 'if false, do not log summaries, for debugging code.')
flags.DEFINE_string('logdir', '/tmp/data', 'directory for summaries and checkpoints.')
flags.DEFINE_string('datadir', '/home/huaxiuyao/Data/', 'directory for datasets.')
flags.DEFINE_bool('resume', True, 'resume training if there is a model available')
flags.DEFINE_bool('train', True, 'True to train, False to test.')
flags.DEFINE_bool('test_set', False, 'Set to true to test on the the test set, False for the validation set.')
flags.DEFINE_integer('train_update_batch_size', -1,
                     'number of examples used for gradient update during training (use if you want to test with a different number).')
flags.DEFINE_float('train_update_lr', -1,
                   'value of inner gradient step step during training. (use if you want to test with a different value)')  # 0.1 for omniglot
flags.DEFINE_integer('memory_size', 2000, 'The total size of the reservoir memory buffer.')
flags.DEFINE_integer('train_iters', 800000 // 10, 'Number of training iterations for autonomous continual learning.')
flags.DEFINE_integer('print_interval', 100, 'Interval for printing training stats.')
flags.DEFINE_integer('validate_every', 0,
    'Run validation every N iters; 0 disables during training.')
flags.DEFINE_bool('defer_eval', True,
    'Skip validation during training and run one pass at the end.')
# main.py (flag definitions)
flags.DEFINE_float("current_query_frac",0.5,
               "Target fraction of queries that come from the current task (approx; exact if meta_batch even).")
flags.DEFINE_bool("replay_uniform", True,
               "If set, draw replay per-sample uniformly across all past samples, then group per task.")
flags.DEFINE_bool('recurring_rot_mnist', False,
               'Enable recurring RotMNIST setting (split even tasks into two halves).')
flags.DEFINE_bool('recurring_perm_mnist', False,
               'Enable recurring PermMNIST setting (split even tasks into two halves).')

# Debug/analysis flags
flags.DEFINE_bool('debug_sampling', True, 'Print which tasks & counts populate each meta-batch (throttled).')
flags.DEFINE_string('events_csv', 'online_events.csv', 'Filename for per-iter loss-window and event logs.')
flags.DEFINE_integer('debug_every', 50, 'How often to print debug sampling breakdown (iters).')
flags.DEFINE_integer(
    'current_stream_eps', -1,
    'How many current-task episodes per iteration should use fresh stream data '
    '(-1 = all current episodes; 0 = none; otherwise a positive integer).'
)




FLAGS = flags.FLAGS

"""def get_task_id_from_stream_idx(stream_idx, task_sizes):
    cum_sum = 0
    for i, size in enumerate(task_sizes):
        cum_sum += size
        if stream_idx < cum_sum:
            return i + 1  # 1-based task id
    return len(task_sizes)"""

class HSML_Online():
    def __init__(self):
        if FLAGS.datasource in ['multidataset', 'mixture']:
            self.NUM_TEST_POINTS = FLAGS.num_test_task
        else:
            self.NUM_TEST_POINTS = 600
        self.clusters = FLAGS.cluster_layer_0

    def train(self, model, saver, exp_string, data_generator, resume_itr=0):
        param_dict = {}
        SUMMARY_INTERVAL = 100
        SAVE_INTERVAL = 1000
        if FLAGS.datasource in ['sinusoid', 'mixture']:
            PRINT_INTERVAL = 1000
            TEST_PRINT_INTERVAL = PRINT_INTERVAL * 5
        else:
            PRINT_INTERVAL = 100
            TEST_PRINT_INTERVAL = PRINT_INTERVAL * 10

        if FLAGS.log:
            train_writer = tf.summary.FileWriter(FLAGS.logdir + '/' + exp_string, self.sess.graph)
        print('Done initializing, starting training.')

        # ==== Rotated / Permuted MNIST ====
        if FLAGS.datasource in ['mnist_rotations', 'mnist_permutations']:
            prelosses, postlosses, embedlosses = [], [], []
            last_res = 1.0
            change_itr = -10**9
            print("--- HSML-D (corrected): current + replay episodes; configurable fresh-stream episodes ---")
            
            import collections

            # ---- Validation helpers (used at END only) ----
            val_per_task = None
            if hasattr(data_generator, 'dataset_val'):
                val_per_task = []
                for i in range(data_generator.num_total_tasks):
                    vx, vy = data_generator.dataset_val[i]
                    val_per_task.append((np.array(vx), np.array(vy)))

            def get_gt_task(stream_idx, task_sizes):
                cum = 0
                for i, sz in enumerate(task_sizes):
                    cum += sz
                    if stream_idx < cum:
                        return i + 1
                return len(task_sizes)

            def get_task_id_from_stream_idx(stream_idx, task_sizes):
                return get_gt_task(stream_idx, task_sizes)

            def build_val_batch_for_task(task_1based, K_support, K_query):
                vx, vy = val_per_task[task_1based - 1]
                n = len(vx)
                idx = np.random.choice(n, size=K_support + K_query, replace=False)
                sx, sy = vx[idx[:K_support]], vy[idx[:K_support]]
                qx, qy = vx[idx[K_support:K_support + K_query]], vy[idx[K_support:K_support + K_query]]
                return sx, sy, qx, qy

            def _pad_to_meta_batch(xa, ya, xb, yb, mbs):
                """Pad single-task tensors to meta-batch size so placeholders always match."""
                if xa.shape[0] == mbs:
                    return xa, ya, xb, yb
                rep = (mbs, 1, 1); rep_y = (mbs, 1)
                return (np.tile(xa, rep), np.tile(ya, rep_y),
                        np.tile(xb, rep), np.tile(yb, rep_y))

            def eval_once_unseen(model, sess, sx, sy, qx, qy):
                B_SUP   = FLAGS.update_batch_size
                B_QUERY = FLAGS.update_batch_size_eval
                mbs = FLAGS.meta_batch_size
                inputa = sx.reshape([1, B_SUP, -1])
                labela = sy.reshape([1, B_SUP])
                inputb = qx.reshape([1, B_QUERY, -1])
                labelb = qy.reshape([1, B_QUERY])
                inputa, labela, inputb, labelb = _pad_to_meta_batch(inputa, labela, inputb, labelb, mbs)
                feed = {model.inputa: inputa, model.labela: labela,
                        model.inputb: inputb, model.labelb: labelb,
                        model.meta_lr: 0.0}
                pre_acc, post_acc = self.sess.run(
                    [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]], feed)
                try:
                    return float(np.mean(pre_acc)), float(np.mean(post_acc))
                except Exception:
                    return float(pre_acc), float(post_acc)
            logical_task_ids = getattr(data_generator, 'logical_task_ids', [])
            if not logical_task_ids:
                nt = getattr(data_generator, 'num_total_tasks', 0)
                logical_task_ids = list(range(nt if nt else len(getattr(data_generator, 'dataset_test', []))))
            recurring_to_orig = getattr(data_generator, 'recurring_to_orig', {})

            test_per_task = []
            if hasattr(data_generator, "dataset_test") and len(data_generator.dataset_test):
                for tx, ty in data_generator.dataset_test:
                    test_per_task.append((np.asarray(tx), np.asarray(ty)))
            if test_per_task and len(test_per_task) != len(logical_task_ids):
                print(f"[WARN] test_per_task count ({len(test_per_task)}) does not match logical task ids ({len(logical_task_ids)}).")
            eval_rows = []
            diag_history = []
            retained_history = []
            eval_seed_base = getattr(FLAGS, "data_seed", getattr(FLAGS, "seed", 0))
            B_SUP = FLAGS.update_batch_size
            B_QUERY = FLAGS.update_batch_size_eval

            def evaluate_on_test(completed_task_idx, global_step):
                if len(test_per_task) == 0:
                    return
                rng = np.random.RandomState(eval_seed_base + completed_task_idx * 1009)
                per_task = []
                for tx, ty in test_per_task:
                    n = len(tx)
                    if n < B_SUP + B_QUERY:
                        continue
                    choice = rng.choice(n, size=B_SUP + B_QUERY, replace=False)
                    sx, sy = tx[choice[:B_SUP]], ty[choice[:B_SUP]]
                    qx, qy = tx[choice[B_SUP:B_SUP + B_QUERY]], ty[choice[B_SUP:B_SUP + B_QUERY]]
                    pre, post = eval_once_unseen(model, self.sess, sx, sy, qx, qy)
                    per_task.append((float(pre), float(post)))
                if not per_task:
                    return
                pre_arr  = np.array([p for p, _ in per_task], dtype=np.float32)
                post_arr = np.array([q for _, q in per_task], dtype=np.float32)
                eval_rows.append(post_arr)
                diag_idx = completed_task_idx - 1
                logical_tid = logical_task_ids[diag_idx] if 0 <= diag_idx < len(logical_task_ids) else None
                base_tid = recurring_to_orig.get(logical_tid, logical_tid) if logical_tid is not None else None
                diag     = post_arr[diag_idx] if 0 <= diag_idx < len(post_arr) else float('nan')
                retained = post_arr[:diag_idx].mean() if diag_idx > 0 else 0.0
                future   = post_arr[diag_idx+1:].mean() if diag_idx + 1 < len(post_arr) else 0.0
                diag_history.append(diag)
                retained_history.append(retained)
                print(f"[EVAL] after task {completed_task_idx:02d}/{len(post_arr):02d} "
                    f"pre_mean={pre_arr.mean():.4f} post_mean={post_arr.mean():.4f} "
                    f"diag={diag:.4f} retained={retained:.4f} future={future:.4f} "
                    f"logical_tid={logical_tid} base_tid={base_tid}")
                if FLAGS.log:
                    train_writer.add_summary(tf.Summary(value=[
                        tf.Summary.Value(tag="eval_after_task/pre_mean",  simple_value=float(pre_arr.mean())),
                        tf.Summary.Value(tag="eval_after_task/post_mean", simple_value=float(post_arr.mean())),
                    ]), global_step)

            # ---- Task-aware rehearsal memory (old strategy; keep for fallback) ----
            """
            # Task-aware per-task buffer
            memory = TaskAwareMemory(memory_size_per_task=FLAGS.memory_size)
            current_task_id = memory.add_task()
            """

            # ---- Global rehearsal memory (current strategy) ----
            memory = GlobalReservoir(capacity=FLAGS.memory_size)  # e.g., 30000
            current_task_id = 1          # logical HSML task id
            seen_task_count = 1          # for logs only
            print(f"Global buffer active (capacity={FLAGS.memory_size}). Starting with logical task {current_task_id}.")

            # ---- Stream (no shuffle) ----
            all_images, all_labels = data_generator.get_continuous_stream()
            stream_idx, N = 0, len(all_images)

            # For nice logs about GT stream switches
            task_sizes = []
            if hasattr(data_generator, 'num_total_tasks') and hasattr(data_generator, 'dataset_train'):
                for i in range(data_generator.num_total_tasks):
                    task_sizes.append(len(data_generator.dataset_train[i][0]))  # RotMNIST
            current_task_id_in_stream = get_task_id_from_stream_idx(stream_idx, task_sizes)
            if task_sizes:
                preview = []
                for idx in range(min(5, len(task_sizes))):
                    logical_tid = logical_task_ids[idx] if idx < len(logical_task_ids) else idx
                    base_tid = recurring_to_orig.get(logical_tid, logical_tid)
                    preview.append((idx + 1, logical_tid, base_tid, task_sizes[idx]))
                print(f"GT task size preview (seq→logical→base→count): {preview}")
            logical_tid0 = logical_task_ids[current_task_id_in_stream - 1] if 0 < current_task_id_in_stream <= len(logical_task_ids) else None
            base_tid0 = recurring_to_orig.get(logical_tid0, logical_tid0) if logical_tid0 is not None else None
            print(f"Starting stream at GT Task {current_task_id_in_stream} (logical={logical_tid0}, base={base_tid0}) stream_idx={stream_idx}")

            # --- CSV: logdir/exp_string/events.csv ---
            events_dir = os.path.join(FLAGS.logdir, exp_string)
            os.makedirs(events_dir, exist_ok=True)
            events_path = os.path.join(events_dir, FLAGS.events_csv)
            evt_fh = open(events_path, 'w', newline='')
            evt_wr = csv.writer(evt_fh)
            evt_wr.writerow([
                'iter','event','old_avg','new_avg','ratio','threshold',
                'clusters','detected_task_count','gt_stream_task','current_task_id','stream_idx'
            ])

            # ---- windows / sizes ----
            B_SUP   = FLAGS.update_batch_size
            B_METAQ = FLAGS.update_batch_size_eval
            mbs     = FLAGS.meta_batch_size
            K, Q    = B_SUP, B_METAQ

            # ---- HSML-simple counters ----
            WINDOW   = int(FLAGS.online_change)       # iters per window
            MU  = float(FLAGS.online_threshold)       # ratio trigger
            q_loss_sum = 0.0
            q_iters = 0
            last_avg = None

            # current-task only collections (optional archives)
            q_buf_x, q_buf_y = [], []
            curr_task_archive_x, curr_task_archive_y = [], []

            # consumption tracker (debug)
            total_stream_consumed = 0

            # util: read a block and advance a pointer
            def _next_block(start, size):
                end = min(start + size, N)
                return (all_images[start:end], all_labels[start:end]), end

            def _sample_current_from_global(K, Q, fallback):
                """Try to draw K+Q for the *current task* from the global buffer; else fallback."""
                fb_sx, fb_sy, fb_qx, fb_qy = fallback
                xs, ys = memory.sample_task(current_task_id, K)
                xq, yq = memory.sample_task(current_task_id, Q)
                if xs is None or xq is None:
                    return np.array(fb_sx), np.array(fb_sy), np.array(fb_qx), np.array(fb_qy)
                return xs, ys, xq, yq

            def _sample_replay_from_global(n_replay_eps, K, Q):
                """Build replay episodes from the global buffer, excluding current task when possible."""
                eps = memory.sample_grouped(n_replay_eps, K, Q, exclude_tid=current_task_id)
                if len(eps) == 0:
                    # as a last resort, allow current task too
                    eps = memory.sample_grouped(n_replay_eps, K, Q, exclude_tid=None)
                return eps

            # util: add a cluster & rebuild graph (unchanged)
            param_dict = {}
            def _add_cluster_and_rebuild(iter_idx):
                nonlocal model, saver, data_generator
                print(f"[HSML] Triggered → adding a cluster.")
                tvars = tf.trainable_variables()
                tvals = self.sess.run(tvars)
                param_dict = {v.name: val for v, val in zip(tvars, tvals)}
                # Close old session before resetting graph
                self.sess.close()
                tf.reset_default_graph()
                self.clusters += 1
                model, saver, data_generator = self.construct_model()
                tf.train.start_queue_runners(self.sess)
                tf.global_variables_initializer().run()
                for v in tf.trainable_variables():
                    if v.name in param_dict:
                        self.sess.run(v.assign(param_dict[v.name]))
                print(f"[HSML] Cluster added at iter {iter_idx}. Total clusters = {self.clusters}")

            # helper: pack episodes to [mbs, K/Q, 784] numpy arrays
            def _pack_episodes(eps):
                inputa = np.zeros((len(eps), K, 784), np.float32)
                labela = np.zeros((len(eps), K),      np.int32)
                inputb = np.zeros((len(eps), Q, 784), np.float32)
                labelb = np.zeros((len(eps), Q),      np.int32)
                for i, (_tid, xs, ys, xq, yq) in enumerate(eps):
                    inputa[i] = np.asarray(xs, np.float32)
                    labela[i] = np.asarray(ys, np.int32)
                    inputb[i] = np.asarray(xq, np.float32)
                    labelb[i] = np.asarray(yq, np.int32)
                return inputa, labela, inputb, labelb

            # ---- main loop ----
            for itr in range(resume_itr, FLAGS.train_iters):
                cur_tid = current_task_id

                # =========================================================
                # Build meta-batch: CURRENT (fresh+mem) + REPLAY
                # =========================================================
                n_cur_eps = max(1, int(round(FLAGS.current_query_frac * mbs)))
                # -1 → all current episodes are fresh; 0 → none fresh; k>0 → cap at k
                n_fresh = n_cur_eps if getattr(FLAGS, 'current_stream_eps', -1) < 0 \
                                    else min(n_cur_eps, max(0, FLAGS.current_stream_eps))

                episodes = []
                fresh_blocks = []   # keep fresh (sx,sy,qx,qy) for archiving/seeding
                ptr = stream_idx    # local cursor over the stream
                made_fresh = 0

                # ---- A) CURRENT task episodes: FRESH from stream ----
                for _ in range(n_fresh):
                    (sx, sy), ptr = _next_block(ptr, K)
                    if len(sx) < K: break
                    (qx, qy), ptr = _next_block(ptr, Q)
                    if len(qx) < Q: break
                    episodes.append((cur_tid, sx, sy, qx, qy))
                    fresh_blocks.append((sx, sy, qx, qy))
                    made_fresh += 1

                if n_fresh > 0 and made_fresh == 0:
                    # ran out of stream; finish gracefully
                    print("[WARN] Stream exhausted while building fresh episodes.")
                    break

                # If we made at least one fresh ep, we can use its data as fallback
                fallback_sx = fresh_blocks[0][0] if len(fresh_blocks) else None
                fallback_sy = fresh_blocks[0][1] if len(fresh_blocks) else None
                fallback_qx = fresh_blocks[0][2] if len(fresh_blocks) else None
                fallback_qy = fresh_blocks[0][3] if len(fresh_blocks) else None

                # Ensure at least one fresh block exists when the global buffer has nothing usable
                need_warm = (made_fresh == 0)
                if need_warm:
                    xs_probe, ys_probe = memory.sample_task(current_task_id, 1)
                    if xs_probe is not None:
                        need_warm = False
                if need_warm:
                    (sx, sy), ptr = _next_block(ptr, K)
                    (qx, qy), ptr = _next_block(ptr, Q)
                    if len(sx) < K or len(qx) < Q:
                        print("[WARN] Stream exhausted during warm-start.")
                        break
                    episodes.append((cur_tid, sx, sy, qx, qy))
                    fresh_blocks.append((sx, sy, qx, qy))
                    made_fresh += 1
                    fallback_sx, fallback_sy, fallback_qx, fallback_qy = sx, sy, qx, qy

                # ---- A2) CURRENT task episodes: from MEMORY (or fallback) ----
                for _ in range(n_cur_eps - made_fresh):
                    # Old task-aware memory (fallback reference)
                    """
                    xs, ys, xq, yq = _safe_sample_current()
                    episodes.append((cur_tid, xs, ys, xq, yq))
                    """
                    # Current: global memory (per-task sample) with fallback to first fresh block
                    xs, ys, xq, yq = _sample_current_from_global(K, Q,
                        (fallback_sx, fallback_sy, fallback_qx, fallback_qy))
                    episodes.append((cur_tid, xs, ys, xq, yq))

                # ---- B) REPLAY episodes ----
                n_replay_eps = max(0, mbs - len(episodes))
                if n_replay_eps > 0:
                    replay_eps = _sample_replay_from_global(n_replay_eps, K, Q)
                    if len(replay_eps) == 0:
                        # no replay yet → duplicate a current episode to fill shapes
                        while len(episodes) < mbs and len(episodes) > 0:
                            episodes.append(episodes[-1])
                    else:
                        episodes.extend(replay_eps)
                    """
                    if FLAGS.replay_uniform:
                        # per-sample uniform replay over past tasks; then group by task
                        ...
                    else:
                        # replay-by-task: pick tasks uniformly, each supplies a full episode
                        ...
                    """

                # Safety: exact meta-batch size
                while len(episodes) < mbs and len(episodes) > 0:
                    episodes.append(episodes[-1])
                episodes = episodes[:mbs]

                # ---- DEBUG: breakdown & stream consumption ----
                if getattr(FLAGS, 'debug_sampling', True) and (itr % max(1, getattr(FLAGS, 'debug_every', 50)) == 0):
                    ep_task_ids = [tid for (tid, *_rest) in episodes]
                    ep_counts = collections.Counter(ep_task_ids)
                    cur_eps = ep_counts.get(cur_tid, 0)
                    rep_eps = len(episodes) - cur_eps
                    consumed_this_iter = made_fresh * (K + Q)
                    total_stream_consumed += consumed_this_iter
                    print(f"[DBG] iter={itr} | meta-batch={len(episodes)} "
                        f"| current_eps={cur_eps} (fresh={made_fresh}, mem={cur_eps - made_fresh}) "
                        f"| replay_eps={rep_eps} | consumed_stream={consumed_this_iter} "
                        f"(total={total_stream_consumed}, stream_idx→{ptr})")
                    for tid, cnt in ep_counts.most_common():
                        tag = "CUR" if tid == cur_tid else "REPLAY"
                        print(f"       - task {tid:>2}  episodes={cnt}  [{tag}]")

                # ---- pack into feed tensors ----
                inputa, labela, metaq_inputb, metaq_labelb = _pack_episodes(episodes)

                # ----- 1) meta-train (support, meta-query) -----
                feed_train = {model.inputa: inputa, model.labela: labela,
                            model.inputb: metaq_inputb, model.labelb: metaq_labelb}
                train_fetches = [
                    model.metatrain_op,
                    model.total_embed_loss,
                    model.total_losses2[FLAGS.num_updates - 1],
                    model.total_accuracy1,
                    model.total_accuracies2[FLAGS.num_updates - 1],
                ]
                _, emb_loss, post_loss, pre_acc, post_acc = self.sess.run(train_fetches, feed_train)

                # ----- archive ONLY current-task FRESH samples -----
                for (sx, sy, qx, qy) in fresh_blocks:
                    """
                    # If you later revert to task-aware or archiving strategy, re-enable this:
                    curr_task_archive_x.extend([np.array(sx), np.array(qx)])
                    curr_task_archive_y.extend([np.array(sy), np.array(qy)])
                    q_buf_x.extend([np.array(sx), np.array(qx)])
                    q_buf_y.extend([np.array(sy), np.array(qy)])
                    """
                    memory.add_samples(current_task_id,
                        np.concatenate([sx, qx], axis=0),
                        np.concatenate([sy, qy], axis=0))

                # ----- 2) HSML-simple: accumulate loss & check every WINDOW iters -----
                q_loss_sum += float(post_loss)
                q_iters += 1

                if FLAGS.online_training and q_iters >= WINDOW:
                    new_avg = q_loss_sum / q_iters
                    if last_avg is not None:
                        ratio = new_avg / (last_avg + 1e-8)
                        evt_wr.writerow([
                            itr, 'loss_window_end', last_avg, new_avg, ratio, MU,
                            self.clusters, seen_task_count, current_task_id_in_stream, current_task_id, ptr
                        ])
                        if itr % (10 * max(1, getattr(FLAGS, 'debug_every', 50))) == 0:
                            evt_fh.flush()

                        print(f"[HSML] window end → old_avg={last_avg:.5f}, new_avg={new_avg:.5f}, "
                            f"ratio={ratio:.4f}, threshold={MU:.4f}")

                        if new_avg > MU * last_avg:
                            # (a) finalize OLD task: optional if already adding fresh each iter
                            """
                            if len(curr_task_archive_x) > 0:
                                old_x = np.concatenate(curr_task_archive_x, axis=0)
                                old_y = np.concatenate(curr_task_archive_y, axis=0)
                                memory.add_samples(current_task_id, old_x, old_y)
                            """
                            # (b) grow clusters (HSML)
                            _add_cluster_and_rebuild(itr)
                            evt_wr.writerow([
                                itr, 'cluster_add', last_avg, new_avg, ratio, MU,
                                self.clusters, seen_task_count, current_task_id_in_stream, current_task_id, ptr
                            ])
                            evt_fh.flush()

                            # (c) start NEW task and optionally seed with q-buffer
                            current_task_id += 1
                            seen_task_count += 1
                            if len(q_buf_x) > 0:
                                new_x = np.concatenate(q_buf_x, axis=0)
                                new_y = np.concatenate(q_buf_y, axis=0)
                                memory.add_samples(current_task_id, new_x, new_y)
                            print(f"--- TASK CHANGE CONFIRMED ---")
                            print(f"New task {current_task_id} detected. Resuming training.\n")

                            # reset archives for the NEW task
                            curr_task_archive_x, curr_task_archive_y = [], []

                    # slide the HSML window
                    last_avg = new_avg
                    q_loss_sum = 0.0
                    q_iters = 0
                    q_buf_x, q_buf_y = [], []

                # ----- 3) advance stream pointer to where we actually consumed -----
                stream_idx = ptr
                if stream_idx >= N:
                    # flush the last task into memory (optional if using only fresh_blocks adds)
                    if len(curr_task_archive_x) > 0:
                        old_x = np.concatenate(curr_task_archive_x, axis=0)
                        old_y = np.concatenate(curr_task_archive_y, axis=0)
                        memory.add_samples(current_task_id, old_x, old_y)
                    print("Stream exhausted."); break

                # ----- Logs (use META-QUERY post acc; no held-out eval) -----
                if itr % PRINT_INTERVAL == 0:
                    try:
                        pre_to_log  = float(np.mean(pre_acc))
                        post_to_log = float(np.mean(post_acc))
                    except Exception:
                        pre_to_log, post_to_log = float(pre_acc), float(post_acc)
                    print(f"Iter: {itr}, Detected Tasks: {seen_task_count}, "
                        f"Post-Acc(METAQ): {post_to_log:.4f}, Emb-Loss: {float(emb_loss):.4f}, "
                        f"clusters={self.clusters}, stream_idx={stream_idx}")

                # ---- GT task switch tracking (for debug / CSV) ----
                prev_task_id = current_task_id_in_stream
                current_task_id_in_stream = get_task_id_from_stream_idx(stream_idx, task_sizes)
                if current_task_id_in_stream != prev_task_id:
                    logical_tid = logical_task_ids[current_task_id_in_stream - 1] if 0 < current_task_id_in_stream <= len(logical_task_ids) else None
                    base_tid = recurring_to_orig.get(logical_tid, logical_tid) if logical_tid is not None else None
                    print(f"*** Datastream moved to GT Task {current_task_id_in_stream} (logical={logical_tid}, base={base_tid}) at stream_idx {stream_idx} ***")
                    evt_wr.writerow([
                        itr, 'gt_task_change', '', '', '', '',
                        self.clusters, seen_task_count, current_task_id_in_stream, current_task_id, stream_idx
                    ])
                    evaluate_on_test(prev_task_id, itr)
                    if current_task_id_in_stream > current_task_id:
                        current_task_id = current_task_id_in_stream
                        seen_task_count = current_task_id_in_stream
                    else:
                        current_task_id = max(current_task_id, current_task_id_in_stream)
                        seen_task_count = max(seen_task_count, current_task_id)



            # Post-training eval (cheap and optional)
            if FLAGS.defer_eval:
                print("\n[FINAL EVAL] Starting deferred validation across seen tasks...")

                def _pad_to_meta_batch(xa, ya, xb, yb, mbs):
                    if xa.shape[0] == mbs: return xa, ya, xb, yb
                    rep, rep_y = (mbs, 1, 1), (mbs, 1)
                    return (np.tile(xa, rep), np.tile(ya, rep_y),
                            np.tile(xb, rep), np.tile(yb, rep_y))

                def eval_once_unseen(model, sess, sx, sy, qx, qy):
                    B_SUP, B_QUERY, mbs = FLAGS.update_batch_size, FLAGS.update_batch_size_eval, FLAGS.meta_batch_size
                    inputa = sx.reshape([1, B_SUP, -1]); labela = sy.reshape([1, B_SUP])
                    inputb = qx.reshape([1, B_QUERY, -1]); labelb = qy.reshape([1, B_QUERY])
                    inputa, labela, inputb, labelb = _pad_to_meta_batch(inputa, labela, inputb, labelb, mbs)
                    feed = {model.inputa: inputa, model.labela: labela,
                            model.inputb: inputb, model.labelb: labelb, model.meta_lr: 0.0}
                    pre_acc, post_acc = self.sess.run(
                        [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]], feed)
                    try: return float(np.mean(pre_acc)), float(np.mean(post_acc))
                    except Exception: return float(pre_acc), float(post_acc)

                B_SUP, B_QUERY = FLAGS.update_batch_size, FLAGS.update_batch_size_eval
                per_task = []
                if hasattr(data_generator, 'dataset_val'):
                    for t in range(data_generator.num_total_tasks):
                        vx, vy = data_generator.dataset_val[t]
                        n = len(vx); idx = np.random.choice(n, size=min(n, B_SUP + B_QUERY), replace=False)
                        sx, sy = vx[idx[:B_SUP]], vy[idx[:B_SUP]]
                        qx, qy = vx[idx[B_SUP:B_SUP + B_QUERY]], vy[idx[B_SUP:B_SUP + B_QUERY]]
                        pre, post = eval_once_unseen(model, self.sess, sx, sy, qx, qy)
                        per_task.append((pre, post))
                else:
                    all_x, all_y = data_generator.get_continuous_stream()
                    start = 0
                    for size in [len(data_generator.dataset_train[i][0]) for i in range(data_generator.num_total_tasks)]:
                        mid = start + size // 2
                        xs = np.concatenate([all_x, all_x, all_x])
                        ys = np.concatenate([all_y, all_y, all_y])
                        st = (mid + 500) % len(all_x)
                        window_x, window_y = xs[st:st+B_SUP+B_QUERY], ys[st:st+B_SUP+B_QUERY]
                        sx, sy = window_x[:B_SUP], window_y[:B_SUP]
                        qx, qy = window_x[B_SUP:B_SUP+B_QUERY], window_y[B_SUP:B_SUP+B_QUERY]
                        pre, post = eval_once_unseen(model, self.sess, sx, sy, qx, qy)
                        per_task.append((pre, post))
                        start += size

                pre_mean  = float(np.mean([p for p, _ in per_task]))
                post_mean = float(np.mean([q for _, q in per_task]))
                print(f"[FINAL EVAL] mean pre/post acc across {len(per_task)} tasks: {pre_mean:.3f} / {post_mean:.3f}")
                if FLAGS.log:
                    summ = tf.Summary(value=[
                        tf.Summary.Value(tag="final_eval/pre_acc_mean",  simple_value=pre_mean),
                        tf.Summary.Value(tag="final_eval/post_acc_mean", simple_value=post_mean),
                    ])
                    train_writer.add_summary(summ, itr if 'itr' in locals() else 0)
                if diag_history:
                    final_post = np.array([post for _, post in per_task], dtype=np.float32)
                    diag_vec = np.array(diag_history, dtype=np.float32)
                    ra_mean = float(np.mean(retained_history)) if retained_history else float('nan')
                    la_mean = float(np.mean(diag_vec))
                    if final_post.size >= diag_vec.size:
                        bti_vals = final_post[:diag_vec.size] - diag_vec
                        bti_mean = float(np.mean(bti_vals))
                    else:
                        bti_mean = float('nan')
                    recurring_mean = float(np.mean(final_post)) if final_post.size else float('nan')
                    print(f"[METRICS] RA_mean={ra_mean:.4f} LA_mean={la_mean:.4f} "
                          f"BTI_mean={bti_mean:.4f} Recurring_mean={recurring_mean:.4f}")
                    if FLAGS.log:
                        metrics_summ = tf.Summary(value=[
                            tf.Summary.Value(tag="metrics/RA_mean",  simple_value=ra_mean),
                            tf.Summary.Value(tag="metrics/LA_mean",  simple_value=la_mean),
                            tf.Summary.Value(tag="metrics/BTI_mean", simple_value=bti_mean),
                            tf.Summary.Value(tag="metrics/Recurring_mean", simple_value=recurring_mean),
                        ])
                        train_writer.add_summary(metrics_summ, itr if 'itr' in locals() else 0)
            evt_fh.close()

        elif FLAGS.datasource == 'nonstationary':
            if not hasattr(data_generator, 'nonstat_order'):
                raise ValueError('Data generator missing nonstationary registry; check --nonstat_data_root.')

            n_way = data_generator.n_way
            k_shot = data_generator.k_shot
            q_query = data_generator.q_query
            meta_batch = FLAGS.meta_batch_size
            K = FLAGS.update_batch_size
            Q = FLAGS.update_batch_size_eval

            # Treat nonstat_epochs_per_dataset as the number of iterations (meta-updates) per dataset.
            iterations_per_dataset = FLAGS.nonstat_epochs_per_dataset
            episodes_per_epoch = FLAGS.nonstat_episodes_per_epoch
            eval_tasks = FLAGS.nonstat_eval_tasks
            eval_tasks_interval = int(getattr(FLAGS, 'nonstat_eval_tasks_interval', 20))

            metrics_dir = os.path.join(FLAGS.logdir, exp_string)
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, 'nonstationary_metrics.json')
            metrics_records = []
            eval_interval = max(1, int(getattr(FLAGS, 'nonstat_eval_interval', 100)))
            acc_matrix = []
            diag_means = []

            use_online = bool(FLAGS.online_training)
            WINDOW = max(1, int(FLAGS.online_change)) if use_online else 1
            MU = float(FLAGS.online_threshold) if use_online else 0.0
            q_loss_sum = 0.0
            q_iters = 0
            last_avg = None

            def _add_cluster_and_rebuild(iter_idx):
                nonlocal model, saver, data_generator, train_fetches
                nonlocal n_way, k_shot, q_query, K, Q
                print("[HSML] Triggered → adding a cluster at iter {}.".format(iter_idx))
                tvars = tf.trainable_variables()
                tvals = self.sess.run(tvars)
                param_dict = {v.name: val for v, val in zip(tvars, tvals)}
                self.sess.close()
                tf.reset_default_graph()
                self.clusters += 1
                model, saver, data_generator = self.construct_model()
                tf.train.start_queue_runners(self.sess)
                tf.global_variables_initializer().run()
                for v in tf.trainable_variables():
                    if v.name in param_dict:
                        self.sess.run(v.assign(param_dict[v.name]))
                n_way = data_generator.n_way
                k_shot = data_generator.k_shot
                q_query = data_generator.q_query
                K = FLAGS.update_batch_size
                Q = FLAGS.update_batch_size_eval
                train_fetches[:] = [
                    model.metatrain_op,
                    model.total_embed_loss,
                    model.total_losses2[FLAGS.num_updates - 1],
                    model.total_accuracy1,
                    model.total_accuracies2[FLAGS.num_updates - 1],
                ]
                memory.set_min_quota(range(1, seen_task_count + 1))
                print("[HSML] Cluster added. Total clusters = {}".format(self.clusters))

            train_fetches = [
                model.metatrain_op,
                model.total_embed_loss,
                model.total_losses2[FLAGS.num_updates - 1],
                model.total_accuracy1,
                model.total_accuracies2[FLAGS.num_updates - 1],
            ]

            def _finalize_stage(reason, iter_idx):
                # --- START OF FIX ---
                # Removed 'episodes_in_stage' from the nonlocal declaration
                nonlocal current_dataset_idx, current_task_id, seen_task_count
                nonlocal last_avg, q_loss_sum, q_iters
                # --- END OF FIX ---
                eval_names = data_generator.nonstat_order[:current_dataset_idx + 1]
                eval_results = evaluate_seen_datasets(eval_names)
                stage_avg = float(np.mean([eval_results[n]['mean'] for n in eval_names])) if eval_names else 0.0

                print('[EVAL] dataset {} complete | reason={} | average={:.4f}'.format(
                    data_generator.nonstat_order[current_dataset_idx], reason, stage_avg))
                for name in eval_names:
                    stats = eval_results[name]
                    print('    {:>12}: {:.4f} ± {:.4f}'.format(name, stats['mean'], stats['std']))

                metrics_records.append({
                    'stage': current_dataset_idx + 1,
                    'dataset': data_generator.nonstat_order[current_dataset_idx],
                    'results': eval_results,
                    'average': stage_avg,
                    'reason': reason,
                    'iter': int(iter_idx),
                })
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_records, f, indent=2)

                # Lower-triangle accuracy matrix bookkeeping
                row_means = [float(eval_results[n]['mean']) for n in eval_names]
                acc_matrix.append(row_means)
                diag_means.append(float(eval_results[data_generator.nonstat_order[current_dataset_idx]]['mean']))

                # Pretty-print lower triangle (percent units)
                print('[MATRIX] Lower-triangle accuracy matrix (percent):')
                for i in range(len(acc_matrix)):
                    parts = []
                    for j in range(total_stages):
                        if j <= i:
                            val = 100.0 * acc_matrix[i][j]
                            parts.append('{:6.2f}'.format(val))
                        else:
                            parts.append('  --  ')
                    print(' '.join(parts))

                # Retained / Learned / BTI from the matrix
                last_row = acc_matrix[-1]
                retained = float(np.mean(last_row)) if last_row else 0.0
                learned = float(np.mean(diag_means)) if diag_means else 0.0
                bti_vals = [last_row[i] - diag_means[i] for i in range(len(diag_means))]
                bti_mean = float(np.mean(bti_vals)) if bti_vals else 0.0
                print('[METRICS] Retained={:.4f} Learned={:.4f} BTI_mean={:.4f}'.format(
                    retained, learned, bti_mean))

                current_dataset_idx += 1
                current_task_id += 1
                seen_task_count += 1
                # --- START OF FIX ---
                # Removed the line 'episodes_in_stage = 0' as it's no longer used
                # --- END OF FIX ---
                memory.set_min_quota(range(1, seen_task_count + 1))

                # Optionally add a new cluster at dataset boundaries.
                if getattr(FLAGS, 'nonstat_add_cluster_on_boundary', True) and current_dataset_idx < total_stages:
                    _add_cluster_and_rebuild(iter_idx)

                if use_online:
                    last_avg = None
                    q_loss_sum = 0.0
                    q_iters = 0

            def _pack_episodes(episodes):
                inputa = np.zeros((len(episodes), K, data_generator.dim_input), dtype=np.float32)
                labela = np.zeros((len(episodes), K), dtype=np.int32)
                inputb = np.zeros((len(episodes), Q, data_generator.dim_input), dtype=np.float32)
                labelb = np.zeros((len(episodes), Q), dtype=np.int32)
                for i, (_tid, sx, sy, qx, qy) in enumerate(episodes):
                    inputa[i] = sx.reshape(K, -1)
                    labela[i] = sy.reshape(K)
                    inputb[i] = qx.reshape(Q, -1)
                    labelb[i] = qy.reshape(Q)
                return inputa, labela, inputb, labelb

            def evaluate_seen_datasets(seen_names):
                results = {}
                for name in seen_names:
                    accs = []
                    for _ in range(eval_tasks):
                        sx, sy, qx, qy = data_generator.sample_eval_episode_nonstationary(name, n_way, k_shot, q_query)
                        inputa = sx.reshape(1, K, -1)
                        labela = sy.reshape(1, K)
                        inputb = qx.reshape(1, Q, -1)
                        labelb = qy.reshape(1, Q)

                        mb = FLAGS.meta_batch_size
                        if inputa.shape[0] != mb:
                            reps = (mb + inputa.shape[0] - 1) // inputa.shape[0]
                            inputa = np.tile(inputa, (reps, 1, 1))[:mb]
                            labela = np.tile(labela, (reps, 1))[:mb]
                            inputb = np.tile(inputb, (reps, 1, 1))[:mb]
                            labelb = np.tile(labelb, (reps, 1))[:mb]

                        feed = {
                            model.inputa: inputa,
                            model.labela: labela,
                            model.inputb: inputb,
                            model.labelb: labelb,
                            model.meta_lr: 0.0,
                        }
                        post_acc = self.sess.run(model.total_accuracies2[FLAGS.num_updates - 1], feed)
                        accs.append(float(np.mean(post_acc)))
                    mean_acc = float(np.mean(accs)) if accs else 0.0
                    std_acc = float(np.std(accs)) if accs else 0.0
                    results[name] = {'mean': mean_acc, 'std': std_acc}
                return results

            total_stages = len(data_generator.nonstat_order)

            memory = GlobalReservoir(capacity=FLAGS.memory_size)
            current_dataset_idx = 0
            current_task_id = 1
            seen_task_count = 1
            memory.set_min_quota([current_task_id])
            
            global_itr = 0 # Use a global iteration counter

            # --- START OF FIX: Restructure the loop ---
            # Outer loop for each dataset (stage)
            for stage_idx in range(total_stages):
                current_dataset_idx = stage_idx
                dataset_name = data_generator.nonstat_order[current_dataset_idx]
                print(f"\n--- Starting Stage {stage_idx + 1}/{total_stages} on dataset: {dataset_name} ---")

                # Track dataset id for buffer; keep task_id aligned with stage index (1-based)
                current_task_id = stage_idx + 1
                seen_task_count = current_task_id
                memory.set_min_quota(range(1, seen_task_count + 1))

                triggered = False
                for iter_idx in range(iterations_per_dataset):
                    itr = global_itr  # Use global_itr for logging
                    episodes = []
                    total_added = 0

                    for ep_idx in range(meta_batch):
                        # Sample a fresh episode with global labels
                        sx, sy, qx, qy, sg, qg = data_generator.sample_episode_nonstationary(
                            dataset_name, n_way, k_shot, q_query, return_global=True)

                        # Build mapping from episodic label -> global label
                        epi_to_global = {}
                        for i in range(n_way):
                            idx = np.where(sy == i)[0]
                            if idx.size > 0:
                                epi_to_global[i] = int(sg[idx[0]])
                            else:
                                idxq = np.where(qy == i)[0]
                                epi_to_global[i] = int(qg[idxq[0]]) if idxq.size > 0 else int(i)

                        # Alternate per-episode buffer quota to reach exact 50% total queries
                        q_buf = (q_query // 2) if (ep_idx % 2 == 0) else (q_query - (q_query // 2))
                        q_act = q_query - q_buf

                        qx_list, qy_list = [], []
                        for i in range(n_way):
                            idx = np.where(qy == i)[0]
                            if idx.size == 0:
                                continue
                            take_act = np.random.choice(idx, size=q_act, replace=(idx.size < q_act))
                            qx_act = qx[take_act]
                            qy_act = qy[take_act]

                            qx_buf = np.zeros((0, qx.shape[1]), dtype=qx.dtype)
                            qy_buf = np.zeros((0,), dtype=np.int32)
                            if stage_idx > 0 and memory.total > 0 and q_buf > 0:
                                g = epi_to_global[i]
                                qx_b, _ = memory.sample_dataset_class(stage_idx, g, q_buf)
                                if qx_b is not None:
                                    qx_buf = qx_b
                                    qy_buf = np.full((qx_b.shape[0],), i, dtype=np.int32)

                            # Top up from current queries if buffer is short/empty
                            if qx_buf.shape[0] < q_buf:
                                need = q_buf - qx_buf.shape[0]
                                take_extra = np.random.choice(idx, size=need, replace=(idx.size < need))
                                qx_extra = qx[take_extra]
                                qy_extra = qy[take_extra]
                                qx_buf = np.concatenate([qx_buf, qx_extra], axis=0)
                                qy_buf = np.concatenate([qy_buf, qy_extra], axis=0)

                            qx_list.append(np.concatenate([qx_act, qx_buf], axis=0))
                            qy_list.append(np.concatenate([qy_act, qy_buf], axis=0))

                            # Add only buffer quota from current query into buffer (global labels)
                            if q_buf > 0:
                                take_add = np.random.choice(idx, size=q_buf, replace=(idx.size < q_buf))
                                memory.add_samples_with_dataset(
                                    current_task_id, stage_idx, qx[take_add], qg[take_add])
                                total_added += int(q_buf)

                        qx_mixed = np.concatenate(qx_list, axis=0)
                        qy_mixed = np.concatenate(qy_list, axis=0)
                        perm = np.random.permutation(qx_mixed.shape[0])
                        qx_mixed = qx_mixed[perm]
                        qy_mixed = qy_mixed[perm]

                        episodes.append((current_task_id,
                                         sx.copy(), sy.copy(),
                                         qx_mixed.copy(), qy_mixed.copy()))

                    inputa, labela, inputb, labelb = _pack_episodes(episodes)
                    feed_train = {
                        model.inputa: inputa,
                        model.labela: labela,
                        model.inputb: inputb,
                        model.labelb: labelb,
                    }
                    _, emb_loss, post_loss, pre_acc, post_acc = self.sess.run(train_fetches, feed_train)

                    if FLAGS.print_interval > 0 and ((itr + 1) % FLAGS.print_interval == 0):
                        print("Iter {:6d} | dataset={} | post_acc={:.4f} | embed_loss={:.4f} | buf_added={}".format(
                            itr + 1, dataset_name, float(np.mean(post_acc)), float(emb_loss), total_added))

                    # Periodic test evaluation across seen datasets.
                    if ((itr + 1) % eval_interval == 0):
                        seen_eval_names = data_generator.nonstat_order[:stage_idx + 1]
                        # Use a smaller eval budget for interval logging to avoid stalls.
                        interval_results = {}
                        for name in seen_eval_names:
                            accs = []
                            for _ in range(eval_tasks_interval):
                                sx, sy, qx, qy = data_generator.sample_eval_episode_nonstationary(name, n_way, k_shot, q_query)
                                inputa = sx.reshape(1, K, -1)
                                labela = sy.reshape(1, K)
                                inputb = qx.reshape(1, Q, -1)
                                labelb = qy.reshape(1, Q)

                                mb = FLAGS.meta_batch_size
                                if inputa.shape[0] != mb:
                                    reps = (mb + inputa.shape[0] - 1) // inputa.shape[0]
                                    inputa = np.tile(inputa, (reps, 1, 1))[:mb]
                                    labela = np.tile(labela, (reps, 1))[:mb]
                                    inputb = np.tile(inputb, (reps, 1, 1))[:mb]
                                    labelb = np.tile(labelb, (reps, 1))[:mb]

                                feed = {
                                    model.inputa: inputa,
                                    model.labela: labela,
                                    model.inputb: inputb,
                                    model.labelb: labelb,
                                    model.meta_lr: 0.0,
                                }
                                post_acc = self.sess.run(model.total_accuracies2[FLAGS.num_updates - 1], feed)
                                accs.append(float(np.mean(post_acc)))
                            mean_acc = float(np.mean(accs)) if accs else 0.0
                            std_acc = float(np.std(accs)) if accs else 0.0
                            interval_results[name] = {'mean': mean_acc, 'std': std_acc}
                        interval_avg = float(np.mean([interval_results[n]['mean'] for n in seen_eval_names]))
                        print('[EVAL-INT] iter={} avg={:.4f}'.format(itr + 1, interval_avg))
                        for name in seen_eval_names:
                            stats = interval_results[name]
                            print('    {:>12}: {:.4f} ± {:.4f}'.format(name, stats['mean'], stats['std']))

                    if use_online:
                        q_loss_sum += float(post_loss)
                        q_iters += 1
                        if q_iters >= WINDOW:
                            new_avg = q_loss_sum / q_iters
                            triggered = False
                            if last_avg is not None:
                                ratio = new_avg / (last_avg + 1e-8) if last_avg is not None else 0.0
                                print(f"[HSML-DEBUG] Window ended. last_avg_loss={last_avg:.4f}, new_avg_loss={new_avg:.4f}, ratio={ratio:.2f}, threshold={MU:.2f}")
                                threshold = MU * last_avg
                                if last_avg == 0.0: threshold = MU * 1e-8
                                if new_avg > threshold: triggered = True
                            last_avg = new_avg
                            q_loss_sum = 0.0
                            q_iters = 0
                            if triggered:
                                print('[HSML] Online detector triggered at iter {} (avg_loss={:.6f}).'.format(itr + 1, new_avg))
                                _add_cluster_and_rebuild(itr + 1)
                                break

                    global_itr += 1

                    if use_online and triggered:
                        break

                # This code now runs AFTER the inner loops for a stage are complete (or broken by trigger)
                _finalize_stage('budget' if not (use_online and triggered) else 'detected', global_itr)
                # The current_task_id is incremented inside _finalize_stage
            # --- END OF FIX ---

            print('\n[NONSTATIONARY] Training complete. Metrics written to {}'.format(metrics_path))
            return

        elif FLAGS.datasource == 'vggflowers_stationary':
            if not hasattr(data_generator, 'nonstat_train_registry'):
                raise ValueError('Stationary mode requires nonstationary dataset registry; check --nonstat_data_root.')

            dataset_name = getattr(data_generator, 'stationary_dataset', FLAGS.stationary_dataset)
            if dataset_name not in data_generator.nonstat_train_registry:
                raise ValueError('Dataset {} not present in registry.'.format(dataset_name))

            n_way = data_generator.n_way
            k_shot = data_generator.k_shot
            q_query = data_generator.q_query
            meta_batch = FLAGS.meta_batch_size
            K = FLAGS.update_batch_size
            Q = FLAGS.update_batch_size_eval

            total_iters = max(1, int(FLAGS.stationary_iterations))
            eval_interval = max(1, int(FLAGS.stationary_eval_interval))
            eval_tasks = FLAGS.nonstat_eval_tasks

            metrics_dir = os.path.join(FLAGS.logdir, exp_string)
            os.makedirs(metrics_dir, exist_ok=True)
            metrics_path = os.path.join(metrics_dir, 'stationary_metrics.json')
            metrics_records = []

            train_fetches = [
                model.metatrain_op,
                model.total_embed_loss,
                model.total_losses2[FLAGS.num_updates - 1],
                model.total_accuracy1,
                model.total_accuracies2[FLAGS.num_updates - 1],
            ]

            def _run_eval(tag, iter_idx):
                accs = []
                for _ in range(eval_tasks):
                    sx, sy, qx, qy = data_generator.sample_eval_episode_nonstationary(dataset_name, n_way, k_shot, q_query)
                    inputa = sx.reshape(1, K, -1)
                    labela = sy.reshape(1, K)
                    inputb = qx.reshape(1, Q, -1)
                    labelb = qy.reshape(1, Q)
                    if meta_batch > 1:
                        reps = (meta_batch + inputa.shape[0] - 1) // inputa.shape[0]
                        inputa = np.tile(inputa, (reps, 1, 1))[:meta_batch]
                        labela = np.tile(labela, (reps, 1))[:meta_batch]
                        inputb = np.tile(inputb, (reps, 1, 1))[:meta_batch]
                        labelb = np.tile(labelb, (reps, 1))[:meta_batch]
                    feed_eval = {
                        model.inputa: inputa,
                        model.labela: labela,
                        model.inputb: inputb,
                        model.labelb: labelb,
                        model.meta_lr: 0.0,
                    }
                    post_acc = self.sess.run(model.total_accuracies2[FLAGS.num_updates - 1], feed_eval)
                    accs.append(float(np.mean(post_acc)))
                mean_acc = float(np.mean(accs)) if accs else 0.0
                std_acc = float(np.std(accs)) if accs else 0.0
                print('[EVAL][{}] iter={} dataset={} mean={:.4f} std={:.4f}'.format(tag, iter_idx, dataset_name, mean_acc, std_acc))
                metrics_records.append({
                    'iter': int(iter_idx),
                    'dataset': dataset_name,
                    'mean_acc': mean_acc,
                    'std_acc': std_acc,
                })
                with open(metrics_path, 'w') as f:
                    json.dump(metrics_records, f, indent=2)

            for itr in range(total_iters):
                inputa, labela, inputb, labelb = data_generator.sample_meta_batch_nonstationary(
                    dataset_name, meta_batch, n_way, k_shot, q_query)
                feed_train = {
                    model.inputa: inputa,
                    model.labela: labela,
                    model.inputb: inputb,
                    model.labelb: labelb,
                }
                _, emb_loss, post_loss, pre_acc, post_acc = self.sess.run(train_fetches, feed_train)
                if FLAGS.print_interval > 0 and ((itr + 1) % FLAGS.print_interval == 0):
                    print('Iter {:6d} | dataset={} | post_acc={:.4f} | embed_loss={:.4f}'.format(
                        itr + 1, dataset_name, float(np.mean(post_acc)), float(emb_loss)))

                if eval_interval > 0 and ((itr + 1) % eval_interval == 0):
                    _run_eval('train', itr + 1)

            _run_eval('final', total_iters)
            print('\n[STATIONARY] Training complete. Metrics written to {}'.format(metrics_path))
            return





        else: # Original training loop
            # ORIGINAL, UNTOUCHED LOOP for sinusoid/mixture
            for itr in range(resume_itr, FLAGS.pretrain_iterations + FLAGS.metatrain_iterations):
                feed_dict = {}
                if 'generate' in dir(data_generator):
                    if FLAGS.datasource == 'sinusoid':
                        batch_x, batch_y, amp, phase = data_generator.generate()
                    elif FLAGS.datasource == 'mixture':
                        batch_x, batch_y, para_func, sel_set = data_generator.generate(itr=itr)

                    inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
                    labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
                    inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]  # b used for testing
                    labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]
                    feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb}

                if itr < FLAGS.pretrain_iterations:
                    input_tensors = [model.pretrain_op]
                else:
                    input_tensors = [model.metatrain_op]

                input_tensors.extend(
                    [model.summ_op, model.total_embed_loss, model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]])
                if model.classification:
                    input_tensors.extend([model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]])

                result = self.sess.run(input_tensors, feed_dict)

                prelosses.append(result[-2])
                postlosses.append(result[-1])
                embedlosses.append(result[2])

                # online training criterion
                # if FLAGS.online_training and itr % check_interval == 0 and itr != 0 and itr != 25000:
                if FLAGS.online_training and itr % PRINT_INTERVAL == 0 and (
                        (np.mean(postlosses) > FLAGS.online_threshold * last_res and not model.classification) or (
                        model.classification and np.mean(postlosses) < FLAGS.online_threshold * last_res)) and ((itr-change_itr)>FLAGS.online_change):
                    print(
                        'the old postlosses is {}, and the new postlosses is {}'.format(
                            last_res, np.mean(postlosses)))
                    # if itr == 25100 and model.classification:
                    tvars = tf.trainable_variables()
                    tvars_vals = self.sess.run(tvars)
                    for var, val in zip(tvars, tvars_vals):
                        # print(var.name, val)
                        param_dict[var.name] = val
                    tf.reset_default_graph()
                    self.clusters += 1
                    model, saver, data_generator = self.construct_model()
                    tf.train.start_queue_runners(self.sess)
                    tf.global_variables_initializer().run()
                    tvars = tf.trainable_variables()
                    tvars_vals = self.sess.run(tvars)
                    for var, val in zip(tvars, tvars_vals):
                        if var.name in param_dict:
                            self.sess.run(var.assign(param_dict[var.name]))
                    # tvars = tf.trainable_variables()
                    # tvars_vals = sess.run(tvars)
                    # for var, val in zip(tvars, tvars_vals):
                    #     print(var.name, val)
                    change_itr = itr
                    print(
                        'add a new cluster in the online training mode in epoch {}, the total number of cluster is {}'.format(
                            change_itr, self.clusters))

                if FLAGS.online_training and itr % PRINT_INTERVAL == 0 and itr != 0:
                    last_res = np.mean(postlosses)
                    print(last_res)


                if itr % SUMMARY_INTERVAL == 0:
                    if FLAGS.log:
                        train_writer.add_summary(result[1], itr)

                if (itr != 0) and itr % PRINT_INTERVAL == 0:
                    if itr < FLAGS.pretrain_iterations:
                        print_str = 'Pretrain Iteration ' + str(itr)
                    else:
                        print_str = 'Iteration ' + str(itr - FLAGS.pretrain_iterations)
                    std = np.std(postlosses)
                    ci95 = 1.96 * std / np.sqrt(PRINT_INTERVAL)
                    print_str += ': preloss: ' + str(np.mean(prelosses)) + ', postloss: ' + str(
                        np.mean(postlosses)) + ', embedding loss: ' + str(np.mean(embedlosses)) + ', confidence: ' + str(
                        ci95)
                    print(print_str)
                    prelosses, postlosses, embedlosses = [], [], []

                if (itr != 0) and itr % SAVE_INTERVAL == 0:
                    saver.save(self.sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

                # sinusoid is infinite data, so no need to test on meta-validation set.
                if (itr != 0) and itr % TEST_PRINT_INTERVAL == 0 and (
                        FLAGS.datasource not in ['sinusoid', 'mixture']):
                    if 'generate' not in dir(data_generator):
                        feed_dict = {}
                        if model.classification:
                            input_tensors = [model.metaval_total_accuracy1,
                                             model.metaval_total_accuracies2[FLAGS.num_updates - 1], model.summ_op]
                        else:
                            input_tensors = [model.metaval_total_loss1, model.metaval_total_losses2[FLAGS.num_updates - 1],
                                             model.summ_op]
                    else:
                        if FLAGS.datasource == 'sinusoid':
                            batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                        elif FLAGS.datasource == 'mixture':
                            batch_x, batch_y, para_func = data_generator.generate(train=False)
                        inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
                        inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
                        labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
                        labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

                        feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                                     model.meta_lr: 0.0}
                        if model.classification:
                            input_tensors = [model.total_accuracy1, model.total_accuracies2[FLAGS.num_updates - 1]]
                        else:
                            input_tensors = [model.total_loss1, model.total_losses2[FLAGS.num_updates - 1]]

                    result = self.sess.run(input_tensors, feed_dict)
                    print('Validation results: ' + str(result[0]) + ', ' + str(result[1]))

            saver.save(self.sess, FLAGS.logdir + '/' + exp_string + '/model' + str(itr))

    def test(self, model, saver, exp_string, data_generator, test_num_updates=None):
        num_classes = data_generator.num_classes  # for classification, 1 otherwise

        np.random.seed(1)
        random.seed(1)

        metaval_accuracies = []
        print(self.NUM_TEST_POINTS)
        for test_itr in range(self.NUM_TEST_POINTS):
            if 'generate' not in dir(data_generator):
                feed_dict = {}
                feed_dict = {model.meta_lr: 0.0}
            else:
                if FLAGS.datasource == 'sinusoid':
                    batch_x, batch_y, amp, phase = data_generator.generate(train=False)
                elif FLAGS.datasource == 'mixture':
                    batch_x, batch_y, para_func, sel_set = data_generator.generate(test_itr, train=False)

                inputa = batch_x[:, :num_classes * FLAGS.update_batch_size, :]
                inputb = batch_x[:, num_classes * FLAGS.update_batch_size:, :]
                labela = batch_y[:, :num_classes * FLAGS.update_batch_size, :]
                labelb = batch_y[:, num_classes * FLAGS.update_batch_size:, :]

                feed_dict = {model.inputa: inputa, model.inputb: inputb, model.labela: labela, model.labelb: labelb,
                             model.meta_lr: 0.0}

            if model.classification:
                result = self.sess.run([model.metaval_total_accuracy1] + model.metaval_total_accuracies2, feed_dict)

            else:  # this is for sinusoid
                result = self.sess.run([model.total_loss1] + model.total_losses2, feed_dict)

            metaval_accuracies.append(result)

        metaval_accuracies = np.array(metaval_accuracies)
        means = np.mean(metaval_accuracies, 0)
        stds = np.std(metaval_accuracies, 0)
        ci95 = 1.96 * stds / np.sqrt(self.NUM_TEST_POINTS)

        print('Mean validation accuracy/loss, stddev, and confidence intervals')
        print((means, stds, ci95))

    def construct_model(self):
        
# TF1-style session with GPU/CPU tuning
        config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,          # place ops on GPU/CPU as needed
            log_device_placement=False
        )
        config.gpu_options.allow_growth = True  # no big upfront GPU malloc
        # Tune these to your host; 8/8 is a safe start on multi-core boxes.
        # Set to 0 to let TF decide if you’re unsure.
        config.intra_op_parallelism_threads = 8
        config.inter_op_parallelism_threads = 8
        self.sess = tf.compat.v1.InteractiveSession(config=config)
        if FLAGS.train == False:
            orig_meta_batch_size = FLAGS.meta_batch_size
            # always use meta batch size of 1 when testing.
            FLAGS.meta_batch_size = 1

        if FLAGS.datasource in ['sinusoid', 'mixture']:
            data_generator = DataGenerator(FLAGS.update_batch_size + FLAGS.update_batch_size_eval,
                                           FLAGS.meta_batch_size)
        else:
            if FLAGS.metatrain_iterations == 0 and FLAGS.datasource in ['miniimagenet', 'multidataset']:
                assert FLAGS.meta_batch_size == 1
                assert FLAGS.update_batch_size == 1
                data_generator = DataGenerator(1, FLAGS.meta_batch_size)  # only use one datapoint,
            else:
                if FLAGS.datasource in ['miniimagenet', 'multidataset']:  # TODO - use 15 val examples for imagenet?
                    if FLAGS.train:
                        data_generator = DataGenerator(FLAGS.update_batch_size + 15,
                                                       FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                    else:
                        data_generator = DataGenerator(FLAGS.update_batch_size * 2,
                                                       FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory
                else:
                    data_generator = DataGenerator(FLAGS.update_batch_size * 2,
                                                   FLAGS.meta_batch_size)  # only use one datapoint for testing to save memory

        dim_output = data_generator.dim_output

        dim_input = data_generator.dim_input

        if FLAGS.datasource in ['miniimagenet', 'omniglot', 'multidataset']:
            tf_data_load = True
            num_classes = data_generator.num_classes

            if FLAGS.train:  # only construct training model if needed
                random.seed(5)
                if FLAGS.datasource in ['miniimagenet', 'omniglot']:
                    image_tensor, label_tensor = data_generator.make_data_tensor()
                elif FLAGS.datasource == 'multidataset':
                    image_tensor, label_tensor = data_generator.make_data_tensor_multidataset(sel_num=self.clusters,
                                                                                             train=True)
                inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
                inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
                labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
                labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
                input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}

            random.seed(6)
            if FLAGS.datasource in ['miniimagenet', 'omniglot']:
                image_tensor, label_tensor = data_generator.make_data_tensor(train=False)
            elif FLAGS.datasource == 'multidataset':
                image_tensor, label_tensor = data_generator.make_data_tensor_multidataset(sel_num=self.clusters,
                                                                                         train=False)
            inputa = tf.slice(image_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            inputb = tf.slice(image_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            labela = tf.slice(label_tensor, [0, 0, 0], [-1, num_classes * FLAGS.update_batch_size, -1])
            labelb = tf.slice(label_tensor, [0, num_classes * FLAGS.update_batch_size, 0], [-1, -1, -1])
            metaval_input_tensors = {'inputa': inputa, 'inputb': inputb, 'labela': labela, 'labelb': labelb}
        else:
            tf_data_load = False
            input_tensors = None

        model = MAML(self.sess, dim_input, dim_output, test_num_updates=self.test_num_updates)

        model.cluster_layer_0 = self.clusters

        if FLAGS.train or not tf_data_load:
            model.construct_model(input_tensors=input_tensors, prefix='metatrain_')
        if tf_data_load:
            model.construct_model(input_tensors=metaval_input_tensors, prefix='metaval_')
        model.summ_op = tf.summary.merge_all()
        saver = loader = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES), max_to_keep=10)

        # Report total trainable parameter count for reproducibility comparisons.
        try:
            total_params = 0
            for v in tf.trainable_variables():
                shape = v.shape.as_list()
                if shape is None:
                    continue
                n = 1
                for d in shape:
                    if d is None:
                        n = 0
                        break
                    n *= int(d)
                total_params += n
            print('[MODEL] Trainable parameters: {:,}'.format(int(total_params)))
        except Exception as e:
            print('[MODEL] Parameter count unavailable: {}'.format(e))

        if FLAGS.train == False:
            # change to original meta batch size when loading model.
            FLAGS.meta_batch_size = orig_meta_batch_size

        if FLAGS.train_update_batch_size == -1:
            FLAGS.train_update_batch_size = FLAGS.update_batch_size
        if FLAGS.train_update_lr == -1:
            FLAGS.train_update_lr = FLAGS.update_lr

        return model, saver, data_generator

    def main(self):
        # Set seeds for reproducibility per run
        seed_val = int(getattr(FLAGS, 'seed', 0))
        np.random.seed(seed_val)
        random.seed(seed_val)
        tf.set_random_seed(seed_val)

        if FLAGS.datasource in ['sinusoid', 'mixture']:
            if FLAGS.train:
                self.test_num_updates = 1
            else:
                self.test_num_updates = 10
        else:
            if FLAGS.datasource in ['miniimagenet', 'multidataset']:
                if FLAGS.train == True:
                    self.test_num_updates = 1  # eval on at least one update during training
                else:
                    self.test_num_updates = 10
            else:
                self.test_num_updates = 10

        model, saver, data_generator = self.construct_model()

        exp_string = 'cls_' + str(FLAGS.num_classes) + '.mbs_' + str(FLAGS.meta_batch_size) + '.ubs_' + str(
            FLAGS.train_update_batch_size) + '.numstep' + str(FLAGS.num_updates) + '.updatelr' + str(
            FLAGS.train_update_lr) + '.metalr' + str(FLAGS.meta_lr) + '.emb_loss_weight' + str(
            FLAGS.emb_loss_weight) + '.num_groups' + str(FLAGS.num_groups) + '.emb_type' + str(
            FLAGS.emb_type) + '.hidden_dim' + str(FLAGS.hidden_dim)

        if FLAGS.num_filters != 64:
            exp_string += 'hidden' + str(FLAGS.num_filters)
        if FLAGS.max_pool:
            exp_string += 'maxpool'
        if FLAGS.stop_grad:
            exp_string += 'stopgrad'
        if FLAGS.norm == 'batch_norm':
            exp_string += 'batchnorm'
        elif FLAGS.norm == 'layer_norm':
            exp_string += 'layernorm'
        elif FLAGS.norm == 'None':
            exp_string += 'nonorm'
        else:
            print('Norm setting not recognized.')

        resume_itr = 0
        model_file = None

        tf.train.start_queue_runners(self.sess)
        variables = tf.global_variables()
        self.sess.run(tf.variables_initializer(variables))
        if FLAGS.resume or not FLAGS.train:
            if FLAGS.train == True:
                # model_file = '{0}/{2}/model{1}'.format(FLAGS.logdir, FLAGS.test_epoch, exp_string)
                model_file = tf.train.latest_checkpoint(FLAGS.logdir + '/' + exp_string)
            else:
                print(FLAGS.test_epoch)
                model_file = '{0}/{2}/model{1}'.format(FLAGS.logdir, FLAGS.test_epoch, exp_string)
            if model_file:
                ind1 = model_file.index('model')
                resume_itr = int(model_file[ind1 + 5:])
                print("Restoring model weights from " + model_file)
                saver.restore(self.sess, model_file)

        if FLAGS.train:
            self.train(model, saver, exp_string, data_generator, resume_itr)
        else:
            self.test(model, saver, exp_string, data_generator, self.test_num_updates)

        print('The number of clusters are {}'.format(self.clusters))


if __name__ == "__main__":
    online_model = HSML_Online()
    online_model.main()
