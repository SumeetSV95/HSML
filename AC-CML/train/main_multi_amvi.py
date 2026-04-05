import shutil
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

import numpy as np
import datetime
import os
import json
import random
from platform import system
from warnings import filterwarnings
import sys
sys.path.append("..")

import optim
from config.configuration import get_run_name
from data_generate.dataset import FewShotImageDataset, RotatedMNISTDataset, RotatedMNISTTaskDataset, PermutedMNISTTaskDataset
from data_generate.sampler import SuppQueryBatchSampler

# Torchmeta imports torchmeta.datasets, which expects older torchvision helper symbols.
# Patch torchvision.datasets.utils to provide stubs so torchmeta can import with newer torchvision.
import torchvision.datasets.utils as tvu
if not hasattr(tvu, '_get_confirm_token'):
    def _get_confirm_token(response):
        return None
    tvu._get_confirm_token = _get_confirm_token
if not hasattr(tvu, '_save_response_content'):
    def _save_response_content(response, destination, chunk_size=32768):
        for chunk in response.iter_content(chunk_size):
            if chunk: destination.write(chunk)
    tvu._save_response_content = _save_response_content

from train import model as models
from train.variational import  VariationalApprox_v2, var_approx_beta
from train.boml import metatrain_seqdataset_multi_amvi
from train.util import enlist_transformation
from train.evidential_sparsity import evidential_sparsity
from evaluate.util_eval import meta_evaluation_amvi_v2, meta_evaluation_amvi_v3

torch.backends.cudnn.enabled = True
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(config, run_spec, seed=0):
    
    torch.manual_seed(seed)

    start_datetime = datetime.datetime.now()
    experiment_date = '{:%Y-%m-%d_%H:%M:%S}'.format(start_datetime)
    config['experiment_parent_dir'] = os.path.join(config['run_dir'], get_run_name(config['dataset_ls']))
    config['experiment_dir'] = os.path.join(config['experiment_parent_dir'],
                                             '{}_{}_{}'.format(run_spec, experiment_date, seed))
    config['experiment_dir'] = os.path.join(config['experiment_parent_dir'],
                                            '{}'.format(run_spec+experiment_date))


    # print the related information on the screen
    os.system("echo 'running {}_{} seed {}'".format(run_spec, experiment_date, seed)) if system() == 'Linux' \
        else print('running {}_{} seed {}'.format(run_spec, experiment_date, seed))
    
    # save config json file
    # if not os.path.exists(config['experiment_dir']):
    #     os.makedirs(config['experiment_dir'])

    if os.path.exists(config['experiment_dir']):
        if (input('{} exists, remove? ([y]/n): '.format(config['experiment_dir'])) != 'n'):
            shutil.rmtree(config['experiment_dir'])
            os.makedirs(config['experiment_dir'])
        else:
            config['experiment_dir'] += experiment_date
            os.makedirs(config['experiment_dir'])
    else:
        os.makedirs(config['experiment_dir'])


    with open(os.path.join(
            config['experiment_dir'],
            'config{}_{}.json'.format(0 if config['completed_task_idx'] is None
                                      else config['completed_task_idx'] + 1, run_spec)
    ), 'w') as outfile:
        outfile.write(json.dumps(config, indent=4))

    # define result directory and previous result directory if applicable
    if config['completed_task_idx'] is not None:
        completed_result_dir = os.path.join(
            os.path.join(os.path.join(config['run_dir'], get_run_name(config['dataset_ls'])),
                         config['completed_exp_name']),
            'result'
        )
    else:
        completed_result_dir = None

    # define tensorboard writer
    writer = SummaryWriter(os.path.join(config['experiment_dir'], 'logtb'))
    result_dir = os.path.join(config['experiment_dir'], 'result')
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # define model
    '''
    Note that the initial variational object should be modified.
    '''
    model = getattr(models, config['net'])(**config['net_kwargs']).to(device=config['device'])
    
    # define the initial variational object of theta
    var_approx_list = []

    # define the list to save the previous veta
    var_previous_beta_alpha = []
    var_previous_beta_beta = []
    
    for i in range(config['k']):
        
        # load the existing parameter or create the new parameter
        if config["completed_task_idx"] is not None:
            mean_init = torch.load(
                os.path.join(completed_result_dir, 'mean{}_varapprox{}.pt'.format(config["completed_task_idx"], i)),
                map_location="cpu"
            )
            covar_init = torch.load(
                os.path.join(completed_result_dir, 'covar{}_varapprox{}.pt'.format(config["completed_task_idx"], i)),
                map_location="cpu"
            )
            approx = VariationalApprox_v2(
                config['device'], config["num_mc_sample"], mean_init=mean_init, 
                covar_init=covar_init, init_optim_lrsch=False
            )
        else:
            model_temp = getattr(models, config['net'])(**config['net_kwargs'])
            approx = VariationalApprox_v2(
                config['device'], config["num_mc_sample"], model=model_temp, init_optim_lrsch=False
            )
        
        approx.update_mean_cov()
        var_approx_list.append(approx)
        
    
    # define the variational object of beta
    alpha = config["alpha"] if config['completed_task_idx'] is None else \
        np.load(os.path.join(completed_result_dir, "alpha{}.npy".format(config['completed_task_idx'])))

    var_beta = var_approx_beta(alpha, config['k'], config['beta_num_mc_sample'], config["device"], init_optim_lrsch=False, implicit=config["implicit"])

    # sparsity_vec = evidential_sparsity(var_beta, var_previous_beta_alpha, var_previous_beta_beta, config['eta'])
    # delete_num = np.sum(np.where(sparsity_vec==1))
    # print(sparsity_vec)
    # print(delete_num)
    # print(aaa)
    # load the checkpoint if existing
    if config['completed_task_idx'] is not None:
        prev_glob_step \
            = torch.load(os.path.join(completed_result_dir, 'prev_glob_step{}.pt'.format(config['completed_task_idx'])))
        evalset = torch.load(
            os.path.join(completed_result_dir, 'evalset{}.pt'.format(config['completed_task_idx'])))
    else:
        prev_glob_step = 0
        evalset = []
    eval_snapshots = []

    def load_rotmnist_tasks(train_path, test_path):
        import pickle
        with open(train_path, 'rb') as f:
            train_tasks = pickle.load(f)
        with open(test_path, 'rb') as f:
            test_tasks = pickle.load(f)
        # ensure list ordering
        if isinstance(train_tasks, dict):
            train_tasks = [train_tasks[k] for k in sorted(train_tasks.keys())]
        if isinstance(test_tasks, dict):
            test_tasks = [test_tasks[k] for k in sorted(test_tasks.keys())]
        return train_tasks, test_tasks

    def load_permmnist_tasks(pt_path):
        train_raw, test_raw = torch.load(pt_path, map_location="cpu")
        def unpack(raw):
            tasks = []
            for entry in raw:
                _tag, x, y = entry
                tasks.append((x.numpy() if hasattr(x, "numpy") else x,
                              y.numpy() if hasattr(y, "numpy") else y))
            return tasks
        return unpack(train_raw), unpack(test_raw)

    print("=============={}================".format(len(evalset)))

    # get all the dataset name
    num_dataset_to_run = len(config['dataset_ls']) if config['num_dataset_to_run'] == 'all' \
        else config['num_dataset_to_run']

    def run_eval_block(tag, step_idx, task_name, evalsets):
        """Evaluate all accumulated evalsets and return list of mean accuracies."""
        means = []
        for ldr_idx, evset in enumerate(evalsets):
            if not config['max_eval']:
                loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi_v2(
                    evset, num_task=config[task_name]['num_eval_task'], task_by_supercls=config[task_name]['eval_task_by_supercls'],
                    num_way=config['net_kwargs']['num_way'], num_shot=config[task_name]['num_shot'],
                    num_query_per_cls=config[task_name]['num_query_per_cls'], model=model,
                    variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1,
                    nstep_inner=config[task_name]['nstep_inner_eval'], lr_inner=config[task_name]['lr_inner'], model_device=config['device'],
                    var_beta=var_beta, sample_batch=config['sample_batch']
                )
            else:
                loss_eval, loss_95ci, accuracy_eval, acc_95ci = meta_evaluation_amvi_v3(
                    evset, num_task=config[task_name]['num_eval_task'], task_by_supercls=config[task_name]['eval_task_by_supercls'],
                    num_way=config['net_kwargs']['num_way'], num_shot=config[task_name]['num_shot'],
                    num_query_per_cls=config[task_name]['num_query_per_cls'], model=model,
                    variational_obj_list=var_approx_list, inner_on_mean=True, n_sample=1,
                    nstep_inner=config[task_name]['nstep_inner_eval'], lr_inner=config[task_name]['lr_inner'], model_device=config['device'],
                    var_beta=var_beta, sample_batch=config['sample_batch']
                )
            means.append(accuracy_eval)
            writer.add_scalar(
                tag=f"{tag}/loss_meta_eval_task{ldr_idx}",
                scalar_value=loss_eval, global_step=step_idx
            )
            writer.add_scalar(
                tag=f"{tag}/accuracy_meta_eval_task{ldr_idx}",
                scalar_value=accuracy_eval, global_step=step_idx
            )
        return means
    
    for task_idx, task in enumerate(config['dataset_ls'][:num_dataset_to_run], 0):
        
        if config['increase_num'] and task_idx != 0:

            # Done: determine the number of cluster
            increase_num = 0
            if config['k'] + config['increase_k'] <= config['max_k']:
                increase_num = config['increase_k']
            elif config['k'] < config['max_k']:
                increase_num = config['max_k'] - config['k']
            
            if increase_num != 0:
                config['k'] += increase_num
                
                # add the cluster of beta
                var_beta.add_cluster(increase_num)

                # add the cluster of theta
                for _ in range(increase_num):
                    model_temp = getattr(models, config['net'])(**config['net_kwargs'])
                    approx = VariationalApprox_v2(
                        config['device'], config["num_mc_sample"], model=model_temp, init_optim_lrsch=False
                    )
                    approx.update_mean_cov()
                    var_approx_list.append(approx)

        print(config['k'])
        
        # init the berns
        # var_beta.update_bern()

        # train the model with different dataset
        if config['completed_task_idx'] is not None and config['completed_task_idx'] >= task_idx:
            # pass the completed dataset
            pass
        else:
            # determine dataset source
            use_rotmnist = (task == 'rotmnist')
            use_perm = task in ['permmnist', 'perm_mnist', 'mnist_permutations']

            # define optimiser and lr scheduler for each variational objective in the list
            for var_approx in var_approx_list:
                var_approx.optimizer = getattr(optim, config[task]['optim_outer_name']) \
                    (list(var_approx.mean.values()) + list(var_approx.covar.values()), **config[task]['optim_outer_kwargs'])
                if config[task]['lr_sch_outer_name'] is None:
                    var_approx.lr_scheduler = None
                else:
                    var_approx.lr_scheduler = getattr(lr_scheduler, config[task]['lr_sch_outer_name']) \
                        (optim_outer, **config[task]['lr_sch_outer_kwargs'])


            var_beta.optimizer = getattr(optim, config[task]['optim_outer_name']) \
                    (var_beta.parameters(), **config[task]['optim_outer_kwargs'])
            if config[task]['lr_sch_outer_name'] is None:
                var_beta.lr_scheduler = None
            else:
                var_beta.lr_scheduler = getattr(lr_scheduler, config[task]['lr_sch_outer_name']) \
                    (optim_outer, **config[task]['lr_sch_outer_kwargs'])

            transformation = transforms.Compose(
                enlist_transformation(img_resize=config['img_resize'], is_grayscale=config['is_grayscale'],
                                      device=config['device'], img_normalise=config[task]['img_normalise'])
            )

            if use_rotmnist and config.get('rotmnist_task_mode', False):
                train_pkl = os.path.join(config['data_dir'], 'rotated_mnist', 'train.pkl')
                test_pkl = os.path.join(config['data_dir'], 'rotated_mnist', 'test.pkl')
                train_tasks, test_tasks = load_rotmnist_tasks(train_pkl, test_pkl)
                evalsets_rot = []

                def pick(entry, keys):
                    for k in keys:
                        if k in entry:
                            return entry[k]
                    raise KeyError(f"None of {keys} found in task entry keys={entry.keys()}")

                for t_idx, task_entry in enumerate(train_tasks):
                    sx = pick(task_entry, ['support_x', 'supportX'])
                    sy = pick(task_entry, ['support_y', 'supportY'])
                    qx = pick(task_entry, ['query_x', 'queryX'])
                    qy = pick(task_entry, ['query_y', 'queryY'])
                    trainset = RotatedMNISTTaskDataset(sx, sy, qx, qy, transform=transformation,
                                                       device=config['device'], cuda_img_tensor=config['cuda_img_tensor'])
                    eval_task = test_tasks[t_idx]
                    esx = pick(eval_task, ['support_x', 'supportX'])
                    esy = pick(eval_task, ['support_y', 'supportY'])
                    eqx = pick(eval_task, ['query_x', 'queryX'])
                    eqy = pick(eval_task, ['query_y', 'queryY'])
                    task_evalset = RotatedMNISTTaskDataset(esx, esy, eqx, eqy, transform=transformation,
                                                           device=config['device'], cuda_img_tensor=config['cuda_img_tensor'])
                    evalsets_rot.append(task_evalset)

                    metatrain_seqdataset_multi_amvi(
                        model=model, var_approx_list=var_approx_list, var_beta=var_beta, model_device=config["device"], 
                        trainset=trainset, evalset=[task_evalset], outer_kl_scale=config[task]['outer_kl_scale'],
                        nstep_outer=config[task]['nstep_outer'], nstep_inner=config[task]['nstep_inner'],
                        lr_inner=config[task]['lr_inner'], first_order=config[task]['first_order'], seqtask=config['seqtask'],
                        num_task_per_itr=config[task]['num_task_per_itr'], task_by_supercls=False,
                        num_way=config['net_kwargs']['num_way'], num_shot=config[task]['num_shot'],
                        num_query_per_cls=config[task]['num_query_per_cls'], eval_prev_task=False,
                        eval_per_num_iter=config[task]['eval_per_num_iter'], num_eval_task=config[task]['num_eval_task'],
                        eval_task_by_supercls=False,
                        nstep_inner_eval=config[task]['nstep_inner_eval'], writer=writer, task_idx=t_idx,
                        prev_glob_step=prev_glob_step, verbose=f"{task}_task{t_idx}", sample_mean=config["sample_mean"], sample_batch=config["sample_batch"],
                        max_eval = config['max_eval']
                    )

                    final_means = run_eval_block(tag='final_eval', step_idx=prev_glob_step + config[task]['nstep_outer'], task_name=task, evalsets=evalsets_rot)
                    if final_means:
                        eval_snapshots.append(final_means)
                        diag = [eval_snapshots[i][i] for i in range(len(eval_snapshots))]
                        final_row = eval_snapshots[-1][:len(evalsets_rot)]
                        la_mean = float(np.mean(diag))
                        ra_mean = float(np.mean(final_row))
                        bti_mean = float(np.mean([f - d for f, d in zip(final_row, diag)]))
                        print(f"[METRICS] RA_mean={ra_mean:.4f} LA_mean={la_mean:.4f} BTI_mean={bti_mean:.4f}")

                    prev_glob_step += config[task]['nstep_outer']

            elif use_perm and config.get('permmnist_task_mode', False):
                perm_pt = os.path.join(config['data_dir'], 'mnist_permutations.pt')
                train_tasks, test_tasks = load_permmnist_tasks(perm_pt)
                evalsets_perm = []

                for t_idx, (x_tr, y_tr) in enumerate(train_tasks):
                    trainset = PermutedMNISTTaskDataset(x_tr, y_tr, transform=transformation,
                                                        device=config['device'], cuda_img_tensor=config['cuda_img_tensor'])
                    x_te, y_te = test_tasks[t_idx]
                    task_evalset = PermutedMNISTTaskDataset(x_te, y_te, transform=transformation,
                                                            device=config['device'], cuda_img_tensor=config['cuda_img_tensor'])
                    evalsets_perm.append(task_evalset)

                    metatrain_seqdataset_multi_amvi(
                        model=model, var_approx_list=var_approx_list, var_beta=var_beta, model_device=config["device"], 
                        trainset=trainset, evalset=[task_evalset], outer_kl_scale=config[task]['outer_kl_scale'],
                        nstep_outer=config[task]['nstep_outer'], nstep_inner=config[task]['nstep_inner'],
                        lr_inner=config[task]['lr_inner'], first_order=config[task]['first_order'], seqtask=config['seqtask'],
                        num_task_per_itr=config[task]['num_task_per_itr'], task_by_supercls=False,
                        num_way=config['net_kwargs']['num_way'], num_shot=config[task]['num_shot'],
                        num_query_per_cls=config[task]['num_query_per_cls'], eval_prev_task=False,
                        eval_per_num_iter=config[task]['eval_per_num_iter'], num_eval_task=config[task]['num_eval_task'],
                        eval_task_by_supercls=False,
                        nstep_inner_eval=config[task]['nstep_inner_eval'], writer=writer, task_idx=t_idx,
                        prev_glob_step=prev_glob_step, verbose=f"{task}_task{t_idx}", sample_mean=config["sample_mean"], sample_batch=config["sample_batch"],
                        max_eval = config['max_eval']
                    )

                    final_means = run_eval_block(tag='final_eval', step_idx=prev_glob_step + config[task]['nstep_outer'], task_name=task, evalsets=evalsets_perm)
                    if final_means:
                        eval_snapshots.append(final_means)
                        diag = [eval_snapshots[i][i] for i in range(len(eval_snapshots))]
                        final_row = eval_snapshots[-1][:len(evalsets_perm)]
                        la_mean = float(np.mean(diag))
                        ra_mean = float(np.mean(final_row))
                        bti_mean = float(np.mean([f - d for f, d in zip(final_row, diag)]))
                        print(f"[METRICS] RA_mean={ra_mean:.4f} LA_mean={la_mean:.4f} BTI_mean={bti_mean:.4f}")

                    prev_glob_step += config[task]['nstep_outer']

            elif use_rotmnist:
                pkl_path = os.path.join(config['data_dir'], 'rotated_mnist', 'mnist_all_rotation_normalized_train_valid.pkl')
                trainset = RotatedMNISTDataset(pkl_path=pkl_path, split='train', transform=transformation,
                                               device=config['device'], cuda_img_tensor=config['cuda_img_tensor'],
                                               verbose='rotmnist train')
                evalset.append(RotatedMNISTDataset(pkl_path=pkl_path, split='test', transform=transformation,
                                                   device=config['device'], cuda_img_tensor=config['cuda_img_tensor'],
                                                   verbose='rotmnist eval'))

                metatrain_seqdataset_multi_amvi(
                    model=model, var_approx_list=var_approx_list, var_beta=var_beta, model_device=config["device"], 
                    trainset=trainset, evalset=evalset, outer_kl_scale=config[task]['outer_kl_scale'],
                    nstep_outer=config[task]['nstep_outer'], nstep_inner=config[task]['nstep_inner'],
                    lr_inner=config[task]['lr_inner'], first_order=config[task]['first_order'], seqtask=config['seqtask'],
                    num_task_per_itr=config[task]['num_task_per_itr'], task_by_supercls=config[task]['task_by_supercls'],
                    num_way=config['net_kwargs']['num_way'], num_shot=config[task]['num_shot'],
                    num_query_per_cls=config[task]['num_query_per_cls'], eval_prev_task=True,
                    eval_per_num_iter=config[task]['eval_per_num_iter'], num_eval_task=config[task]['num_eval_task'],
                    eval_task_by_supercls=config[task]['eval_task_by_supercls'],
                    nstep_inner_eval=config[task]['nstep_inner_eval'], writer=writer, task_idx=task_idx,
                    prev_glob_step=prev_glob_step, verbose=task, sample_mean=config["sample_mean"], sample_batch=config["sample_batch"],
                    max_eval = config['max_eval']
                )

                final_means = run_eval_block(tag='final_eval', step_idx=prev_glob_step + config[task]['nstep_outer'], task_name=task, evalsets=evalset)
                if final_means:
                    eval_snapshots.append(final_means)
                    diag = [eval_snapshots[i][i] for i in range(len(eval_snapshots))]
                    final_row = eval_snapshots[-1][:len(eval_snapshots)]
                    la_mean = float(np.mean(diag))
                    ra_mean = float(np.mean(final_row))
                    bti_mean = float(np.mean([f - d for f, d in zip(final_row, diag)]))
                    print(f"[METRICS] RA_mean={ra_mean:.4f} LA_mean={la_mean:.4f} BTI_mean={bti_mean:.4f}")

                prev_glob_step += config[task]['nstep_outer'] 

            else:
                split_dir = os.path.join(os.path.join(config['data_dir'], task), config['split_folder'])
                trainset = FewShotImageDataset(
                        task_list=np.load(os.path.join(split_dir, 'metatrain.npy'), allow_pickle=True).tolist(),
                        supercls=config[task]['supercls'], img_lvl=int(config[task]['supercls']) + 1, transform=transformation,
                        relabel=None, device=var_approx.device, cuda_img_tensor=config['cuda_img_tensor'],
                        verbose='{} trainset'.format(task)
                )

                evalset.append(FewShotImageDataset(
                    task_list=np.load(os.path.join(split_dir, 'metatest.npy'), allow_pickle=True).tolist(),
                    supercls=config[task]['eval_supercls'], img_lvl=int(config[task]['eval_supercls']) + 1,
                    transform=transformation, relabel=None, device=config['device'],
                    cuda_img_tensor=config['cuda_img_tensor'], verbose='{} evalset'.format(task)
                ))

                metatrain_seqdataset_multi_amvi(
                    model=model, var_approx_list=var_approx_list, var_beta=var_beta, model_device=config["device"], 
                    trainset=trainset, evalset=evalset, outer_kl_scale=config[task]['outer_kl_scale'],
                    nstep_outer=config[task]['nstep_outer'], nstep_inner=config[task]['nstep_inner'],
                    lr_inner=config[task]['lr_inner'], first_order=config[task]['first_order'], seqtask=config['seqtask'],
                    num_task_per_itr=config[task]['num_task_per_itr'], task_by_supercls=config[task]['task_by_supercls'],
                    num_way=config['net_kwargs']['num_way'], num_shot=config[task]['num_shot'],
                    num_query_per_cls=config[task]['num_query_per_cls'], eval_prev_task=True,
                    eval_per_num_iter=config[task]['eval_per_num_iter'], num_eval_task=config[task]['num_eval_task'],
                    eval_task_by_supercls=config[task]['eval_task_by_supercls'],
                    nstep_inner_eval=config[task]['nstep_inner_eval'], writer=writer, task_idx=task_idx,
                    prev_glob_step=prev_glob_step, verbose=task, sample_mean=config["sample_mean"], sample_batch=config["sample_batch"],
                    max_eval = config['max_eval']
                )

                final_means = run_eval_block(tag='final_eval', step_idx=prev_glob_step + config[task]['nstep_outer'], task_name=task, evalsets=evalset)
                if final_means:
                    eval_snapshots.append(final_means)
                    diag = [eval_snapshots[i][i] for i in range(len(eval_snapshots))]
                    final_row = eval_snapshots[-1][:len(eval_snapshots)]
                    la_mean = float(np.mean(diag))
                    ra_mean = float(np.mean(final_row))
                    bti_mean = float(np.mean([f - d for f, d in zip(final_row, diag)]))
                    print(f"[METRICS] RA_mean={ra_mean:.4f} LA_mean={la_mean:.4f} BTI_mean={bti_mean:.4f}")

                prev_glob_step += config[task]['nstep_outer'] 

            
            # save the previous beta distribution
            var_previous_beta_alpha.append(var_beta.var_gamma1.data.cpu().numpy())
            var_previous_beta_beta.append(var_beta.var_gamma2.data.cpu().numpy())
            

            if var_beta is not None:
                gamma = np.vstack([var_beta.var_gamma1.detach().cpu().numpy(), \
                    var_beta.var_gamma2.detach().cpu().numpy()])
                # print(gamma)
                np.save(os.path.join(result_dir, "alpha{}_nonsparse.npy".format(task_idx)), gamma)

                # rho = var_beta.bern_pro.detach().cpu().numpy()
                # np.save(os.path.join(result_dir, "rho{}.npy".format(task_idx)), rho)

            for i,var_approx in enumerate(var_approx_list):
                torch.save(var_approx.mean, f=os.path.join(result_dir, 'mean{}_varapprox{}__nonsparse.pt'.format(task_idx, i)))
                torch.save(var_approx.covar, f=os.path.join(result_dir, 'covar{}_varapprox{}__nonsparse.pt'.format(task_idx, i)))

            # evidential sparsity
            if config['sparsity'] and len(var_approx_list) != 1:
                sparsity_vec = evidential_sparsity(var_beta, var_previous_beta_alpha, var_previous_beta_beta, config['eta'])
                delete_num = np.sum(np.where(sparsity_vec==0))

                # update the beta
                var_beta.delete_cluster(sparsity_vec)
                config['k'] -= delete_num

                # update the theta
                delete_index = np.argmin(sparsity_vec)
                pop_index = np.where(sparsity_vec==0)[0]
                for i in range(1,len(pop_index)+1):
                    var_approx_list.pop(pop_index[-i])

            # save the model
            torch.save(prev_glob_step, f=os.path.join(result_dir, 'prev_glob_step{}.pt'.format(task_idx)))
            torch.save(evalset, f=os.path.join(result_dir, 'evalset{}.pt'.format(task_idx)))
            for i,var_approx in enumerate(var_approx_list):
                torch.save(var_approx.mean, f=os.path.join(result_dir, 'mean{}_varapprox{}.pt'.format(task_idx, i)))
                torch.save(var_approx.covar, f=os.path.join(result_dir, 'covar{}_varapprox{}.pt'.format(task_idx, i)))
            
            if var_beta is not None:
                gamma = np.vstack([var_beta.var_gamma1.detach().cpu().numpy(), \
                    var_beta.var_gamma2.detach().cpu().numpy()])
                # print(gamma)
                np.save(os.path.join(result_dir, "alpha{}.npy".format(task_idx)), gamma)

                # rho = var_beta.bern_pro.detach().cpu().numpy()
                # np.save(os.path.join(result_dir, "rho{}.npy".format(task_idx)), rho)

            # update mean and covariance of meta-parameters
            var_beta.update_posterior(len(var_approx_list))
            for var_approx in var_approx_list:
                var_approx.update_mean_cov() 

        torch.cuda.empty_cache()

    # check how long it ran
    run_time_print = '\ncompleted in {}'.format(datetime.datetime.now() - start_datetime)
    os.system('echo "{}"'.format(run_time_print)) if system() == 'Linux' else print(run_time_print)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('BOMVI Sequential Dataset')
    parser.add_argument('--config_path', type=str, help='Path of .json file to import config from')
    args = parser.parse_args()
    # load config file
    jsonfile = open(str(args.config_path))
    config = json.loads(jsonfile.read())
    # train
    train(config=config, run_spec=os.path.splitext(os.path.split(args.config_path)[-1])[0], seed=random.getrandbits(24))
    eval_snapshots = []  # store mean accuracies for seen tasks after each task boundary
