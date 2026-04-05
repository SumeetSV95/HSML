# run_continual_meta.py
import importlib
import torch
import numpy as np
import random

from non_stationary_trainer import ContinualMetaTrainer

# Import the centralized parser
from parser import get_parser  # expects a function that returns an ArgumentParser
from utils import misc_utils
# def set_seed(seed: int):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     if torch.cuda.is_available():
#         torch.cuda.manual_seed_all(seed)

def main():
    parser = get_parser()
    args = parser.parse_args()

    # set_seed(getattr(args, "seed", 0))

    # initialize seeds
    misc_utils.init_seed(args.seed)

    # setup logging
    timestamp = misc_utils.get_date_time()
    args.log_dir, args.tf_dir = misc_utils.log_dir(args, timestamp)
    # load model

    args.context_size = args.n_way * args.k_shot  # context size for VERSA
    args.batch_size = args.q_query  # total batch size per task

    Model = importlib.import_module('model.' + args.model)
    model = Model.Net(args.n_input, args.n_way, args.num_task_per_iter, args)
    # memory = TaskRelationalReservoir(args)
    # model.memory = memory  # if the model expects this linkage

    trainer = ContinualMetaTrainer(
        model=model,
        # memory=memory,
        args=args,
        data_root=args.data_root,  
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        epochs_per_dataset=args.epochs_per_dataset,
        device=args.device,
        known_boundary=args.known_boundary,
    )

    trainer.train()

if __name__ == "__main__":
    main()