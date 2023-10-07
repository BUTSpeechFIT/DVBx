#!/usr/bin/env python3

# @Authors: Dominik Klement (xkleme15@vutbr.cz), Brno University of Technology

from types import SimpleNamespace

from VBx.enums.losses import AvailableLosses
from VBx.enums.backprop_types import AvailableBackpropTypes


def process_lr(args):
    if type(args.lr) == SimpleNamespace:
        lr = vars(args.lr)
        for key in lr:
            lr[key] = float(lr[key])
        final_lr = lr
    elif type(args.lr) == int or type(args.lr) == float:
        if args.lr <= 0:
            raise 'Learning rate has to be a positive number.'

        final_lr = {
            'all': args.lr
        }
    elif type(args.lr) == str:
        try:
            parsed_lr = float(args.lr)
            if parsed_lr <= 0:
                raise 'Learning rate has to be a positive number.'
            final_lr = {
                'all': parsed_lr
            }
        except ValueError:
            raise 'Wrong "lr" value. It should be either a dictionary or a positive number.'

    else:
        raise 'Wrong "lr" value. It should be either a dictionary or a nonzero number.'

    return final_lr


# Validate arguments
def process_args(args):
    if args.avg_last_n_iters != -1 and args.avg_last_n_iters < 0:
        raise '\"avg_last_n_iters\" has to be -1 (averaging all VB iterations) or a positive number.'

    if args.backprop_type == 'after_each_iter':
        args.backprop_type = AvailableBackpropTypes.AFTER_EACH_ITER
    elif args.backprop_type == 'after_convergence':
        args.backprop_type = AvailableBackpropTypes.AFTER_CONVERGENCE
    else:
        raise 'Wrong backprop type. Use after_each_iter or after_convergence.'

    if args.batch_size < 1:
        raise 'Batch size has to be at least 1.'

    if args.eval_max_iters < 1:
        raise 'Number of eval VB iterations has to be at least 1.'

    if args.epochs < 1:
        raise "Number of epochs has to be at least 1."

    if args.initial_loss_scale < 0:
        raise "Initial loss scale has to be positive."

    if args.loss == 'BCE':
        args.loss = AvailableLosses.BCE
    elif args.loss == 'EDE':
        args.loss = AvailableLosses.EDE
    else:
        raise 'Loss not implemented. Use BCE or EDE.'

    # Convert learning rate namespace to dictionary
    args.lr = process_lr(args)

    if args.max_iters < 1:
        raise '"max_iters" should be at least 1.'

    if args.regularization_coeff_eb < 0:
        raise 'Regularization coefficient has to be a non-negative number.'

    if args.regularization_coeff_ew < 0:
        raise 'Regularization coefficient has to be a non-negative number.'

    args.trainable_params = set(args.trainable_params)
