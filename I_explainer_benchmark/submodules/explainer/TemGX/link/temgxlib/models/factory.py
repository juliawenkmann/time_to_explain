from temgxlib.common import create_dataset_from_args, create_tgn_wrapper_from_args, create_tgat_wrapper_from_args, create_ttgnn_wrapper_from_args

def build_dataset(args):
    return create_dataset_from_args(args)

def build_wrapper(args, dataset=None):
    if args.type == 'TGN':
        return create_tgn_wrapper_from_args(args, dataset)
    elif args.type == 'TGAT':
        return create_tgat_wrapper_from_args(args, dataset)
    elif getattr(args, 'type', None) in ('TTGN','TTGAT'):
        return create_ttgnn_wrapper_from_args(args, dataset)
    else:
        raise NotImplementedError