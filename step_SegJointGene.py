import copy
import cv2

from utils import *
from SegJointGene.global_stitcher import GlobalStitchingEvaluator
from SegJointGene.update_label import update_label
from SegJointGene.get_gene_celltype import get_gene_celltype
from SegJointGene.compute_CID import compute_CID
from SegJointGene.compute_IG import compute_IG

def step_SegJointGene(root_path, args):
    # set
    assert args.net_sub_suffix.startswith("SegJointGene")
    if '_' in args.net_sub_suffix:
        suffix = args.net_sub_suffix.split('_')[1]
        assert args.attr_method == suffix
    else:
        args.net_sub_suffix = args.net_sub_suffix + '_' + args.attr_method
    path_dict, step_suffix = step_set_path(root_path, args)
    setup_seed(args.random_seed)

    # data
    train_set, test_set, train_loader, test_loader = step_get_datasets_loader(path_dict, args)

    # gene celltype
    target_gene, target_celltype, target_gene_names, target_celltype_names = get_gene_celltype(root_path, args, train_loader)

    # model
    net = get_net(net_name='unet', args=args)
    net = step_set_seed(path_dict, net)
    optimizer = step_get_optimizer(net, args)
    criterion = nn.CrossEntropyLoss().cuda()
    logger = []
    net, optimizer, logger, start_epoch = step_load_ckpt(path_dict, net, optimizer, logger, args, if_load=args.if_load_ckpt)
    if args.if_load_ckpt:
        step_load_label_cache(path_dict, start_epoch, train_set, test_set)

    for epoch_id in range(start_epoch, args.net_epoch + 1):
        # Stitcher Init, do not take attrs into stitcher unless for visualization
        stitcher = GlobalStitchingEvaluator(
            test_set=test_set, train_set=train_set,
            patch_size=args.patch_size, pixel_distance=args.pixel_distance,
            target_gene=target_gene, target_celltypes=target_celltype,
            target_gene_names=target_gene_names, target_celltype_names=target_celltype_names, if_attr=False
        )


        # train
        run_attr = (epoch_id >= args.attr_epoch)
        net.train()
        for batch_id, batch_data in enumerate(train_loader):
            image, label, instance_label, spots, dapi, idx, rows, cols, fixed_label, fixed_inst = batch_data
            image, label = image.cuda(), label.cuda().long()
            spots, instance_label = spots.cuda(), instance_label.cuda()
            fixed_inst = fixed_inst.cuda().long()

            output = net(image)
            loss = criterion(output, label)
            prediction = torch.argmax(output, dim=1)
            acc = (prediction == label).sum().item() / label.numel()

            if epoch_id != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if run_attr:
                if args.attr_method == 'CID':
                    net, attr = compute_CID(net, image, target_gene, target_celltype,
                                            n_steps=args.CID_n_steps, lr=args.CID_lr,
                                            lambda_param=args.CID_lambda_param, beta=args.CID_beta,
                                            cell_chunk_size=args.CID_cell_chunk_size,
                                            gene_chunk_size=args.CID_gene_chunk_size, return_spatial=True)
                elif args.attr_method == 'IG':
                    attr = compute_IG(image, net, target_gene, target_celltype)
                else:
                    attr = None
            else:
                attr = None

            if epoch_id == 0:
                new_label, new_inst = copy.deepcopy(label), copy.deepcopy(instance_label)
            else:
                new_label, new_inst = update_label(label, instance_label, (fixed_inst > 0), output, epoch_id, batch_id,
                                                   attributions=attr, target_celltype=target_celltype)
            train_set.update_label_cache(idx, new_label, new_inst)
            stitcher.update(label, spots, rows, cols, attributions=None)
            step_print_SegJointGene(batch_id, epoch_id, loss.data.cpu(), acc, num_batch=len(train_loader), if_train=True)

        # test
        for batch_id, batch_data in enumerate(test_loader):
            image, label, instance_label, spots, dapi, idx, rows, cols, fixed_label, fixed_inst = batch_data
            image, label = image.cuda(), label.cuda().long()
            spots, instance_label = spots.cuda(), instance_label.cuda()
            fixed_inst = fixed_inst.cuda().long()

            net.eval()
            with torch.no_grad():
                output = net(image)
                loss = criterion(output, label)
                prediction = torch.argmax(output, dim=1)
                acc = (prediction == label).sum().item() / label.numel()

            if run_attr:
                if args.attr_method == 'CID':
                    net, attr = compute_CID(net, image, target_gene, target_celltype,
                                            n_steps=args.CID_n_steps, lr=args.CID_lr,
                                            lambda_param=args.CID_lambda_param, beta=args.CID_beta,
                                            cell_chunk_size=args.CID_cell_chunk_size,
                                            gene_chunk_size=args.CID_gene_chunk_size, return_spatial=True)
                elif args.attr_method == 'IG':
                    attr = compute_IG(image, net, target_gene, target_celltype)
                else:
                    attr = None
            else:
                attr = None

            if epoch_id == 0:
                new_label, new_inst = copy.deepcopy(label), copy.deepcopy(instance_label)
            else:
                new_label, new_inst = update_label(label, instance_label, (fixed_inst > 0), output, epoch_id, batch_id,
                                                   attributions=attr, target_celltype=target_celltype)
            test_set.update_label_cache(idx, new_label, new_inst)
            stitcher.update(new_label, spots, rows, cols, attributions=None)
            step_print_SegJointGene(batch_id, epoch_id, loss.data.cpu(), acc, num_batch=len(test_loader), if_train=False)

        # --- End Epoch ---
        cell_calling_score = stitcher.compute_score()
        print(f'\n[Epoch {epoch_id}] Cell calling score: {cell_calling_score}')

        # save
        if epoch_id % 10 == 0:
            step_save_ckpt(path_dict, epoch_id, net, optimizer, logger, args)
            step_save_label_cache(path_dict, epoch_id, train_set, test_set)