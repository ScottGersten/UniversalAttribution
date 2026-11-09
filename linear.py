from utils.logger import create_logger

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from utils.evaluation import evaluate_multiclass
import numpy as np
import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import argparse
from torch.utils.data import TensorDataset
import json
import pandas as pd

from utils.common import get_train_paths, load_config
from utils.config import parse_data_str
from utils.data.dataset import BaseData
from utils.evaluation import metric_ood, compute_oscr
from utils.feature_extraction import extract_features
from models.logreg import LogRegModule
from models.gendet_modules import GenDetAugmenter, TeacherStudentGate


def parse_args():
    parser = argparse.ArgumentParser(description='Logistic Regression experiment with GenDet')
    parser.add_argument('-c', '--config_name', type=str, help='model configuration file')
    parser.add_argument('-d', '--data', type=str, dest="dataset_str")
    parser.add_argument('--device', type=str, help='cuda:n or cpu')
    parser.add_argument('-o', '--output_dir', type=str, default='output')
    parser.add_argument('--max_train_iters', type=int)
    parser.add_argument('-b', '--blocks', nargs='+', type=int, help='blocks to extract features from')
    parser.add_argument('--backbone', type=str)
    parser.add_argument('--c_min', type=int, help='range of C values to sweep', default=-6)
    parser.add_argument('--c_max', type=int, help='range of C values to sweep', default=5)
    parser.add_argument('--c_step', type=int, help='range of C values to sweep', default=45)
    parser.add_argument('--C', type=float, help='C value for linear classifier')
    parser.add_argument('--gendet_epochs', type=int, default=10, help='training epochs for GenDet modules')
    parser.add_argument('--gendet_lr', type=float, default=1e-4, help='learning rate for GenDet modules')
    parser.add_argument('--disable_gendet', action='store_true', help='disable GenDet pipeline and run baseline')

    parser.set_defaults(
        config_name='base',
        dataset_str='GenImage:split=split1',
        device='cuda:0',
        output_dir='output/linear-probe/',
        train_backbone=False,
        debug=False,
        max_train_iters=1000,
        blocks=None,
        backbone='vit_base_patch16_clip_224.openai'
    )

    args = parser.parse_args()
    return args


def train_gendet_pipeline(
    train_features,
    train_labels,
    real_class_idx=0,
    device="cuda:0",
    epochs=10,
    lr=1e-4,
    batch_size=128
):
    """
    Trains the GenDet Augmenter and Teacher-Student Gate in feature space.
    Returns:
        hardened_train_features: Tensor of original + warped synthetic features
        hardened_train_labels: Tensor of corresponding labels
        ts_gate: Trained TeacherStudentGate module
    """
    dim = train_features.shape[1]
    augmenter = GenDetAugmenter(dim=dim).to(device)
    ts_gate = TeacherStudentGate(dim=dim).to(device)

    aug_optimizer = torch.optim.Adam(augmenter.parameters(), lr=lr)
    student_optimizer = torch.optim.Adam(ts_gate.student.parameters(), lr=lr)

    dataset = TensorDataset(train_features, train_labels)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for feats, labels in loader:
            feats = feats.to(device)
            labels = labels.to(device)
            is_real = (labels == real_class_idx)
            is_fake = ~is_real

            # --- 1. Train Teacher-Student Gate ---
            student_optimizer.zero_grad()
            discrepancy, _, _ = ts_gate(feats)

            # Real minimizes discrepancy; Synthetic maximizes discrepancy
            loss_real = discrepancy[is_real].mean() if is_real.any() else torch.tensor(0.0, device=device)
            loss_fake = torch.clamp(2.0 - discrepancy[is_fake], min=0.0).mean() if is_fake.any() else torch.tensor(0.0, device=device)
            ts_loss = loss_real + loss_fake
            ts_loss.backward()
            student_optimizer.step()

            # --- 2. Train Augmenter (Adversarial on Synthetic features) ---
            if is_fake.any():
                aug_optimizer.zero_grad()
                warped_fake = augmenter(feats[is_fake])
                warped_disc, _, _ = ts_gate(warped_fake)
                # Augmenter tries to minimize discrepancy to fool the gate
                aug_loss = warped_disc.mean()
                aug_loss.backward()
                aug_optimizer.step()

    # --- 3. Synthesize Hardened Feature Set for the Linear Probe ---
    augmenter.eval()
    with torch.no_grad():
        fake_mask = (train_labels != real_class_idx)
        fake_feats = train_features[fake_mask].to(device)
        fake_labels = train_labels[fake_mask]

        warped_feats = augmenter(fake_feats).cpu()

        hardened_train_features = torch.cat([train_features, warped_feats], dim=0)
        hardened_train_labels = torch.cat([train_labels, fake_labels], dim=0)

    return hardened_train_features, hardened_train_labels, ts_gate


def sweep_C_values(
        train_features,
        train_labels,
        test_data_loader,
        out_data_loader,
        max_train_iters,
        results_path="",
        metric="AUC",
        c_min=-6,
        c_max=5,
        c_step=45,
        ts_gate=None
):
    best_stats = None
    best_C = None

    C_POWER_RANGE = np.linspace(c_min, c_max, c_step)
    ALL_C = 10 ** C_POWER_RANGE
    all_stats = {}

    for C in ALL_C:
        C = C.item()
        logger.info(f"Training with C={C}")
        model = LogRegModule(C, max_iter=max_train_iters, device=opt.device)
        model.fit(train_features, train_labels)
        stats = evaluate_model(
            model=model,
            test_data_loader=test_data_loader,
            out_data_loader=out_data_loader,
            config=config,
            ts_gate=ts_gate
        )
        all_stats[C] = stats
        if best_stats is None or stats[metric] > best_stats[metric]:
            best_stats = stats
            best_C = C

    with open(results_path, 'a') as f:
        f.write(json.dumps(all_stats))
        f.write('\n')
    return best_stats, best_C


def predict_set(
        model,
        data_loader,
        run_type="test",
):
    all_preds, all_targets = [], []
    for data, targets in data_loader:
        with torch.no_grad():
            outputs = model(data, targets)
            all_preds.append(outputs["preds"])
            all_targets.append(outputs["target"])
    all_preds = torch.cat(all_preds, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    if 'open-set' in run_type:
        return all_targets.numpy(), all_preds.numpy()
    else:
        results = evaluate_multiclass(all_targets, all_preds.argmax(dim=1))
        CM = confusion_matrix(all_targets, all_preds.argmax(dim=1))
        perf = round(results['accuracy'], 4) * 100
        logger.info('%s results: %s' % (run_type, str(results)))
        logger.info('%s confusion matrix: %s' % (run_type, str(CM)))
        return all_targets.numpy(), all_preds.numpy(), perf


def evaluate_model(
        model,
        test_data_loader,
        out_data_loader,
        config,
        ts_gate=None
):
    unknown_classes = config.unknown_classes['all']
    in_targets, in_preds, closed_results = predict_set(model, test_data_loader, run_type='val')
    out_targets, out_preds = predict_set(model, out_data_loader, run_type='open-set')

    # If TS Gate is present, replace softmax maximum with inverted discrepancy score
    if ts_gate is not None:
        device = next(ts_gate.parameters()).device

        in_scores = []
        for feats, _ in test_data_loader:
            scores = ts_gate.get_anomaly_score(feats.to(device)).cpu().numpy()
            in_scores.append(scores)
        x1 = -np.concatenate(in_scores, axis=0)

        out_scores = []
        for feats, _ in out_data_loader:
            scores = ts_gate.get_anomaly_score(feats.to(device)).cpu().numpy()
            out_scores.append(scores)
        x2 = -np.concatenate(out_scores, axis=0)
    else:
        x1, x2 = np.max(in_preds, axis=1), np.max(out_preds, axis=1)

    out_results = metric_ood(x1, x2)
    oscr_score = compute_oscr(in_preds, out_preds, in_targets)
    logger.info('OSCR: %.4f' % oscr_score['oscr'])
    out_result_details = {}
    for i, label_u in enumerate(np.unique(out_targets)):
        pred_u = out_preds[out_targets == label_u]

        if ts_gate is not None:
            pred_disc = x2[out_targets == label_u]
            result = metric_ood(x1, pred_disc, verbose=False)['Bas']
            avg_score = -np.mean(pred_disc)
        else:
            result = metric_ood(x1, np.max(pred_u, axis=1), verbose=False)['Bas']
            avg_score = np.mean(np.max(pred_u, axis=1))

        pred = np.argmax(pred_u, axis=1)
        pred_labels = list(set(pred))
        pred_nums = [np.sum(pred == p) for p in pred_labels]
        logger.info("{}\t \t mostly pred class: {}\t \t average score: {}\t AUROC (%): {:.2f}".format(
            unknown_classes[i],
            config.known_classes[pred_labels[np.argmax(pred_nums)]],
            avg_score,
            result['AUROC']
        ))
        out_result_details[str(i)] = {
            'unknown_class': '\t' + unknown_classes[i],
            'pred_class': '\t' + config.known_classes[pred_labels[np.argmax(pred_nums)]],
            'average_score': '\t' + str(round(avg_score, 4)),
            'AUROC': '\t' + str(round(result['AUROC'], 2))
        }

    return {
        'AUC': out_results['Bas']['AUROC'],
        'OSCR': oscr_score['oscr'] * 100,
        'accuracy': closed_results,
        'out_results_details': out_result_details
    }


if __name__ == '__main__':
    opt = parse_args()

    # Load configuration
    config = load_config('configs.{}'.format(opt.config_name))
    run_dir = os.path.join(opt.output_dir, f'{opt.backbone}')
    os.makedirs(run_dir, exist_ok=True)
    logger = create_logger(run_dir, log_name='train.log')
    logger.info(f'logging to {run_dir}')

    # Arguments
    max_train_iters = opt.max_train_iters
    c = opt.C

    # Create model
    model = timm.create_model(opt.backbone, pretrained=True, num_classes=0, global_pool='')
    model = model.to(opt.device)
    model.eval()
    model_cfg = resolve_data_config(model.pretrained_cfg, model=model)
    transform = create_transform(**model_cfg, is_training=False)
    if 'input_size' in model_cfg:
        input_size = model_cfg['input_size']
    else:
        input_size = (3, 224, 224)
    logger.debug(f"Input size: {input_size}")
    _, sample_out = model.forward_intermediates(torch.randn(1, *input_size).to(opt.device))
    logger.debug(f"Model output shape: {sample_out[0].shape}")
    num_blocks = len(sample_out)
    logger.debug(f"Model has {num_blocks} blocks")
    if opt.blocks is None:
        blocks = list(range(num_blocks))
    else:
        assert all([block < num_blocks for block in opt.blocks])
        blocks = opt.blocks

    # Load data configuration
    data_list, kwargs = parse_data_str(opt.dataset_str)
    print(kwargs)
    train_data_path, val_data_path = get_train_paths(data_list)
    test_data_path, out_data_paths = data_list['test_data_path'], data_list['out_data_paths']

    # Set configuration variables for the dataset
    config.known_classes = data_list['known_classes']
    config.unknown_classes = data_list['unknown_classes']
    config.class_num = len(config.known_classes)

    batch_size = config.batch_size
    logger.debug('config.class_num', config.class_num)
    Data = BaseData(train_data_path, val_data_path,
                    test_data_path, out_data_paths,
                    opt, config, transform)

    tokens = opt.dataset_str.split(":")
    name = tokens[0]
    kwargs = {}

    for token in tokens[1:]:
        key, value = token.split("=")
        assert key in ("root", "extra", "split")
        kwargs[key] = value
    filename = os.path.join(run_dir, f'{name}_{kwargs["split"]}.json')
    if os.path.exists(filename):
        os.remove(filename)

    start = time.time()

    for block in blocks:
        # features
        train_features, train_labels = extract_features(Data.train_loader, model, block, device=opt.device)
        val_features, val_labels = extract_features(Data.val_loader, model, block, device=opt.device)
        out_features, out_labels = extract_features(Data.out_loaders['all'], model, block, device=opt.device)

        logger.debug(f"Features shape: {train_features.shape}")

        # GenDet pipeline hook
        ts_gate = None
        probe_train_features = train_features
        probe_train_labels = train_labels

        if not opt.disable_gendet:
            logger.info(f"Training GenDet Augmenter and Teacher-Student Gate for Block {block}...")
            real_idx = config.known_classes.index('real') if 'real' in config.known_classes else 0
            probe_train_features, probe_train_labels, ts_gate = train_gendet_pipeline(
                train_features=train_features,
                train_labels=train_labels,
                real_class_idx=real_idx,
                device=opt.device,
                epochs=opt.gendet_epochs,
                lr=opt.gendet_lr
            )
            logger.info(f"Augmented training pool: {train_features.shape[0]} -> {probe_train_features.shape[0]} vectors")

        val_data_loader = torch.utils.data.DataLoader(
            TensorDataset(val_features, val_labels),
            batch_size=batch_size,
            drop_last=False,
        )
        out_data_loader = torch.utils.data.DataLoader(
            TensorDataset(out_features, out_labels),
            batch_size=batch_size,
            drop_last=False,
        )

        if len(train_labels.shape) > 1:
            num_classes = train_labels.shape[1]
        else:
            num_classes = train_labels.max() + 1

        if c is None:
            best_stats, best_C = sweep_C_values(
                probe_train_features,
                probe_train_labels,
                val_data_loader,
                out_data_loader,
                max_train_iters,
                results_path=filename,
                c_min=opt.c_min,
                c_max=opt.c_max,
                c_step=opt.c_step,
                ts_gate=ts_gate
            )
            c = best_C

        test_out_features, test_out_labels = extract_features(Data.out_loaders['test_all'], model, block, device=opt.device)
        test_features, test_labels = extract_features(Data.test_loader, model, block, device=opt.device)

        test_out_data_loader = torch.utils.data.DataLoader(
            TensorDataset(test_out_features, test_out_labels),
            batch_size=batch_size,
            drop_last=False,
        )
        test_data_loader = torch.utils.data.DataLoader(
            TensorDataset(test_features, test_labels),
            batch_size=batch_size,
            drop_last=False,
        )

        clf = LogRegModule(c, max_iter=max_train_iters, device=opt.device)
        clf.fit(probe_train_features, probe_train_labels)
        stats = evaluate_model(
            model=clf,
            test_data_loader=test_data_loader,
            out_data_loader=test_out_data_loader,
            config=config,
            ts_gate=ts_gate
        )

        out_details = stats.get('out_results_details', stats.get('out_result_details', {}))
        if out_details:
            df = pd.DataFrame(out_details)
            data = df.values
            data = list(map(list, zip(*data)))
            data = pd.DataFrame(data)
            data.to_csv(os.path.join(run_dir, f'block-{block}_result_details.csv'), header=0)
        logger.info(f"Block {block} results: {stats}")