import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from pytorch_metric_learning import losses
from tqdm import tqdm


def add_weight_regularization(model, l1_scale=0.0, l2_scale=0.01):
    l1_reg, l2_reg = 0.0, 0.0

    for parameter in model.parameters():
        if parameter.requires_grad:
            l1_reg += l1_scale * parameter.abs().sum()
            l2_reg += l2_scale * parameter.pow(2).sum()
            
    return l1_reg + l2_reg


def get_loss_by_type(loss_type, loss_t, loss_d, loss_f):
    loss_dict = {
        'ALL': loss_t + loss_d + loss_f,
        'TDF': loss_t + loss_d + loss_f,
        'TD': loss_t + loss_d,
        'TF': loss_t + loss_f,
        'DF': loss_d + loss_f,
        'T': loss_t,
        'D': loss_d,
        'F': loss_f
    }
    if loss_type not in loss_dict:
        raise ValueError(f"Invalid loss type: {loss_type}")
    return loss_dict[loss_type]


def train(args, encoder, clf, encoder_optimizer, clf_optimizer, loader, mode='pretrain', device='cuda'):
    encoder.train() if mode != 'freeze' else encoder.eval() 
    clf.train() if mode != 'pretrain' else None

    if mode == 'pretrain':
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad = True
        # clf.eval()
    elif mode == 'finetune':
        encoder.train()
        for param in encoder.parameters():
            param.requires_grad = True
        clf.train()
    elif mode == 'freeze':
        encoder.eval()
        for name, param in encoder.named_parameters():
            if 'input_layer' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        clf.train()
    
    scaler = GradScaler()
    
    info_loss = losses.NTXentLoss(temperature=args.temperature)
    info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_loss_c = 0
    total_samples = 0
    
    pbar = tqdm(loader, desc=f"Training ({mode})")
    for batch in pbar:
        xt, dx, xf, y = batch      
        xt     = xt.float().to(device, non_blocking=True)
        dx     = dx.float().to(device, non_blocking=True)
        xf     = xf.float().to(device, non_blocking=True)
        y      = y.to(device, non_blocking=True) 
        if mode == 'pretrain':
            with torch.no_grad():
                xt_aug = gpu_data_transform_td(xt.clone())
                dx_aug = gpu_data_transform_td(dx.clone())
                xf_aug = gpu_data_transform_fd(xf.clone())
        else:
            with torch.no_grad():
                xt_aug = gpu_data_transform_td(xt.clone()) 
                dx_aug = gpu_data_transform_td(dx.clone())
                xf_aug = gpu_data_transform_fd(xf.clone())
        
        encoder_optimizer.zero_grad()
        if mode != 'pretrain':
            clf_optimizer.zero_grad()
        
        with autocast(enabled=True):
            ht, hd, hf, zt, zd, zf = encoder(xt, dx, xf)
            ht_aug, hd_aug, hf_aug, zt_aug, zd_aug, zf_aug = encoder(xt_aug, dx_aug, xf_aug)
            loss_t = info_criterion(zt, zt_aug)
            loss_d = info_criterion(zd, zd_aug)
            loss_f = info_criterion(zf, zf_aug)
            
            loss_cl = get_loss_by_type(args.loss_type, loss_t, loss_d, loss_f)
            reg_e   = add_weight_regularization(encoder)

            if mode == 'pretrain':
                # Only contrastive + encoder reg
                loss = loss_cl + reg_e
            else:
                # Finetune / baseline / freeze
                logit = clf(zt, zd, zf) if args.feature == 'latent' else clf(ht, hd, hf)
                loss_c = criterion(logit, y.long())
                reg_c  = add_weight_regularization(clf)

                loss = args.lam * (loss_cl+ reg_e) + loss_c  + reg_c

        scaler.scale(loss).backward()
        
        # # Gradient clipping
        # torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=10.0)
        # if mode != 'pretrain':
        #     torch.nn.utils.clip_grad_norm_(clf.parameters(), max_norm=10.0)

        if mode != 'freeze':
            scaler.step(encoder_optimizer)
        if mode != 'pretrain':
            scaler.step(clf_optimizer)
        
        scaler.update()

        total_loss += loss.item() * xt.size(0)
        if mode != 'pretrain':
            total_loss_c += loss_c.item() * xt.size(0)
        total_samples += xt.size(0)
        
        pbar.set_postfix({'loss': loss.item(), 'loss_t': loss_t.item(), 'loss_d': loss_d.item(), 'loss_f': loss_f.item()})
    
    avg_loss = total_loss / total_samples
    avg_loss_c = total_loss_c / total_samples
    
    if mode == 'pretrain':
        return avg_loss
    else:
        return avg_loss, avg_loss_c        


def test(args, encoder, clf, loader, mode='pretrain', device='cuda'):
    encoder.eval()
    clf.eval() if mode != 'pretrain' else None

    info_loss = losses.NTXentLoss(temperature=args.temperature)
    info_criterion = losses.SelfSupervisedLoss(info_loss, symmetric=True)
    criterion = nn.CrossEntropyLoss()
    
    total_loss = 0
    total_loss_c = 0
    total_samples = 0
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Testing ({mode})")
        for batch in pbar:      
            xt, dx, xf, y = batch
            xt     = xt.float().to(device, non_blocking=True)
            dx     = dx.float().to(device, non_blocking=True)
            xf     = xf.float().to(device, non_blocking=True)
            y      = y.to(device, non_blocking=True) 
            if mode == 'pretrain':
                xt_aug = gpu_data_transform_td(xt.clone())
                dx_aug = gpu_data_transform_td(dx.clone())
                xf_aug = gpu_data_transform_fd(xf.clone())

                with autocast(enabled=True):
                    ht, hd, hf, zt, zd, zf = encoder(xt, dx, xf)
                    ht_aug, hd_aug, hf_aug, zt_aug, zd_aug, zf_aug = encoder(xt_aug, dx_aug, xf_aug)
                    
                    loss_t = info_criterion(zt, zt_aug)
                    loss_d = info_criterion(zd, zd_aug)
                    loss_f = info_criterion(zf, zf_aug)
                    
                    loss = get_loss_by_type(args.loss_type, loss_t, loss_d, loss_f) + add_weight_regularization(encoder)
            else:
                with autocast(enabled=True):
                    ht, hd, hf, zt, zd, zf = encoder(xt, dx, xf)
                    logit = clf(zt, zd, zf) if args.feature == 'latent' else clf(ht, hd, hf)
                    loss_c = criterion(logit, y.long())
                    loss = loss_c + add_weight_regularization(clf)

            total_loss += loss.item() * xt.size(0)
            if mode != 'pretrain':
                total_loss_c += loss_c.item() * xt.size(0)
            total_samples += xt.size(0)
            
            # pbar.set_postfix({'loss': loss.item(), 'loss_t': loss_t.item(), 'loss_d': loss_d.item(), 'loss_f': loss_f.item()})
            pbar.set_postfix({'loss': loss.item()})

    
    avg_loss = total_loss / total_samples
    avg_loss_c = total_loss_c / total_samples
    
    if mode == 'pretrain':
        return avg_loss
    else:
        return avg_loss, avg_loss_c 


## Pretrained model loader
def remove_module_prefix(state_dict):
    # Prevent multi-gpu -> single-gpu errors
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]  # Remove the first 7 characters ('module.')
        elif key.startswith('_orig_mod.'):
            new_key = key[10:]  # Remove the first 7 characters ('module.')
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
@torch.jit.script # JIT Compile this function
def gpu_data_transform_td(batch: torch.Tensor, sigma: float = 0.1) -> torch.Tensor:
    # Generates noise for the entire batch at once on the GPU
    return batch + torch.normal(mean=0., std=sigma, size=batch.shape, device=batch.device)
@torch.jit.script # JIT Compile this function
def gpu_data_transform_fd(batch: torch.Tensor, pertub_ratio: float = 0.05) -> torch.Tensor:
    # Masking for frequency domain
    # 1. Remove frequency
    mask_remove = torch.rand(batch.shape, device=batch.device) > pertub_ratio
    aug_1 = batch * mask_remove
    
    # 2. Add frequency
    mask_add = torch.rand(batch.shape, device=batch.device) > (1 - pertub_ratio)
    max_amplitude = batch.amax(dim=(1, 2), keepdim=True) # efficient batch max
    random_am = torch.rand(mask_add.shape, device=batch.device) * (max_amplitude * 0.1)
    aug_2 = batch + (mask_add * random_am)
    
    return aug_1 + aug_2


def load_encoder(encoder, checkpoint_path, new_num_feature=None):
    print(f"\n=== load_encoder: loading from {checkpoint_path} ===")
    # Load raw checkpoint
    state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # If it's a full checkpoint, extract encoder state
    if isinstance(state_dict, dict) and 'encoder_state_dict' in state_dict:
        print("Found 'encoder_state_dict' key in checkpoint.")
        state_dict = state_dict['encoder_state_dict']
    else:
        print("Using raw state_dict (no 'encoder_state_dict' key).")

    # Show some example keys
    print("Example keys (before prefix removal):")
    for k in list(state_dict.keys())[:10]:
        print("  ", k)

    # Remove 'module.' prefix if present
    state_dict = remove_module_prefix(state_dict)

    # DEBUG: show input_layer shapes from checkpoint
    print("Input layer keys & shapes (from checkpoint):")
    for k in state_dict.keys():
        if 'input_layer' in k and ('weight' in k or 'bias' in k):
            print(f"  {k}: shape={tuple(state_dict[k].shape)}")

    # Handle new_num_feature (if needed)
    if new_num_feature is not None:
        input_layers = ['input_layer_t', 'input_layer_d', 'input_layer_f']
        for layer_name in input_layers:
            w_key = f'{layer_name}.weight'
            b_key = f'{layer_name}.bias'
            if w_key not in state_dict or b_key not in state_dict:
                print(f"[WARN] {w_key} or {b_key} missing in checkpoint; cannot remap num_feature.")
                continue

            old_weight = state_dict[w_key]
            old_bias = state_dict[b_key]
            old_num_feature = old_weight.size(1)

            if new_num_feature != old_num_feature:
                print(f"[INFO] Remapping {layer_name}: old_num_feature={old_num_feature} -> new_num_feature={new_num_feature}")
                new_linear = nn.Linear(new_num_feature, old_weight.size(0))
                nn.init.xavier_uniform_(new_linear.weight)
                state_dict[w_key] = new_linear.weight.data
                state_dict[b_key] = torch.zeros_like(old_bias)
            else:
                print(f"[INFO] {layer_name}: num_feature unchanged ({old_num_feature}).")

    # Clean NaN/Inf
    for key, param in state_dict.items():
        if torch.is_tensor(param) and (torch.isnan(param).any() or torch.isinf(param).any()):
            print(f"[WARN] Found NaN/Inf in {key}, replacing with zeros.")
            state_dict[key] = torch.nan_to_num(param, nan=0.0, posinf=0.0, neginf=0.0)

    # Remove q_func keys
    keys_to_remove = [key for key in state_dict.keys() if 'q_func' in key]
    for key in keys_to_remove:
        print(f"[INFO] Removing key from checkpoint: {key}")
        state_dict.pop(key)
    
    encoder.load_state_dict(state_dict, strict=False)

    return encoder

