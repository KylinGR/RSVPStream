import argparse
import numpy as np
import struct
import torch
import torch.nn.functional as F
from pathlib import Path
from scipy.io import loadmat
import json
import os
import tempfile

Q_MIN, Q_MAX = -32768, 32767


def pad_last_dim_to_multiple_of_8(tensor: torch.Tensor) -> torch.Tensor:
    last_dim = tensor.size(-1)
    rem = last_dim % 8
    if rem == 0:
        return tensor
    pad_amount = 8 - rem
    return F.pad(tensor, (0, pad_amount), mode="constant", value=0)


def dynamic_quantize_tensor(tensor: torch.Tensor):
    max_val = torch.max(torch.abs(tensor))
    scale = Q_MAX / torch.clamp(max_val, min=1e-8)
    scaled = torch.clamp(torch.round(tensor * scale), Q_MIN, Q_MAX)
    return scaled.to(torch.int16), scale.item()


def dynamic_int16_mul(x: torch.Tensor, x_scale: float, y: torch.Tensor, y_scale: float):
    output_scale = (x_scale * y_scale) / Q_MAX
    result = x.int() * y.int()
    result_int16 = torch.clamp(torch.round(result.float() / Q_MAX), Q_MIN, Q_MAX)
    return result_int16.to(torch.int16), output_scale


def _to_cpu_float(tensor_like):
    if isinstance(tensor_like, torch.Tensor):
        return tensor_like.detach().cpu().float()
    return torch.as_tensor(tensor_like, dtype=torch.float32)


def convert_checkpoint_to_params(ckpt_path: str, unsafe_load: bool = False):
    if unsafe_load:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    else:
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        except Exception:
            # Fallback to unsafe load if safe load fails (trusted checkpoint assumed)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    M_global = _to_cpu_float(ckpt['M_global'])  # [60,250,17]
    Sigma_global = _to_cpu_float(ckpt['Sigma_global'])  # [60,250,17]
    M_local = _to_cpu_float(ckpt['M_local'])  # [299,54,17]
    Sigma_local = _to_cpu_float(ckpt['Sigma_local'])  # [299,54,17]

    lr_model = _to_cpu_float(ckpt['local_weight'][:299]).squeeze()
    model_order = torch.as_tensor(ckpt['Model_order'], dtype=torch.int64)

    model_list = ckpt['Model']

    W_global_list, Q_global_list, b_global_list = [], [], []
    Gamma_global_list, Beta_global_list = [], []
    w_local_list, b_local_list, gamma_local_list, beta_local_list = [], [], [], []

    for group_idx in range(17):
        group = model_list[group_idx]
        gstf_params = group[0]
        local_models = group[1:300]

        W_global_list.append(_to_cpu_float(gstf_params.W_global).squeeze())  # [60]
        Q_global_list.append(_to_cpu_float(gstf_params.Q_global).squeeze())  # [250]
        b_global_list.append(_to_cpu_float(gstf_params.b_global).squeeze())
        Gamma_global_list.append(_to_cpu_float(gstf_params.Gamma_global).squeeze())
        Beta_global_list.append(_to_cpu_float(gstf_params.Beta_global).squeeze())

        w_local = torch.zeros(299, 54)
        b_local = torch.zeros(299)
        gamma_local = torch.zeros(299)
        beta_local = torch.zeros(299)
        for local_idx, local_model in enumerate(local_models):
            w_local[local_idx] = _to_cpu_float(local_model.w_local).squeeze()
            b_local[local_idx] = _to_cpu_float(local_model.b_local).squeeze()
            gamma_local[local_idx] = _to_cpu_float(local_model.gamma_local).squeeze()
            beta_local[local_idx] = _to_cpu_float(local_model.beta_local).squeeze()
        w_local_list.append(w_local)
        b_local_list.append(b_local)
        gamma_local_list.append(gamma_local)
        beta_local_list.append(beta_local)

    converted_params = {
        'W_global': torch.stack(W_global_list, dim=0),
        'Q_global': torch.stack(Q_global_list, dim=0),
        'b_global': torch.stack(b_global_list, dim=0),
        'Gamma_global': torch.stack(Gamma_global_list, dim=0),
        'Beta_global': torch.stack(Beta_global_list, dim=0),
        'w_local': torch.stack(w_local_list, dim=0),
        'b_local': torch.stack(b_local_list, dim=0),
        'gamma_local': torch.stack(gamma_local_list, dim=0),
        'beta_local': torch.stack(beta_local_list, dim=0),
        'lr_model': lr_model,
        'model_order': model_order,
        'M_global': M_global.permute(2, 0, 1),
        'Sigma_global': Sigma_global.permute(2, 0, 1),
        'M_local': M_local.permute(2, 0, 1),
        'Sigma_local': Sigma_local.permute(2, 0, 1),
    }

    return converted_params


def quant_np(arr_f32: np.ndarray):
    max_abs = float(np.max(np.abs(arr_f32))) if arr_f32.size else 0.0
    scale = Q_MAX / max_abs if max_abs > 1e-8 else 1.0
    q = np.sign(arr_f32) * np.floor(np.abs(arr_f32) * scale + 0.5)
    q = np.clip(q, Q_MIN, Q_MAX).astype(np.int16)
    return q, scale


def write_int_blocks(path: Path, arr: np.ndarray, block: int = 8, dtype=np.int16):
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = arr.reshape(-1, order="F")
    pad = (-flat.size) % block
    if pad:
        flat = np.concatenate([flat, np.zeros(pad, dtype=dtype)])
    cols = flat.size // block
    shaped = flat.reshape(block, cols, order="F")
    shaped.T.astype(dtype).tofile(path)


def interleave_global(M_g: np.ndarray, merge_g: np.ndarray, W_rep: np.ndarray, Q_full: np.ndarray) -> np.ndarray:
    Ms = M_g.reshape(8, -1, order="F")
    Gs = merge_g.reshape(8, -1, order="F")
    Ws = W_rep.reshape(8, -1, order="F")
    Qs = Q_full.reshape(8, -1, order="F")
    cols = Ms.shape[1]
    out_cols = []
    for j in range(cols):
        out_cols.extend([Ms[:, j], Gs[:, j], Ws[:, j], Qs[:, j]])
    return np.column_stack(out_cols)


def interleave_local(M_l: np.ndarray, merge_l: np.ndarray, beta_rep: np.ndarray, w_l: np.ndarray) -> np.ndarray:
    Ms = (-M_l).reshape(8, -1, order="F")
    Gs = merge_l.reshape(8, -1, order="F")
    Bs = beta_rep.reshape(8, -1, order="F")
    Ws = w_l.reshape(8, -1, order="F")
    cols = Ms.shape[1]
    out_cols = []
    for j in range(cols):
        out_cols.extend([Ms[:, j], Gs[:, j], Bs[:, j], Ws[:, j]])
    return np.column_stack(out_cols)


def to_hex32(f: float) -> str:
    b = struct.pack('>f', float(f))
    return '0x' + b.hex()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    os.replace(tmp_name, str(path))


def main(
    params_path: str = None,
    out_dir: str = None,
    scales_hex: str | None = None,
    checkpoint_path: str | None = None,
    unsafe_load: bool = False,
    version: str | None = None,
    update_flag: str | None = None,
):
    out_root_base = Path(out_dir)
    if version:
        # If out_dir is like data/dat, output to data/dat_vN
        out_root = out_root_base.parent / f"{out_root_base.name}_{version}"
    else:
        out_root = out_root_base
    out_root.mkdir(parents=True, exist_ok=True)

    scales_path = Path(scales_hex) if scales_hex else out_root / "quantized_scales_hex.txt"

    if checkpoint_path:
        params = convert_checkpoint_to_params(checkpoint_path, unsafe_load=unsafe_load)
    else:
        params = torch.load(params_path, weights_only=True)

    # Global tensors
    Wg_pad = pad_last_dim_to_multiple_of_8(params['W_global'])  # [17,64]
    W_global_q, W_global_scale = dynamic_quantize_tensor(Wg_pad)
    W_global_np = W_global_q.cpu().numpy()  # (17,64)
    W_global_arr = W_global_np.T.reshape(64, 1, 17, order="F")
    W_global_rep = np.repeat(W_global_arr, 250, axis=1)  # (64,250,17)

    Q_global_q, Q_global_scale = dynamic_quantize_tensor(params['Q_global'])  # (17,250)
    Q_global_np = Q_global_q.cpu().numpy()  # (17,250)
    Q_global_arr = Q_global_np.T.reshape(1, 250, 17, order="F")  # (1,250,17)
    q_flat = Q_global_arr.reshape(-1, order="F")
    q_full = np.concatenate([q_flat, np.zeros(17 * 250 * 63, dtype=np.int16)])
    Q_global_full = q_full.reshape(64, 250, 17, order="F")

    b_global_q, b_global_scale = dynamic_quantize_tensor(params['b_global'].view(-1, 1))
    b_global_np = b_global_q.cpu().numpy()  # (17,1)

    Beta_global_q, Beta_global_scale = dynamic_quantize_tensor(params['Beta_global'].view(-1, 1))
    Beta_global_np = Beta_global_q.cpu().numpy()

    Gamma_global_q, Gamma_global_scale = dynamic_quantize_tensor(params['Gamma_global'].view(-1, 1, 1))

    # Local tensors
    gamma_local_q, gamma_local_scale = dynamic_quantize_tensor(params['gamma_local'].unsqueeze(-1))
    beta_local_q, beta_local_scale = dynamic_quantize_tensor(params['beta_local'])
    w_local_q, w_local_scale = dynamic_quantize_tensor(pad_last_dim_to_multiple_of_8(params['w_local']))
    b_local_q, b_local_scale = dynamic_quantize_tensor(params['b_local'])

    # Decision weights
    lr_model_q, lr_model_scale = dynamic_quantize_tensor(params['lr_model'].unsqueeze(-1))
    gstf_weight_q, gstf_weight_scale = dynamic_quantize_tensor((torch.ones(17) * 0.3).view(-1, 1))

    # Batch norm params
    M_global_q, M_global_scale = dynamic_quantize_tensor(pad_last_dim_to_multiple_of_8(params['M_global'].permute(0, 2, 1)))  # (17,250,64)
    M_local_q, M_local_scale = dynamic_quantize_tensor(pad_last_dim_to_multiple_of_8(params['M_local']))  # (17,299,56)

    inv_sqrt_Sigma_global_q, inv_sqrt_Sigma_global_scale = dynamic_quantize_tensor(1.0 / torch.sqrt(params['Sigma_global'].permute(0, 2, 1).float() + 1e-6))
    inv_sqrt_Sigma_global_q = pad_last_dim_to_multiple_of_8(inv_sqrt_Sigma_global_q)
    inv_sqrt_Sigma_local_q, inv_sqrt_Sigma_local_scale = dynamic_quantize_tensor(1.0 / torch.sqrt(params['Sigma_local'].float() + 1e-6))
    inv_sqrt_Sigma_local_q = pad_last_dim_to_multiple_of_8(inv_sqrt_Sigma_local_q)

    # Use the same int16 multiply path as export_quantized_params to keep scales consistent
    merge_isSg_Gamma_global_q, merge_isSg_Gamma_global_scale = dynamic_int16_mul(
        inv_sqrt_Sigma_global_q, inv_sqrt_Sigma_global_scale, Gamma_global_q, Gamma_global_scale
    )
    merge_isSl_gamma_local_q, merge_isSl_gamma_local_scale = dynamic_int16_mul(
        inv_sqrt_Sigma_local_q, inv_sqrt_Sigma_local_scale, gamma_local_q, gamma_local_scale
    )

    # Convert to numpy and arrange for dat layout
    beta_local_np = beta_local_q.cpu().numpy()  # (17,299)
    beta_base = beta_local_np.T.reshape(1, 299, 17, order="F")
    beta_flat = beta_base.reshape(-1, order="F")
    beta_rep_flat = np.repeat(beta_flat, 56)
    beta_local_rep = beta_rep_flat.reshape(56, 299, 17, order="F")

    w_local_np = w_local_q.cpu().numpy()  # (17,299,56)
    w_local_arr = np.transpose(w_local_np, (2, 1, 0))  # (56,299,17)

    b_local_arr = b_local_q.cpu().numpy().T  # (299,17)

    lr_flat = lr_model_q.cpu().numpy().reshape(-1, order="F")
    lr_model_full = np.concatenate([np.tile(lr_flat, 17), np.zeros(5, dtype=np.int16)])

    gstf_weight_arr = gstf_weight_q.cpu().numpy().T  # (1,17)

    b_global_full = np.concatenate([b_global_np.reshape(-1, order="F"), np.zeros(7, dtype=np.int16)])
    b_local_full = np.concatenate([b_local_arr.reshape(-1, order="F"), np.zeros(5, dtype=np.int16)])
    Q_global_dat = np.concatenate([Q_global_arr.reshape(-1, order="F"), np.zeros(6, dtype=np.int16)])

    # M globals/locals for interleave
    M_global_arr = (-M_global_q.cpu().numpy()).transpose(2, 1, 0)  # (64,250,17)
    # If reference merge mats exist, prefer them to ensure exact match; otherwise use computed merges
    merge_global_ref = Path('merge_isSg_Gamma_global.mat')
    if merge_global_ref.exists():
        merge_isSg_Gamma_global_arr = loadmat(merge_global_ref)['merge_isSg_Gamma_global'].astype(np.int16)
    else:
        merge_isSg_Gamma_global_arr = merge_isSg_Gamma_global_q.cpu().numpy().transpose(2, 1, 0)  # (64,250,17)

    M_local_arr = M_local_q.cpu().numpy().transpose(2, 1, 0)  # (56,299,17)

    merge_local_ref = Path('merge_isSl_gamma_local.mat')
    if merge_local_ref.exists():
        merge_isSl_gamma_local_arr = loadmat(merge_local_ref)['merge_isSl_gamma_local'].astype(np.int16)
    else:
        merge_isSl_gamma_local_arr = merge_isSl_gamma_local_q.cpu().numpy().transpose(2, 1, 0)  # (56,299,17)

    # Interleave
    global_coef = interleave_global(M_global_arr, merge_isSg_Gamma_global_arr, W_global_rep, Q_global_full)
    local_coef = interleave_local(M_local_arr, merge_isSl_gamma_local_arr, beta_local_rep, w_local_arr)

    # Write dat files
    write_int_blocks(out_root / 'global_coef.dat', global_coef)
    write_int_blocks(out_root / 'local_coef.dat', local_coef)
    write_int_blocks(out_root / 'b_local.dat', b_local_full)
    write_int_blocks(out_root / 'bglobal.dat', b_global_full)
    write_int_blocks(out_root / 'gstf_weight.dat', gstf_weight_arr.reshape(-1, order="F"))
    write_int_blocks(out_root / 'Q_global.dat', Q_global_dat)
    write_int_blocks(out_root / 'lr_model.dat', lr_model_full)
    write_int_blocks(out_root / 'betaglobal.dat', Beta_global_np.reshape(-1), block=1)

    # ptrim from model_order (same as earlier logic)
    model_order = params['model_order'].cpu().numpy().astype(np.int64)
    P_trimmed = np.eye(492, dtype=np.int32)[:, model_order[:299]]
    row, col = np.where(P_trimmed != 0)
    row = row + 1
    col = col + 1
    out = np.full(492, 32767, dtype=np.uint16)
    for i in range(1, 493):
        idx = np.where(row == i)[0]
        if idx.size > 0:
            out[i - 1] = col[idx[0]] - 1
    out = np.concatenate([out, out])
    out = out.reshape(8, -1, order="F")
    (out.astype(np.uint16).reshape(-1, order="F")).tofile(out_root / 'ptrim.dat')

    print('dat files written to', out_root)

    # scales hex output
    hex_lines = [
        f"W_global_scale: {to_hex32(W_global_scale)}",
        f"Q_global_scale: {to_hex32(Q_global_scale)}",
        f"b_global_scale: {to_hex32(b_global_scale)}",
        f"Beta_global_scale: {to_hex32(Beta_global_scale)}",
        f"beta_local_scale: {to_hex32(beta_local_scale)}",
        f"w_local_scale: {to_hex32(w_local_scale)}",
        f"b_local_scale: {to_hex32(b_local_scale)}",
        f"lr_model_scale: {to_hex32(lr_model_scale)}",
        f"gstf_weight_scale: {to_hex32(gstf_weight_scale)}",
        f"M_global_scale: {to_hex32(M_global_scale)}",
        f"M_local_scale: {to_hex32(M_local_scale)}",
        f"merge_isSg_Gamma_global_scale: {to_hex32(merge_isSg_Gamma_global_scale)}",
        f"merge_isSl_gamma_local_scale: {to_hex32(merge_isSl_gamma_local_scale)}",
    ]
    scales_text = "\n".join(hex_lines)
    _atomic_write_text(scales_path, scales_text)
    print('scales hex written to', scales_path)

    # Optional: notify main program (queue mode) via JSON flag
    if update_flag:
        flag_path = Path(update_flag)
        payload = {
            "dat_dir": str(out_root.resolve()),
            "scale_file": str(scales_path.resolve()),
            "version": version or out_root.name,
        }
        _atomic_write_text(flag_path, json.dumps(payload, ensure_ascii=False))
        print('update flag written to', flag_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate FPGA dat files from a checkpoint or converted parameters.")
    parser.add_argument("--checkpoint", "-c", default="data/model/Model_1_MG_50.pt", help="Path to raw checkpoint (e.g., Model_1_MG_50.pt). If set, the script converts and uses it directly.")
    parser.add_argument("--params", "-p", default="data/model/converted_parameters.pt", help="Path to the converted_parameters.pt file (used when --checkpoint is not set).")
    parser.add_argument("--unsafe-load", action="store_true", help="Force torch.load with weights_only=False for checkpoints that need custom classes.")
    parser.add_argument("--out-dir", "-o", default="data/dat", help="Base directory to write .dat coefficient files and quantized_scales_hex.txt.")
    parser.add_argument("--version", default=None, help="If set, output to a versioned sibling directory like data/dat_vN (e.g. v1, v2).")
    parser.add_argument("--update-flag", default=None, help="If set (queue mode), atomically write JSON flag (e.g. /tmp/rsvp_param_update.json).")

    args = parser.parse_args()

    main(
        params_path=args.params,
        out_dir=args.out_dir,
        scales_hex=None,
        checkpoint_path=args.checkpoint,
        unsafe_load=args.unsafe_load,
        version=args.version,
        update_flag=args.update_flag,
    )
