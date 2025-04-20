from __future__ import annotations
from .k_diffusion import sampling as k_diffusion_sampling
from .extra_samplers import uni_pc
from typing import TYPE_CHECKING, Callable, NamedTuple
if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.model_base import BaseModel
    from comfy.controlnet import ControlBase
import torch
from functools import partial
import collections
from comfy import model_management
import math
import logging
import comfy.sampler_helpers
import comfy.model_patcher
import comfy.patcher_extension
import comfy.hooks
import scipy.stats
import numpy as np
from collections import namedtuple
from typing import Dict

# one‑time declaration
Cond = namedtuple(
    "Cond",
    ["input_x", "mult", "conditioning", "area", "control", "patches", "uuid", "hooks"]
)


def add_area_dims(area, num_dims):
    while (len(area) // 2) < num_dims:
        area = [2147483648] + area[:len(area) // 2] + [0] + area[len(area) // 2:]
    return area


# simple cache: uuid → {cond_key: processed_tensor}
_ramp_cache: Dict[Tuple[int,int], torch.Tensor] = {}
_conditioning_cache: Dict[str, Dict[str, torch.Tensor]] = {}
_model_sampling_map: Dict[int, object] = {}

def get_area_and_mult(conds, x_in, timestep_in):
    dims = tuple(x_in.shape[2:])
    area = None
    strength = 1.0

    if 'timestep_start' in conds:
        t0 = conds['timestep_start']
        if timestep_in[0] > t0:
            return None
    if 'timestep_end' in conds:
        t1 = conds['timestep_end']
        if timestep_in[0] < t1:
            return None

    if 'area' in conds:
        area = list(conds['area'])
        area = add_area_dims(area, len(dims))
        if (len(area)//2) > len(dims):
            half = len(area)//2
            area = area[:len(dims)] + area[half:half+len(dims)]

    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in
    if area is not None:
        for i in range(len(dims)):
            start = area[len(dims)+i]
            size  = area[i]
            size = min(input_x.shape[2+i] - start, size)
            area[i] = size
            input_x = input_x.narrow(i+2, start, size)

    if 'mask' in conds:
        mask_strength = conds.get("mask_strength", 1.0)
        mask = conds['mask']
        assert mask.shape[1:] == x_in.shape[2:]
        mask = mask[:input_x.shape[0]]
        for i in range(len(dims)):
            start = area[len(dims)+i]
            size  = area[i]
            mask = mask.narrow(i+1, start, size)
        mask = mask * mask_strength
        mask = mask.unsqueeze(1).repeat(
            input_x.shape[0]//mask.shape[0], input_x.shape[1], 1, 1
        )
    else:
        mask = torch.ones_like(input_x)

    mult = mask * strength

    # --- optimized “fuzz” border ramp (GPU only) with caching ---
    if 'mask' not in conds and area is not None:
        fuzz = 8
        region_shape = tuple(mult.shape[2:2+len(dims)])
        full_shape   = tuple(x_in.shape[2:2+len(dims)])
        sizes        = area[:len(dims)]
        offsets      = area[len(dims):]

        for dim_idx, size in enumerate(region_shape):
            rr = min(fuzz, size // 4)
            if rr <= 0:
                continue

            device = mult.device
            dtype  = mult.dtype
            key = (size, rr)

            # get or build 1D ramp
            if key in _ramp_cache:
                base_ramp = _ramp_cache[key].to(device=device, dtype=dtype)
            else:
                ramp = torch.ones(size, dtype=dtype, device=device)
                if offsets[dim_idx] != 0:
                    ramp[:rr] = torch.arange(1, rr+1, device=device, dtype=dtype) / rr
                if (sizes[dim_idx] + offsets[dim_idx]) < full_shape[dim_idx]:
                    rev = torch.arange(1, rr+1, device=device, dtype=dtype) / rr
                    rev = rev.flip(0)
                    ramp[-rr:] = ramp[-rr:] * rev
                _ramp_cache[key] = ramp.cpu()
                base_ramp = ramp

            shape = [1] * mult.ndim
            shape[2 + dim_idx] = size
            weight = base_ramp.view(shape)
            mult = mult * weight

    # --- cached conditioning build ---
    uid = conds['uuid']
    if uid not in _conditioning_cache:
        cache: Dict[str, torch.Tensor] = {}
        for key, cond_obj in conds["model_conds"].items():
            cache[key] = cond_obj.process_cond(
                batch_size = input_x.shape[0],
                device     = input_x.device,
                area       = area
            )
        _conditioning_cache[uid] = cache
    conditioning = _conditioning_cache[uid]

    hooks   = conds.get('hooks', None)
    control = conds.get('control', None)
    patches = None

    if 'gligen' in conds:
        gtype, gmodel, *gargs = conds['gligen']
        patch = (
            gmodel.model.set_position if gtype=="position" else gmodel.model.set_empty
        )(input_x.shape, *(gargs), input_x.device)
        patches = {'middle_patch': [patch]}

    return Cond(
        input_x,
        mult,
        conditioning,
        area,
        control,
        patches,
        uid,
        hooks
    )

def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list):
    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out

def finalize_default_conds(model: 'BaseModel', hooked_to_run: dict[comfy.hooks.HookGroup,list[tuple[tuple,int]]], default_conds: list[list[dict]], x_in, timestep, model_options):
    # need to figure out remaining unmasked area for conds
    default_mults = []
    for _ in default_conds:
        default_mults.append(torch.ones_like(x_in))
    # look through each finalized cond in hooked_to_run for 'mult' and subtract it from each cond
    for lora_hooks, to_run in hooked_to_run.items():
        for cond_obj, i in to_run:
            # if no default_cond for cond_type, do nothing
            if len(default_conds[i]) == 0:
                continue
            area: list[int] = cond_obj.area
            if area is not None:
                curr_default_mult: torch.Tensor = default_mults[i]
                dims = len(area) // 2
                for i in range(dims):
                    curr_default_mult = curr_default_mult.narrow(i + 2, area[i + dims], area[i])
                curr_default_mult -= cond_obj.mult
            else:
                default_mults[i] -= cond_obj.mult
    # for each default_mult, ReLU to make negatives=0, and then check for any nonzeros
    for i, mult in enumerate(default_mults):
        # if no default_cond for cond type, do nothing
        if len(default_conds[i]) == 0:
            continue
        torch.nn.functional.relu(mult, inplace=True)
        # if mult is all zeros, then don't add default_cond
        if torch.max(mult) == 0.0:
            continue

        cond = default_conds[i]
        for x in cond:
            # do get_area_and_mult to get all the expected values
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue
            # replace p's mult with calculated mult
            p = p._replace(mult=mult)
            if p.hooks is not None:
                model.current_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
            hooked_to_run.setdefault(p.hooks, list())
            hooked_to_run[p.hooks] += [(p, i)]

def calc_cond_batch(model: 'BaseModel', conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _calc_cond_batch,
        comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.CALC_COND_BATCH, model_options, is_model_options=True)
    )
    return executor.execute(model, conds, x_in, timestep, model_options)

def _calc_cond_batch(model: BaseModel, conds: list[list[dict]], x_in: torch.Tensor, timestep, model_options):
    """
    Fully optimized:
      1) Flatten cond → cond_obj via get_area_and_mult
      2) Group by (hooks_id, input_x.shape)
      3) Batch each group once through U‑Net
      4) Inverse‑counts per‑pixel normalization
    """
    # Pre‑bind invariants to locals
    cur_patcher = model.current_patcher
    apply_model = model.apply_model
    merge_dicts = comfy.patcher_extension.merge_nested_dicts

    batch_size, C, *spatial_shape = x_in.shape

    # 1) Build flat list of (cond_obj, slot)
    all_items = []
    for slot_idx, slot_conds in enumerate(conds):
        for c in slot_conds:
            p = get_area_and_mult(c, x_in, timestep)
            if p is None:
                continue
            if p.hooks:
                cur_patcher.prepare_hook_patches_current_keyframe(timestep, p.hooks, model_options)
            all_items.append((p, slot_idx))

    if not all_items:
        return [torch.zeros_like(x_in) for _ in conds]

    # 2) Group by (hooks_id, shape)
    groups: dict[tuple[int, tuple], list] = {}
    for p, slot_idx in all_items:
        key = (id(p.hooks), p.input_x.shape)
        groups.setdefault(key, []).append((p, slot_idx))

    # Prepare accumulators
    out_conds  = [torch.zeros_like(x_in) for _ in conds]
    out_counts = [torch.zeros_like(x_in) for _ in conds]

    # Cache transformer_options per hook‑group
    transformer_options_cache: dict[int, dict] = {}

    # 3) Process each group
    for (hooks_id, _), members in groups.items():
        # unpack inputs, mults, conds_dic, slots, hooks
        inputs    = [p.input_x      for (p, _) in members]
        mults     = [p.mult         for (p, _) in members]
        conds_dic = [p.conditioning for (p, _) in members]
        slots     = [slot_idx       for (_, slot_idx) in members]
        hooks     = members[0][0].hooks

        # batch concat
        big_x    = torch.cat(inputs, dim=0)
        big_mult = torch.cat(mults,  dim=0)
        tsteps   = timestep.repeat(len(inputs))

        # merge conditioning dicts
        merged_c = cond_cat(conds_dic)

        # get or build transformer_options
        if hooks_id in transformer_options_cache:
            topts = transformer_options_cache[hooks_id]
        else:
            topts = cur_patcher.apply_hooks(hooks=hooks)
            if 'transformer_options' in model_options:
                topts = merge_dicts(
                    topts,
                    model_options['transformer_options'],
                    copy_dict1=False
                )
            transformer_options_cache[hooks_id] = topts
        merged_c['transformer_options'] = topts

        # control network if present
        ctrl = members[0][0].control
        if ctrl:
            merged_c['control'] = ctrl.get_control(
                big_x, tsteps, merged_c, len(members), topts
            )

        # run model once and split outputs
        # inject cond_or_uncond into transformer_options so IPAdapter’s attention patch sees it
        topts = merged_c['transformer_options']
        topts['cond_or_uncond'] = slots
        merged_c['transformer_options'] = topts
        if 'model_function_wrapper' in model_options:
            # model_function_wrapper will pick up cond_or_uncond from merged_c['cond_or_uncond']
            outs = model_options['model_function_wrapper'](
                apply_model,
                {"input": big_x, "timestep": tsteps, "c": merged_c}
            ).chunk(len(members))
        else:
            # pass merged_c (which now contains cond_or_uncond) into the U‑Net
            outs = apply_model(big_x, tsteps, **merged_c).chunk(len(members))

        # accumulate weighted sums and counts
        for (p, slot_idx), piece in zip(members, outs):
            out_conds[slot_idx]  += piece * p.mult
            out_counts[slot_idx] += p.mult

    # 4) Inverse‑counts per‑pixel normalization
    final = []
    for oc, cnt in zip(out_conds, out_counts):
        inv = torch.where(cnt > 0, 1.0 / cnt, 0.0)
        final.append(oc * inv)
    return final

def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options): #TODO: remove
    logging.warning("WARNING: The comfy.samplers.calc_cond_uncond_batch function is deprecated please use the calc_cond_batch one instead.")
    return tuple(calc_cond_batch(model, [cond, uncond], x_in, timestep, model_options))

def cfg_function(model, cond_pred, uncond_pred, cond_scale, x, timestep, model_options={}, cond=None, uncond=None):
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "cond_scale": cond_scale, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result

#The main sampling function shared by all the samplers
#Returns denoised
def sampling_function(model, x, timestep, uncond, cond, cond_scale, model_options={}, seed=None):
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    conds = [cond, uncond_]
    out = calc_cond_batch(model, conds, x, timestep, model_options)

    for fn in model_options.get("sampler_pre_cfg_function", []):
        args = {"conds":conds, "conds_out": out, "cond_scale": cond_scale, "timestep": timestep,
                "input": x, "sigma": timestep, "model": model, "model_options": model_options}
        out  = fn(args)

    return cfg_function(model, out[0], out[1], cond_scale, x, timestep, model_options=model_options, cond=cond, uncond=uncond_)


class KSamplerX0Inpaint:
    def __init__(self, model, sigmas):
        self.inner_model = model
        self.sigmas = sigmas
    def __call__(self, x, sigma, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.scale_latent_inpaint(x=x, sigma=sigma, noise=self.noise, latent_image=self.latent_image) * latent_mask
        out = self.inner_model(x, sigma, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

def simple_scheduler(model_sampling, steps):
    """
    Pure‑PyTorch version of:
        ss = len(sigmas)/steps
        for x in range(steps): pick sigma[-(1+int(x*ss))]
        append 0.0
    """
    sigmas = model_sampling.sigmas
    device = sigmas.device
    dtype  = sigmas.dtype
    L = sigmas.size(0)

    # vector of positions 0..steps-1
    x = torch.arange(steps, device=device, dtype=dtype)
    # compute floor(x * (L/steps)), then reverse & clamp
    idxs = (L - 1 - (x * (L / steps)).floor()).clamp(0, L - 1).to(torch.long)
    # gather and append final zero
    sched = torch.cat([sigmas[idxs], torch.zeros(1, device=device, dtype=dtype)])
    return sched.to(torch.float32)

def ddim_scheduler(model_sampling, steps):
    """
    GPU‑only DDIM scheduler:
      - signature and outputs identical to original
      - only uses GPU tensor ops (plus a tiny device‑local isclose check)
    """
    sigmas = model_sampling.sigmas
    device = sigmas.device
    dtype  = sigmas.dtype
    N = sigmas.size(0)

    # 1) Decide if the second sigma is treated as zero
    #    matching: if sigma[1] ≈ 0, then omit the final zero and bump steps

    zero = torch.tensor(0.0, device=device, dtype=dtype)
    zero1 = torch.zeros(1,   device=device, dtype=dtype)
    if torch.isclose(sigmas[1], zero, atol=1e-5):
        steps_adj = steps + 1
        include_zero = False
    else:
        steps_adj = steps
        include_zero = True

    # 2) Compute the step interval (at least 1)
    ss = max(N // steps_adj, 1)

    # 3) Build indices [1, 1+ss, 1+2*ss, ...] on GPU
    idxs = torch.arange(1, N, ss, device=device, dtype=torch.long)

    # 4) Gather those sigma values and reverse
    sched = torch.flip(sigmas[idxs], dims=[0])

    # 5) Append a literal zero if needed
    if include_zero:
        sched = torch.cat([sched, zero1])

    # 6) Return as float32 on CPU (mirroring original)
    return sched.to(torch.float32)

def normal_scheduler(model_sampling, steps, sgm=False, floor=False):
    """
    Pure‑PyTorch “normal” scheduler:
      - builds timesteps on‑device
      - calls model_sampling.sigma(timesteps) in one shot (falling back to a brief loop only if needed)
      - appends 0.0 if required
    """
    sigmas = model_sampling.sigmas
    device = sigmas.device
    dtype  = sigmas.dtype
    s      = model_sampling

    # get scalar start/end as Python floats, then lift to device tensors
    start = s.timestep(s.sigma_max)
    end   = s.timestep(s.sigma_min)
    t0    = start.to(device=device, dtype=dtype)
    t1    = end.to(device=device, dtype=dtype)

    append_zero = True
    if sgm:
        # (steps+1) points, drop last
        t = torch.linspace(t0, t1, steps + 1, device=device, dtype=dtype)[:-1]
    else:
        # if the final sigma would be exactly zero, skip that sample
        if math.isclose(float(s.sigma(end)), 0.0, abs_tol=1e-5):
            steps += 1
            append_zero = False
        t = torch.linspace(t0, t1, steps, device=device, dtype=dtype)

    # try vectorized call
    try:
        vals = s.sigma(t)  # should produce a tensor of shape (steps,)
    except Exception:
        # fallback to minimal Python loop if necessary
        vals = torch.stack([
            torch.tensor(float(s.sigma(ti.item())), device=device, dtype=dtype)
            for ti in t
        ])

    if append_zero:
        zero1 = torch.zeros(1, device=device, dtype=dtype)
        vals  = torch.cat([vals, zero1])

    return vals.to(torch.float32)

# Implemented based on: https://arxiv.org/abs/2407.12173
def beta_scheduler(model_sampling, steps, alpha=0.6, beta=0.6):
    sigmas = model_sampling.sigmas
    total  = len(sigmas) - 1
    device = sigmas.device
    dtype  = sigmas.dtype

    ts = 1.0 - np.linspace(0.0, 1.0, steps, endpoint=False)
    idxs_cpu = np.rint(scipy.stats.beta.ppf(ts, alpha, beta) * total).astype(np.int64)

    unique = [idxs_cpu[0]] if len(idxs_cpu)>0 else []
    for x in idxs_cpu[1:]:
        if x != unique[-1]:
            unique.append(x)
    unique.append(total+1)  # temporary marker for the final 0.0
    # remove the marker from gathering, we'll append 0 separately
    unique_idxs = unique[:-1]

    idxs = torch.tensor(unique_idxs, device=device, dtype=torch.long).clamp(0, total)

    sched = sigmas[idxs]
    sched = torch.cat([sched, torch.zeros(1, device=device, dtype=dtype)])

    return sched.to(torch.float32)

# from: https://github.com/genmoai/models/blob/main/src/mochi_preview/infer.py#L41
def linear_quadratic_schedule(model_sampling, steps, threshold_noise=0.025, linear_steps=None):
    sigmas = model_sampling.sigmas
    device = sigmas.device
    dtype = sigmas.dtype

    if steps == 1:
        sched = torch.tensor([1.0, 0.0], device=device, dtype=dtype)
    else:
        ls = linear_steps if linear_steps is not None else steps // 2
        qs = steps - ls
        lin = torch.arange(ls, device=device, dtype=dtype) * (threshold_noise / ls)
        tnsd = ls - threshold_noise * steps
        quad_coef = tnsd / (ls * qs * qs)
        lin_coef = threshold_noise / ls - 2 * tnsd / (qs * qs)
        const = quad_coef * (ls ** 2)
        i = torch.arange(ls, steps, device=device, dtype=dtype)
        quad = quad_coef * i * i + lin_coef * i + const
        sched = torch.cat([lin, quad, torch.tensor([1.0], device=device, dtype=dtype)])
        sched = 1.0 - sched

    sigma_max = model_sampling.sigma_max
    sigma_max_dev = sigma_max.to(device=device, dtype=dtype) if isinstance(sigma_max, torch.Tensor) else torch.tensor(sigma_max, device=device, dtype=dtype)
    sched = sched * sigma_max_dev

    return sched.cpu().to(torch.float32)

# Referenced from https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/15608
def kl_optimal_scheduler(n: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    """
    Pure‑PyTorch KL‑optimal scheduler:
      - vectorized from the arXiv formula
      - returns length (n+1), with final entry = 0.0
    """
    # device-agnostic: small tensor on CPU is fine (will be moved later)
    adj = torch.arange(n, dtype=torch.float32) / (n - 1)
    a   = torch.atan(torch.tensor(sigma_min, dtype=torch.float32))
    b   = torch.atan(torch.tensor(sigma_max, dtype=torch.float32))
    vals = torch.tan(adj * a + (1 - adj) * b)
    return torch.cat([vals, torch.zeros(1, dtype=torch.float32)])

def zipf_linear_scheduler(model_sampling, steps: int, x_start=3.2, x_end=2.75):
    """
    Generates a sigma schedule using a Zipf-based weighting function where the exponent
    changes gradually from x_start to x_end over the course of the schedule.
    """
    # pick device + dtype
    base_sigmas = model_sampling.sigmas
    device = base_sigmas.device
    dtype  = base_sigmas.dtype

    # cast sigma_min/max to 1‑element tensors on GPU
    sigma_min = torch.tensor(float(model_sampling.sigma_min), device=device, dtype=dtype)
    sigma_max = torch.tensor(float(model_sampling.sigma_max), device=device, dtype=dtype)

    # 1) exponent curve
    x_curve = torch.linspace(x_start, x_end, steps, device=device, dtype=dtype)
    ranks   = torch.arange(1, steps + 1, device=device, dtype=dtype)

    # 2) Zipf weights
    weights = 1.0 / (ranks ** x_curve)
    weights = weights / weights.sum()

    # 3) cumulative, with a [1]-element zero prefix
    zero_prefix  = torch.zeros(1, device=device, dtype=dtype)
    cum_weights = torch.cat([zero_prefix, torch.cumsum(weights, dim=0)])  # now shape (steps+1,)
    cum_weights = 1.0 - cum_weights / cum_weights[-1]

    # 4) scale into [sigma_min, sigma_max]
    sigmas = sigma_min + (sigma_max - sigma_min) * cum_weights
    sigmas[-1] = sigma_min   # enforce exact min

    return sigmas

def get_mask_aabb(masks):
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty

def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                modified = c.copy()
                a = area[1:]
                a_len = len(a) // 2
                area = ()
                for d in range(len(dims)):
                    area += (max(1, round(a[d] * dims[d])),)
                for d in range(len(dims)):
                    area += (round(a[d + a_len] * dims[d]),)

                modified['area'] = area
                c = modified
                conditions[i] = c

        if 'mask' in c:
            mask = c['mask']
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=dims, mode='bilinear', align_corners=False).squeeze(1)

            if modified.get("set_area_to_bounds", False): #TODO: handle dim != 2
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    modified['area'] = (8, 8, 0, 0)
                else:
                    box = boxes[0]
                    H, W, Y, X = (box[3] - box[1] + 1, box[2] - box[0] + 1, box[1], box[0])
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified['area'] = area

            modified['mask'] = mask
            conditions[i] = modified

def resolve_areas_and_cond_masks(conditions, h, w, device):
    logging.warning("WARNING: The comfy.samplers.resolve_areas_and_cond_masks function is deprecated please use the resolve_areas_and_cond_masks_multidim one instead.")
    return resolve_areas_and_cond_masks_multidim(conditions, [h, w], device)

def create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c:
        return

    def area_inside(a, area_cmp):
        a = add_area_dims(a, len(area_cmp) // 2)
        area_cmp = add_area_dims(area_cmp, len(a) // 2)

        a_l = len(a) // 2
        area_cmp_l = len(area_cmp) // 2
        for i in range(min(a_l, area_cmp_l)):
            if a[a_l + i] < area_cmp[area_cmp_l + i]:
                return False
        for i in range(min(a_l, area_cmp_l)):
            if (a[i] + a[a_l + i]) > (area_cmp[i] + area_cmp[area_cmp_l + i]):
                return False
        return True

    c_area = c['area']
    smallest = None
    for x in conds:
        if 'area' in x:
            a = x['area']
            if area_inside(c_area, a):
                if smallest is None:
                    smallest = x
                elif 'area' not in smallest:
                    smallest = x
                else:
                    if math.prod(smallest['area'][:len(smallest['area']) // 2]) > math.prod(a[:len(a) // 2]):
                        smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest:
        if smallest['area'] == c_area:
            return

    out = c.copy()
    out['model_conds'] = smallest['model_conds'].copy() #TODO: which fields should be copied?
    conds += [out]

def calculate_start_end_timesteps(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        # handle clip hook schedule, if needed
        if 'clip_start_percent' in x:
            timestep_start = s.percent_to_sigma(max(x['clip_start_percent'], x.get('start_percent', 0.0)))
            timestep_end = s.percent_to_sigma(min(x['clip_end_percent'], x.get('end_percent', 1.0)))
        else:
            if 'start_percent' in x:
                timestep_start = s.percent_to_sigma(x['start_percent'])
            if 'end_percent' in x:
                timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if (timestep_start is not None):
                n['timestep_start'] = timestep_start
            if (timestep_end is not None):
                n['timestep_end'] = timestep_end
            conds[t] = n

def pre_run_control(model, conds):
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            x['control'].pre_run(model, percent_to_timestep_function)

def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    for x in range(len(cond_cnets)):
        temp = uncond_other[x % len(uncond_other)]
        o = temp[0]
        if name in o and o[name] is not None:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond += [n]
        else:
            n = o.copy()
            n[name] = uncond_fill_func(cond_cnets, x)
            uncond[temp[1]] = n

def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs):
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4: #TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x['model_conds'].copy()
        for k in out:
            model_conds[k] = out[k]
        x['model_conds'] = model_conds
        conds[t] = x
    return conds

class Sampler:
    def sample(self):
        pass

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

KSAMPLER_NAMES = ["euler", "euler_cfg_pp", "euler_ancestral", "euler_ancestral_cfg_pp", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_cfg_pp", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",
                  "ipndm", "ipndm_v", "deis", "res_multistep", "res_multistep_cfg_pp", "res_multistep_ancestral", "res_multistep_ancestral_cfg_pp",
                  "gradient_estimation", "er_sde", "seeds_2", "seeds_3"]

class KSAMPLER(Sampler):
    def __init__(self, sampler_function, extra_options={}, inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self, model_wrap, sigmas, extra_args, callback, noise, latent_image=None, denoise_mask=None, disable_pbar=False):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callback = None
        total_steps = len(sigmas) - 1
        if callback is not None:
            k_callback = lambda x: callback(x["i"], x["denoised"], x["x"], total_steps)

        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callback=k_callback, disable=disable_pbar, **self.extra_options)
        samples = model_wrap.inner_model.model_sampling.inverse_noise_scaling(sigmas[-1], samples)
        return samples


def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    if sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callback, disable):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callback=callback, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callback, disable, **extra_options):
            if len(sigmas) <= 1:
                return noise

            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callback=callback, disable=disable, **extra_options)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

    return KSAMPLER(sampler_function, extra_options, inpaint_options)


def process_conds(model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None):
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, 'extra_conds'):
        for k in conds:
            conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    #make sure each cond area has an opposite one with the same area
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        for c in conds[k]:
            if 'hooks' in c:
                for hook in c['hooks'].hooks:
                    hook.initialize_timesteps(model)

    for k in conds:
        pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), conds[k], 'control', lambda cond_cnets, x: cond_cnets[x])
                apply_empty_x_to_equal_area(positive, conds[k], 'gligen', lambda cond_cnets, x: cond_cnets[x])

    return conds


def preprocess_conds_hooks(conds: dict[str, list[dict[str]]]):
    # determine which ControlNets have extra_hooks that should be combined with normal hooks
    hook_replacement: dict[tuple[ControlBase, comfy.hooks.HookGroup], list[dict]] = {}
    for k in conds:
        for kk in conds[k]:
            if 'control' in kk:
                control: 'ControlBase' = kk['control']
                extra_hooks = control.get_extra_hooks()
                if len(extra_hooks) > 0:
                    hooks: comfy.hooks.HookGroup = kk.get('hooks', None)
                    to_replace = hook_replacement.setdefault((control, hooks), [])
                    to_replace.append(kk)
    # if nothing to replace, do nothing
    if len(hook_replacement) == 0:
        return

    # for optimal sampling performance, common ControlNets + hook combos should have identical hooks
    # on the cond dicts
    for key, conds_to_modify in hook_replacement.items():
        control = key[0]
        hooks = key[1]
        hooks = comfy.hooks.HookGroup.combine_all_hooks(control.get_extra_hooks() + [hooks])
        # if combined hooks are not None, set as new hooks for all relevant conds
        if hooks is not None:
            for cond in conds_to_modify:
                cond['hooks'] = hooks

def filter_registered_hooks_on_conds(conds: dict[str, list[dict[str]]], model_options: dict[str]):
    '''Modify 'hooks' on conds so that only hooks that were registered remain. Properly accounts for
    HookGroups that have the same reference.'''
    registered: comfy.hooks.HookGroup = model_options.get('registered_hooks', None)
    # if None were registered, make sure all hooks are cleaned from conds
    if registered is None:
        for k in conds:
            for kk in conds[k]:
                kk.pop('hooks', None)
        return
    # find conds that contain hooks to be replaced - group by common HookGroup refs
    hook_replacement: dict[comfy.hooks.HookGroup, list[dict]] = {}
    for k in conds:
        for kk in conds[k]:
            hooks: comfy.hooks.HookGroup = kk.get('hooks', None)
            if hooks is not None:
                if not hooks.is_subset_of(registered):
                    to_replace = hook_replacement.setdefault(hooks, [])
                    to_replace.append(kk)
    # for each hook to replace, create a new proper HookGroup and assign to all common conds
    for hooks, conds_to_modify in hook_replacement.items():
        new_hooks = hooks.new_with_common_hooks(registered)
        if len(new_hooks) == 0:
            new_hooks = None
        for kk in conds_to_modify:
            kk['hooks'] = new_hooks


def get_total_hook_groups_in_conds(conds: dict[str, list[dict[str]]]):
    hooks_set = set()
    for k in conds:
        for kk in conds[k]:
            hooks_set.add(kk.get('hooks', None))
    return len(hooks_set)


def cast_to_load_options(model_options: dict[str], device=None, dtype=None):
    '''
    If any patches from hooks, wrappers, or callbacks have .to to be called, call it.
    '''
    if model_options is None:
        return
    to_load_options = model_options.get("to_load_options", None)
    if to_load_options is None:
        return

    casts = []
    if device is not None:
        casts.append(device)
    if dtype is not None:
        casts.append(dtype)
    # if nothing to apply, do nothing
    if len(casts) == 0:
        return

    # try to call .to on patches
    if "patches" in to_load_options:
        patches = to_load_options["patches"]
        for name in patches:
            patch_list = patches[name]
            for i in range(len(patch_list)):
                if hasattr(patch_list[i], "to"):
                    for cast in casts:
                        patch_list[i] = patch_list[i].to(cast)
    if "patches_replace" in to_load_options:
        patches = to_load_options["patches_replace"]
        for name in patches:
            patch_list = patches[name]
            for k in patch_list:
                if hasattr(patch_list[k], "to"):
                    for cast in casts:
                        patch_list[k] = patch_list[k].to(cast)
    # try to call .to on any wrappers/callbacks
    wrappers_and_callbacks = ["wrappers", "callbacks"]
    for wc_name in wrappers_and_callbacks:
        if wc_name in to_load_options:
            wc: dict[str, list] = to_load_options[wc_name]
            for wc_dict in wc.values():
                for wc_list in wc_dict.values():
                    for i in range(len(wc_list)):
                        if hasattr(wc_list[i], "to"):
                            for cast in casts:
                                wc_list[i] = wc_list[i].to(cast)


class CFGGuider:
    def __init__(self, model_patcher: ModelPatcher):
        self.model_patcher = model_patcher
        self.model_options = model_patcher.model_options
        self.original_conds = {}
        self.cfg = 1.0

    def set_conds(self, positive, negative):
        self.inner_set_conds({"positive": positive, "negative": negative})

    def set_cfg(self, cfg):
        self.cfg = cfg

    def inner_set_conds(self, conds):
        for k in conds:
            self.original_conds[k] = comfy.sampler_helpers.convert_cond(conds[k])

    def __call__(self, *args, **kwargs):
        return self.predict_noise(*args, **kwargs)

    def predict_noise(self, x, timestep, model_options={}, seed=None):
        return sampling_function(self.inner_model, x, timestep, self.conds.get("negative", None), self.conds.get("positive", None), self.cfg, model_options=model_options, seed=seed)

    def inner_sample(self, noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed):
        if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
            latent_image = self.inner_model.process_latent_in(latent_image)

        self.conds = process_conds(self.inner_model, noise, self.conds, device, latent_image, denoise_mask, seed)

        extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
        extra_model_options.setdefault("transformer_options", {})["sample_sigmas"] = sigmas
        extra_args = {"model_options": extra_model_options, "seed": seed}

        executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
            sampler.sample,
            sampler,
            comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE, extra_args["model_options"], is_model_options=True)
        )
        samples = executor.execute(self, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)
        return self.inner_model.process_latent_out(samples.to(torch.float32))

    def outer_sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        self.inner_model, self.conds, self.loaded_models = comfy.sampler_helpers.prepare_sampling(self.model_patcher, noise.shape, self.conds, self.model_options)
        device = self.model_patcher.load_device

        if denoise_mask is not None:
            denoise_mask = comfy.sampler_helpers.prepare_mask(denoise_mask, noise.shape, device)

        device = self.model_patcher.load_device
        noise, latent_image, sigmas = [
            t.to(device=device) for t in (noise, latent_image, sigmas)
        ]
        cast_to_load_options(self.model_options, device=device, dtype=self.model_patcher.model_dtype())

        try:
            self.model_patcher.pre_run()
            output = self.inner_sample(noise, latent_image, device, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            self.model_patcher.cleanup()

        comfy.sampler_helpers.cleanup_models(self.conds, self.loaded_models)
        del self.inner_model
        del self.loaded_models
        return output

    def sample(self, noise, latent_image, sampler, sigmas, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
        if sigmas.shape[-1] == 0:
            return latent_image

        self.conds = {}
        for k in self.original_conds:
            self.conds[k] = list(map(lambda a: a.copy(), self.original_conds[k]))
        preprocess_conds_hooks(self.conds)

        try:
            orig_model_options = self.model_options
            self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)
            # if one hook type (or just None), then don't bother caching weights for hooks (will never change after first step)
            orig_hook_mode = self.model_patcher.hook_mode
            if get_total_hook_groups_in_conds(self.conds) <= 1:
                self.model_patcher.hook_mode = comfy.hooks.EnumHookMode.MinVram
            comfy.sampler_helpers.prepare_model_patcher(self.model_patcher, self.conds, self.model_options)
            filter_registered_hooks_on_conds(self.conds, self.model_options)
            executor = comfy.patcher_extension.WrapperExecutor.new_class_executor(
                self.outer_sample,
                self,
                comfy.patcher_extension.get_all_wrappers(comfy.patcher_extension.WrappersMP.OUTER_SAMPLE, self.model_options, is_model_options=True)
            )
            output = executor.execute(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)
        finally:
            cast_to_load_options(self.model_options, device=self.model_patcher.offload_device)
            self.model_options = orig_model_options
            self.model_patcher.hook_mode = orig_hook_mode
            self.model_patcher.restore_hook_patches()

        del self.conds
        return output


def sample(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_cfg(cfg)
    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed)


SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]

class SchedulerHandler(NamedTuple):
    handler: Callable[..., torch.Tensor]
    # Boolean indicates whether to call the handler like:
    #  scheduler_function(model_sampling, steps) or
    #  scheduler_function(n, sigma_min: float, sigma_max: float)
    use_ms: bool = True

SCHEDULER_HANDLERS = {
    "normal": SchedulerHandler(normal_scheduler),
    "karras": SchedulerHandler(k_diffusion_sampling.get_sigmas_karras, use_ms=False),
    "exponential": SchedulerHandler(k_diffusion_sampling.get_sigmas_exponential, use_ms=False),
    "sgm_uniform": SchedulerHandler(partial(normal_scheduler, sgm=True)),
    "simple": SchedulerHandler(simple_scheduler),
    "ddim_uniform": SchedulerHandler(ddim_scheduler),
    "beta": SchedulerHandler(beta_scheduler),
    "linear_quadratic": SchedulerHandler(linear_quadratic_schedule),
    "kl_optimal": SchedulerHandler(kl_optimal_scheduler, use_ms=False),
    "zipf_linear": SchedulerHandler(zipf_linear_scheduler),
}
SCHEDULER_NAMES = list(SCHEDULER_HANDLERS)

# --- Scheduler memoization ---
from functools import lru_cache
@lru_cache(maxsize=32)
def _calculate_sigmas_cached(ms_id: int, scheduler_name: str, steps: int, sigma_min: float, sigma_max: float) -> torch.Tensor:
    model_sampling = _model_sampling_map[ms_id]
    handler = SCHEDULER_HANDLERS.get(scheduler_name)
    if handler is None:
        raise ValueError(f"invalid scheduler {scheduler_name}")
    if handler.use_ms:
        return handler.handler(model_sampling, steps)
    return handler.handler(n=steps, sigma_min=sigma_min, sigma_max=sigma_max)


def calculate_sigmas(model_sampling: object, scheduler_name: str, steps: int) -> torch.Tensor:
    """
    Cached wrapper around scheduler handlers. Uses LRU cache keyed by
    (model_sampling id, scheduler_name, steps, sigma_min, sigma_max).
    """
    # stash sampling instance for use in cached call
    ms_id = id(model_sampling)
    _model_sampling_map[ms_id] = model_sampling
    return _calculate_sigmas_cached(
        ms_id,
        scheduler_name,
        steps,
        float(model_sampling.sigma_min),
        float(model_sampling.sigma_max),
    )

def sampler_object(name):
    if name == "uni_pc":
        sampler = KSAMPLER(uni_pc.sample_unipc)
    elif name == "uni_pc_bh2":
        sampler = KSAMPLER(uni_pc.sample_unipc_bh2)
    elif name == "ddim":
        sampler = ksampler("euler", inpaint_options={"random": True})
    else:
        sampler = ksampler(name)
    return sampler

class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self, model, steps, device, sampler=None, scheduler=None, denoise=None, model_options={}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps):
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas(self.model.get_model_object("model_sampling"), self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps, denoise=None):
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            if denoise <= 0.0:
                self.sigmas = torch.FloatTensor([])
            else:
                new_steps = int(steps/denoise)
                sigmas = self.calculate_sigmas(new_steps).to(self.device)
                self.sigmas = sigmas[-(steps + 1):]

    def sample(self, noise, positive, negative, cfg, latent_image=None, start_step=None, last_step=None, force_full_denoise=False, denoise_mask=None, sigmas=None, callback=None, disable_pbar=False, seed=None):
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)

        sampler = sampler_object(self.sampler)

        return sample(self.model, noise, positive, negative, cfg, self.device, sampler, sigmas, self.model_options, latent_image=latent_image, denoise_mask=denoise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)
