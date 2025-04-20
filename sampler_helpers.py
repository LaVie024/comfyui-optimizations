from __future__ import annotations
import uuid
import comfy.model_management
import comfy.conds
import comfy.utils
import comfy.hooks
import comfy.patcher_extension
from typing import TYPE_CHECKING
from functools import lru_cache

if TYPE_CHECKING:
    from comfy.model_patcher import ModelPatcher
    from comfy.model_base import BaseModel
    from comfy.controlnet import ControlBase

# Caches for optimizations
_additional_models_cache: dict[tuple[int,int], tuple[list, int]] = {}
_memory_required_cache: dict[tuple[int, tuple[int, ...]], int] = {}

def prepare_mask(noise_mask, shape, device):
    return comfy.utils.reshape_mask(noise_mask, shape).to(device)

def get_models_from_cond(cond, model_type):
    models = []
    for c in cond:
        if model_type in c:
            if isinstance(c[model_type], list):
                models += c[model_type]
            else:
                models += [c[model_type]]
    return models

def get_hooks_from_cond(cond, full_hooks: comfy.hooks.HookGroup):
    # collect any explicit hooks and control‐net extra_hooks
    cnets: list[ControlBase] = []
    for c in cond:
        if 'hooks' in c:
            for hook in c['hooks'].hooks:
                full_hooks.add(hook)
        if 'control' in c:
            cnets.append(c['control'])

    def recurse_extra(cnet: ControlBase, lst: list):
        if cnet.extra_hooks is not None:
            lst.append(cnet.extra_hooks)
        if cnet.previous_controlnet is not None:
            recurse_extra(cnet.previous_controlnet, lst)

    hooks_list = []
    for base in set(cnets):
        recurse_extra(base, hooks_list)

    extra = comfy.hooks.HookGroup.combine_all_hooks(hooks_list)
    if extra is not None:
        for h in extra.hooks:
            full_hooks.add(h)

    return full_hooks

def convert_cond(cond):
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        temp["uuid"] = uuid.uuid4()
        out.append(temp)
    return out

def get_additional_models(conds, dtype):
    """loads additional models in conditioning (cached per conds id & dtype)"""
    key = (id(conds), dtype)
    if key in _additional_models_cache:
        return _additional_models_cache[key]

    cnets, gligen, add_models = [], [], []
    for k in conds:
        cnets += get_models_from_cond(conds[k], "control")
        gligen += get_models_from_cond(conds[k], "gligen")
        add_models += get_models_from_cond(conds[k], "additional_models")

    control_nets = set(cnets)
    control_models, inference_memory = [], 0
    for m in control_nets:
        control_models += m.get_models()
        inference_memory += m.inference_memory_requirements(dtype)

    gligen_models = [x[1] for x in gligen]
    models = control_models + gligen_models + add_models

    _additional_models_cache[key] = (models, inference_memory)
    return models, inference_memory

def get_additional_models_from_model_options(model_options: dict[str]=None):
    """loads additional models from registered AddModels hooks"""
    models = []
    if model_options is not None and "registered_hooks" in model_options:
        registered: comfy.hooks.HookGroup = model_options["registered_hooks"]
        for hook in registered.get_type(comfy.hooks.EnumHookType.AdditionalModels):
            hook: comfy.hooks.AdditionalModelsHook
            models.extend(hook.models)
    return models

def cleanup_additional_models(models):
    """cleanup additional models that were loaded"""
    for m in models:
        if hasattr(m, 'cleanup'):
            m.cleanup()

def prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None):
    executor = comfy.patcher_extension.WrapperExecutor.new_executor(
        _prepare_sampling,
        comfy.patcher_extension.get_all_wrappers(
            comfy.patcher_extension.WrappersMP.PREPARE_SAMPLING,
            model_options,
            is_model_options=True
        )
    )
    return executor.execute(model, noise_shape, conds, model_options=model_options)

def _prepare_sampling(model: ModelPatcher, noise_shape, conds, model_options=None):
    real_model: BaseModel
    models, inference_memory = get_additional_models(conds, model.model_dtype())
    models += get_additional_models_from_model_options(model_options)
    models += model.get_nested_additional_models()

    # Cached memory calculations
    full_shape = tuple([noise_shape[0] * 2] + list(noise_shape[1:]))
    key_full = (id(model), full_shape)
    if key_full in _memory_required_cache:
        memory_required = _memory_required_cache[key_full] + inference_memory
    else:
        mr = model.memory_required(full_shape)
        _memory_required_cache[key_full] = mr
        memory_required = mr + inference_memory

    min_shape = tuple([noise_shape[0]] + list(noise_shape[1:]))
    key_min = (id(model), min_shape)
    if key_min in _memory_required_cache:
        minimum_memory_required = _memory_required_cache[key_min] + inference_memory
    else:
        mr_min = model.memory_required(min_shape)
        _memory_required_cache[key_min] = mr_min
        minimum_memory_required = mr_min + inference_memory

    comfy.model_management.load_models_gpu(
        [model] + models,
        memory_required=memory_required,
        minimum_memory_required=minimum_memory_required
    )
    real_model = model.model

    return real_model, conds, models

def cleanup_models(conds, models):
    cleanup_additional_models(models)
    control_cleanup = []
    for k in conds:
        control_cleanup += get_models_from_cond(conds[k], "control")
    cleanup_additional_models(set(control_cleanup))

def prepare_model_patcher(model: ModelPatcher, conds, model_options: dict):
    HookGroup = comfy.hooks.HookGroup
    merge_dicts = comfy.patcher_extension.merge_nested_dicts

    # Gather all hooks
    hooks = HookGroup()
    for k in conds:
        get_hooks_from_cond(conds[k], hooks)

    # Inject wrappers & callbacks
    model_options["transformer_options"]["wrappers"] = comfy.patcher_extension.copy_nested_dicts(model.wrappers)
    model_options["transformer_options"]["callbacks"] = comfy.patcher_extension.copy_nested_dicts(model.callbacks)

    # Register weight / transformer / additional‐models hooks
    registered = HookGroup()
    target_dict = comfy.hooks.create_target_dict(comfy.hooks.EnumWeightTarget.Model)

    for hook in hooks.get_type(comfy.hooks.EnumHookType.TransformerOptions):
        hook: comfy.hooks.TransformerOptionsHook
        hook.add_hook_patches(model, model_options, target_dict, registered)

    for hook in hooks.get_type(comfy.hooks.EnumHookType.AdditionalModels):
        hook: comfy.hooks.AdditionalModelsHook
        hook.add_hook_patches(model, model_options, target_dict, registered)

    model.register_all_hook_patches(hooks, target_dict, model_options, registered)

    if len(registered) > 0:
        model_options["registered_hooks"] = registered

    to_load = model_options.setdefault("to_load_options", {})
    for wc_name in ("wrappers", "callbacks"):
        merge_dicts(
            to_load.setdefault(wc_name, {}),
            model_options["transformer_options"][wc_name],
            copy_dict1=False
        )

    return to_load
