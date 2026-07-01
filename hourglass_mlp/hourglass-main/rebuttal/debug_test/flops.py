import torch
from thop import profile


def get_model_stats(model, device, input_dim, output_dim):
    dummy_input = torch.randn(1, input_dim).to(device)
    dummy_target = torch.randn(1, output_dim).to(device)
    flops_per_block = {}
    skipadd_total = 0
    def hook_fn(m, inputs, outputs):
        nonlocal skipadd_total
        if hasattr(m, "__flops__"):
            fl, _ = m.__flops__(inputs, outputs)
            block_id = getattr(m, "block_id", None)
            name = type(m).__name__
            if block_id is not None:
                flops_per_block.setdefault(f"block{block_id}", {})
                flops_per_block[f"block{block_id}"][name] = flops_per_block[f"block{block_id}"].get(name, 0) + fl
            else:
                flops_per_block.setdefault("others", {})
                flops_per_block["others"][name] = flops_per_block["others"].get(name, 0) + fl
            if name == "SkipAdd":
                skipadd_total += fl
    hooks = [m.register_forward_hook(hook_fn) for m in model.modules()]
    macs, params = profile(model, inputs=(dummy_input, dummy_target), verbose=False)
    total_flops = int(macs * 2)
    for h in hooks:
        h.remove()
    total_flops += skipadd_total
    total_params = int(params)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    skipadd_ratio = skipadd_total / total_flops if total_flops > 0 else 0
    return total_params, total_flops, trainable_params, skipadd_ratio, flops_per_block

def filter_flops_info(total_params, total_flops, trainable_params, skipadd_ratio, flops_per_block):
    filtered_blocks = {}
    for block, modules in flops_per_block.items():
        if "SkipAdd" in modules:
            filtered_blocks[block] = {"SkipAdd": modules["SkipAdd"]}
    return {
        "total_params": total_params,
        "total_flops": total_flops,
        "trainable_params": trainable_params,
        "skipadd_flops_ratio": skipadd_ratio,
        "flops_per_block": filtered_blocks
    }
def skipadd_flops_counter(self, inputs, outputs):
        return inputs[0].numel(), 0
    
