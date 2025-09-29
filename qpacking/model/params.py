"""
# ------------------------------------------------------------------------------
# Author:    Dou Zhixin
# Email:     bj600800@gmail.com
# DATE:      2025/9/29

# Description: model params processing
# ------------------------------------------------------------------------------
"""
from qpacking.common import logger
logger = logger.setup_log(name=__name__)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
            logger.info(f"{name} 🔥 Trainable")
        else:
            logger.info(f"{name} 🧊 Frozen")
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable: {round(100 * trainable_params / all_param, 2)}%"
    )


def unfreeze_backbone(model, unfreeze_last_n, model_prefix):
    """
    Freeze the backbone model parameters.
    """
    # unfrozen the last layers
    if unfreeze_last_n > 0:
        layer_nums = set()
        for name, _ in model.named_parameters():
            if model_prefix in name:
                try:
                    layer_id = int(name.split(model_prefix + ".")[1].split(".")[0])
                    layer_nums.add(layer_id)
                except:
                    continue

        if not layer_nums:
            print("❌ No encoder found")

        else:
            total_layers = max(layer_nums) + 1
            target_layers = list(range(total_layers - unfreeze_last_n, total_layers))
            print(f"Total {total_layers} layers, unfrozen the last {unfreeze_last_n} layers: {target_layers}")

            matched_count = 0
            for name, param in model.named_parameters():
                if any(f"{model_prefix}.{i}." in name for i in target_layers):
                    param.requires_grad = True
                    matched_count += 1
                    # print(f"✅ Unfroze: {name}")
                else:
                    param.requires_grad = False

            print(f"\n✅ unfrozen {matched_count} params")

    else:
        for _, param in model.named_parameters():
            param.requires_grad = False
        print("❄️ Frozen all params")
