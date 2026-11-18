"""LoRA configuration from the original run_clm_finetune.py."""


def attach(model, args):
    from peft import LoraConfig, get_peft_model

    target_modules = args.lora_target_modules
    if isinstance(target_modules, str):
        # Try comma separation first, then space separation
        if "," in target_modules:
            target_modules = target_modules.split(",")
        else:
            target_modules = target_modules.split()

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()
    return model
