import yaml, sys

regime_map = {
    "baseline": "configs/baseline.yaml",
    "hinton": "configs/hinton.yaml",
    "multi_teacher_fixed": "configs/multi_teacher_fixed.yaml",
    "arcd": "configs/arcd.yaml",
    "arcd_topk": "configs/arcd_topk.yaml",
}

for name, path in regime_map.items():
    cfg = yaml.safe_load(open(path))
    cfg["training"]["epochs"] = 30
    cfg["training"]["early_stopping_patience"] = 8
    cfg["output"]["checkpoint_dir"] = "outputs/checkpoints_patient"
    out_path = f"configs/{name}_patient.yaml"
    yaml.dump(cfg, open(out_path, "w"))
    print(f"{out_path} créé")
