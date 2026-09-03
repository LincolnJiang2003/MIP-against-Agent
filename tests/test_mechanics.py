import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]
MODEL_PATH = ROOT / "src/mip_attack/model.py"
GROUNDING_DINO_ROOT = ROOT / "src/mip_attack/parsing/groundingdino"


def _method(name):
    module = ast.parse(MODEL_PATH.read_text())
    model_class = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "Llama32"
    )
    return next(
        node
        for node in model_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def test_crafting_loop_is_bounded_by_optimizer_steps():
    method = _method("adversarial_attack")
    loops = [node for node in ast.walk(method) if isinstance(node, ast.While)]
    bounded = [
        loop for loop in loops if ast.unparse(loop.test) == "optimizer_step < max_optimizer_steps"
    ]
    assert len(bounded) == 1


def test_crafting_freezes_models_and_disables_cache():
    source = MODEL_PATH.read_text()
    assert ".eval()" in source
    assert "param.requires_grad = False" in source
    assert "use_cache=False" in ast.unparse(_method("adversarial_attack"))


def test_generation_metrics_follow_sampled_tokens():
    source = ast.unparse(_method("process_images"))
    assert "'output_scores': True" in source
    assert "'num_return_sequences': generation_count" in source
    assert "sequence_index=sequence_index" in source
    assert "trim_generated_ids(output_ids, eos_token_id)" in source
    assert "exact_token_match(output_ids, malicious_target_ids[0])" in source
    assert "generation_options['temperature'] = None" in source
    assert "print(output_str)" not in source


def test_each_model_loss_is_averaged_over_models_and_settings():
    source = ast.unparse(_method("adversarial_attack"))
    assert "model_count=len(self.models)" in source
    assert "settings_per_update=settings_per_update" in source


def test_crafting_progress_tracks_optimizer_steps_and_formats_metrics():
    source = ast.unparse(_method("adversarial_attack"))
    assert "total=max_optimizer_steps" in source
    assert "dynamic_ncols=True" in source
    assert "progress_bar.update(1)" in source
    assert "argmax_rate" in source
    assert source.count(".4f") >= 6


def test_groundingdino_inference_avoids_training_checkpointing_and_deprecated_apis():
    loader = (GROUNDING_DINO_ROOT / "icon_localization.py").read_text()
    transformer = (
        GROUNDING_DINO_ROOT / "models/GroundingDINO/transformer.py"
    ).read_text()
    bertwarper = (
        GROUNDING_DINO_ROOT / "models/GroundingDINO/bertwarper.py"
    ).read_text()

    assert "args.use_checkpoint = False" in loader
    assert "args.use_transformer_ckpt = False" in loader
    assert "torch.cuda.amp.autocast" not in transformer
    assert "attention_mask, input_shape, device" not in bertwarper
