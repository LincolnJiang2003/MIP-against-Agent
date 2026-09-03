from types import SimpleNamespace

import torch
from PIL import Image

from mip_attack.model import Llama32


class FakeBatch(dict):
    def to(self, _device):
        return self


class FakeTokenizer:
    def __call__(self, _text, **_kwargs):
        return SimpleNamespace(input_ids=torch.tensor([[1, 2]]))

    def decode(self, output_ids):
        return " ".join(str(token) for token in output_ids.tolist())


class FakeProcessor:
    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, **_kwargs):
        return "prompt"

    def __call__(self, **_kwargs):
        return FakeBatch(input_ids=torch.tensor([[7, 8]]))

    def decode(self, output_ids):
        return self.tokenizer.decode(output_ids)


class FakeVLM:
    def __init__(self):
        self.device = torch.device("cpu")
        self.generation_config = SimpleNamespace(eos_token_id=9)
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        batch_size = kwargs["num_return_sequences"]
        prompt = kwargs["input_ids"].expand(batch_size, -1)
        generated = torch.tensor([[1, 2]]).expand(batch_size, -1)
        scores = []
        for token_id in (1, 2):
            step_scores = torch.zeros(batch_size, 4)
            step_scores[:, token_id] = 4.0
            scores.append(step_scores)
        return {"sequences": torch.cat((prompt, generated), dim=1), "scores": scores}


def test_stochastic_evaluation_generations_are_batched_by_temperature():
    model_name = "Llama-3.2-11B-Vision-Instruct"
    llama = object.__new__(Llama32)
    llama.models = [model_name]
    llama.processors = {model_name: FakeProcessor()}
    llama.vlms = {model_name: FakeVLM()}
    llama.prepare_text_images_individually = False

    results = llama.process_images(
        system_prompt="system",
        question="question",
        images=[Image.new("RGB", (1, 1))],
        temperatures=[0.0, 0.5, 1.0, 1.5],
        num_generations=5,
        malicious_generation="target",
    )

    calls = llama.vlms[model_name].calls
    assert len(calls) == 4
    assert [call["num_return_sequences"] for call in calls] == [1, 5, 5, 5]
    assert [call["do_sample"] for call in calls] == [False, True, True, True]
    assert [len(results[temperature]["output_ids"]) for temperature in results] == [
        1,
        5,
        5,
        5,
    ]
    assert all(
        all(results[temperature]["sample_matches_target"])
        for temperature in results
    )
