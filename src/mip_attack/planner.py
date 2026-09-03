from mip_attack import prompts
from mip_attack.model import Llama32


class Llama32Planner:
    def __init__(
        self,
        models,
        prepare_text_images_individually=False,
        model_revisions=None,
        gpu_memory_reserve_gib=2,
    ):
        self.model = Llama32(
            models,
            prepare_text_images_individually,
            model_revisions=model_revisions,
            gpu_memory_reserve_gib=gpu_memory_reserve_gib,
        )
        self.system_prompt = prompts.planning_system_message

    def plan(
        self,
        images,
        user_query,
        temperatures,
        num_generations,
        malicious_generation,
    ):
        return self.model.process_images(
            system_prompt=self.system_prompt,
            question=user_query,
            images=images,
            temperatures=temperatures,
            num_generations=num_generations,
            malicious_generation=malicious_generation,
        )

    def attack(self, **kwargs):
        return self.model.adversarial_attack(
            system_prompt=self.system_prompt,
            **kwargs,
        )
