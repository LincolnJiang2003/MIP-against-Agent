from PIL import Image

from mip_attack import prompts
from mip_attack.artifacts import active_run_root
from mip_attack.parser import AgentParser


def previous_actions_text(previous_actions, limit=3):
    return "".join(
        (
            "Screen is currently at time step T. Below is the action executed "
            f"at time step T-{offset}: \n{action}\n\n"
        )
        for offset, action in enumerate(reversed(previous_actions[-limit:]), start=1)
    )


def place_png_patch(
    image,
    patch_path,
    x_range,
    y_range,
    crop_from_full_image=False,
):
    with Image.open(patch_path) as source:
        source = source.convert("RGB")
        patch = (
            source.crop((x_range[0], y_range[0], x_range[1], y_range[1]))
            if crop_from_full_image
            else source.copy()
        )

    expected_size = (x_range[1] - x_range[0], y_range[1] - y_range[0])
    if patch.size != expected_size:
        raise ValueError(
            f"Patch size {patch.size} does not match placement area {expected_size}"
        )
    output = image.convert("RGB")
    output.paste(patch, (x_range[0], y_range[0]))
    return output


class NaviAgent:
    def __init__(
        self,
        models,
        som_origin="oss",
        use_annotated_screen_only=True,
        prepare_text_images_individually=False,
        oss_box_threshold=0.03,
        parser_model_root="models",
        model_revisions=None,
        gpu_memory_reserve_gib=2,
        load_parser=True,
        load_planner=True,
    ):
        self.som_origin = som_origin
        self.use_annotated_screen_only = use_annotated_screen_only
        self.agent_parser = (
            AgentParser(
                model_root=parser_model_root,
                oss_box_threshold=oss_box_threshold,
            )
            if load_parser
            else None
        )
        self.vlm_planner = None
        if load_planner:
            if not models:
                raise ValueError("At least one model is required when loading the planner")
            from mip_attack.planner import Llama32Planner

            self.vlm_planner = Llama32Planner(
                models=models,
                prepare_text_images_individually=prepare_text_images_individually,
                model_revisions=model_revisions,
                gpu_memory_reserve_gib=gpu_memory_reserve_gib,
            )
        self.memory_block_text_empty = """
        ```memory
        # empty memory block
        ```
        """
        self.previous_action_limit = 15
        self.reset()

    def _prompt_images(self, image, annotated_image):
        # Annotated-only lowers crafting cost; ablations found comparable attack
        # effectiveness with WAA's original [raw, annotated] input.
        if self.use_annotated_screen_only:
            return [annotated_image]
        return [image, annotated_image]

    def _user_question(self, instruction, observation, rendering, entities):
        return prompts.build_user_msg_visual(
            instruction,
            observation["window_title"],
            observation["window_names_str"],
            observation["computer_clipboard"],
            rendering,
            entities,
            previous_actions_text(self.prev_actions, self.previous_action_limit),
            self.memory_block_text,
        )

    def attack(
        self,
        obs,
        example_id,
        malicious_generation,
        load_adversarial_patch,
        crop_patch_from_image,
        center_constraint_clipping_factor,
        keep_specific_color,
        max_num_patch_positions,
        resized_values,
        adv_patch_size,
        adv_x_area,
        adv_y_area,
        adv_optimizer,
        adv_lr,
        betas,
        increased_betas,
        termination_likelihood,
        random_pert_intensity,
        settings_per_update,
        epsilon,
        parser_origins,
        max_optimizer_steps=100000,
        checkpoint_interval=10,
        resume=True,
        reset_betas_on_collapse=False,
        run_artifacts=None,
    ):
        if run_artifacts is None:
            raise ValueError("run_artifacts is required for crafting")
        if self.agent_parser is None or self.vlm_planner is None:
            raise RuntimeError("Crafting requires both the parser and planner")
        if self.som_origin not in parser_origins:
            raise ValueError("parser_origins must include the primary parser")

        diagnostics_dir = run_artifacts.crafting(example_id).diagnostics_dir
        settings = []
        last_entities = None
        last_questions = None

        for image_id, image in enumerate(obs["screenshot"]):
            for som_origin in parser_origins:
                _, _, _, annotated, entities, rendering = self.agent_parser.parse(
                    image=image,
                    accessibility_tree=obs["accessibility_tree"],
                    som_origin=som_origin,
                )
                annotated.save(
                    diagnostics_dir
                    / f"{example_id}_annotated_{image_id:02d}_{som_origin}.png"
                )
                questions = [
                    self._user_question(instruction, obs, rendering, entities)
                    for instruction in obs["instruction"]
                ]
                settings.append(
                    {
                        "images": self._prompt_images(image, annotated),
                        "user_query": questions,
                    }
                )
                last_entities, last_questions = entities, questions

        # Parser models are no longer needed once the crafting inputs exist.
        self.agent_parser = None
        adversarial_image = self.vlm_planner.attack(
            example_id=example_id,
            agent_input=settings,
            malicious_generation=malicious_generation,
            load_adversarial_patch=load_adversarial_patch,
            crop_patch_from_image=crop_patch_from_image,
            center_constraint_clipping_factor=center_constraint_clipping_factor,
            keep_specific_color=keep_specific_color,
            max_num_patch_positions=max_num_patch_positions,
            resized_values=resized_values,
            adv_patch_size=adv_patch_size,
            adv_x_area=adv_x_area,
            adv_y_area=adv_y_area,
            adv_optimizer=adv_optimizer,
            adv_lr=adv_lr,
            betas=betas,
            increased_betas=increased_betas,
            termination_likelihood=termination_likelihood,
            random_pert_intensity=random_pert_intensity,
            settings_per_update=settings_per_update,
            epsilon=epsilon,
            max_optimizer_steps=max_optimizer_steps,
            checkpoint_interval=checkpoint_interval,
            resume=resume,
            reset_betas_on_collapse=reset_betas_on_collapse,
            run_artifacts=run_artifacts,
        )
        return adversarial_image, last_entities, last_questions

    def prepare_screen(self, image, accessibility_tree=None, som_origin=None):
        if self.agent_parser is None:
            raise RuntimeError("Screen preparation requires the parser")
        origin = som_origin or self.som_origin
        _, _, _, annotated, entities, rendering = self.agent_parser.parse(
            image=image,
            accessibility_tree=accessibility_tree,
            som_origin=origin,
        )
        return image, annotated, entities, rendering

    def predict_prepared(
        self,
        instruction,
        obs,
        prepared_screen,
        malicious_generation,
        temperatures,
        num_generations,
    ):
        if self.vlm_planner is None:
            raise RuntimeError("Prediction requires the planner")
        image, annotated, entities, rendering = prepared_screen
        question = self._user_question(instruction, obs, rendering, entities)
        results = self.vlm_planner.plan(
            images=self._prompt_images(image, annotated),
            user_query=question,
            temperatures=temperatures,
            num_generations=num_generations,
            malicious_generation=malicious_generation,
        )
        return results, None, entities, question

    def predict(
        self,
        instruction,
        obs,
        example_id,
        patch_id,
        screen_id,
        malicious_generation,
        temperatures,
        num_generations,
        place_adversarial_patch,
        adv_full_images,
        adversarial_patches_on_image,
        adv_x_ranges,
        adv_y_ranges,
    ):
        if len(obs["screenshot"]) != 1 or len(instruction) != 1:
            raise ValueError("Evaluation requires one screenshot and one instruction")

        image = obs["screenshot"][0]
        if place_adversarial_patch:
            lengths = {
                len(adv_full_images),
                len(adversarial_patches_on_image),
                len(adv_x_ranges),
                len(adv_y_ranges),
            }
            if len(lengths) != 1:
                raise ValueError("Patch placement arguments must have equal lengths")
            for full_image, patch_path, x_range, y_range in zip(
                adv_full_images,
                adversarial_patches_on_image,
                adv_x_ranges,
                adv_y_ranges,
            ):
                image = place_png_patch(
                    image,
                    patch_path,
                    x_range,
                    y_range,
                    crop_from_full_image=full_image,
                )

        evaluation_dir = (
            active_run_root()
            / "examples"
            / example_id
            / "evaluation"
            / patch_id
            / str(screen_id)
        )
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        image.save(evaluation_dir / "image_with_adversarial_patch.png")

        prepared_screen = self.prepare_screen(
            image,
            accessibility_tree=obs["accessibility_tree"],
        )
        _, annotated, _, _ = prepared_screen
        annotated.save(evaluation_dir / "annotated_image_with_adversarial_patch.png")
        return self.predict_prepared(
            instruction=instruction[0],
            obs=obs,
            prepared_screen=prepared_screen,
            malicious_generation=malicious_generation,
            temperatures=temperatures,
            num_generations=num_generations,
        )

    def reset(self):
        self.memory_block_text = self.memory_block_text_empty
        self.prev_actions = []
