# Responsible use

This repository demonstrates an attack on multimodal agents that can generate
actionable OS commands. Restrict use to authorized security research.

- Run against synthetic screenshots and disposable, isolated test systems.
- Do not connect the evaluation process to an agent that automatically executes
  its output.
- Do not upload crafted patches to public services or expose uninformed users.
- Do not open a public issue for a vulnerability that could expose users of a
  deployed OS agent. Use GitHub's private vulnerability reporting feature
  for this repository.
- Report affected deployed systems privately to their maintainers before public
  disclosure.
- Do not use real credentials, personal files, or production accounts.
- Follow the licenses and acceptable-use policies of models, datasets, and
  services involved in an experiment.

The target strings are research fixtures. This pipeline scores generated text;
it does not execute those strings.
