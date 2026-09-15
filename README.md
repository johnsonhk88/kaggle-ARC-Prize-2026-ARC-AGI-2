# ARC Prize 2026 - ARC-AGI-2

> Create an AI capable of novel reasoning

Working repository for the [ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview) Kaggle competition, hosted by the [Abstraction and Reasoning Corpus](https://www.kaggle.com/organizations/arc) organization.

- Kaggle competition: https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview
- ARC Prize 2026: https://arcprize.org/competitions/2026
- ARC-AGI-2 benchmark: https://arcprize.org/arc-agi/2

## About This Repository

This repo holds our code and experiments for the ARC-AGI-2 track of ARC Prize 2026.

```
.
└── src/
    └── sample-code/
        └── failed-in-aimo.ipynb
```

- `src/sample-code/failed-in-aimo.ipynb` - a sample Kaggle notebook (originally run in the Kaggle environment). It enforces the 12-hour runtime budget, writes an `arc_loader.py` helper for loading ARC tasks, and experiments with Hugging Face Transformers. It is not a competitive solution; treat it as a starting point for local experiments.

## Competition Description

Today's AI systems excel at what they were trained to do, but often fall short when something unfamiliar comes along. Most benchmarks reward pattern recognition, not genuine problem-solving.

The ARC Prize focuses on true generalization: whether a system can quickly learn new skills in unfamiliar situations. Instead of rewarding pattern recognition on known tasks, it evaluates how well systems adapt to new problems they have never encountered before. The evaluation environment is designed so systems cannot just memorize solutions.

This competition is a relaunch of [ARC Prize 2025](https://www.kaggle.com/competitions/arc-prize-2025/overview). Participants are encouraged to also join the [paper track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track) to document their approach.

**Objective:** reach 85% accuracy on the ARC-AGI-2 private evaluation dataset within the Kaggle efficiency limits.

## Evaluation

Submissions are scored on the percentage of correct predictions.

- For each task, make **2 attempts** to predict the exact outputs for every test input grid contained in the task. A task can have more than one test input that needs a predicted output.
- Each task test output has one ground truth. If **any of the 2 predicted outputs** matches the ground truth exactly, you score `1` for that task test output, otherwise `0`.
- The final score is the sum of the highest score per task output, averaged over the total number of task test outputs.

## Submission File

The submission file must be JSON named `submission.json`.

For each task output in the evaluation set, make exactly 2 predictions (`attempt_1`, `attempt_2`). Many tasks have multiple outputs, so the value is a list of dictionaries; they must be in the same order as the corresponding test inputs. Every task ID present in the input challenges JSON must also be present in `submission.json`, and both `attempt_1` and `attempt_2` must be present even if your submission does not have 2 predictions.

```json
{"00576224": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
 "009d5c81": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}],
 "12997ef3": [{"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]},
              {"attempt_1": [[0, 0], [0, 0]], "attempt_2": [[0, 0], [0, 0]]}]
}
```

## Timeline

| Date | Event |
| --- | --- |
| March 25, 2026 | Start Date |
| October 26, 2026 | Entry Deadline - accept the competition rules before this date to compete |
| October 26, 2026 | Team Merger Deadline - last day to join or merge teams |
| November 2, 2026 | Final Submission Deadline |
| December 4, 2026 | Winners announcement |

All deadlines are at 11:59 PM UTC on the corresponding day unless otherwise noted. Organizers reserve the right to update the timeline.

## Prizes

**Total prizes available: $700,000**

| Award | Amount |
| --- | --- |
| Progress Prizes | $275,000 |
| Grand Prize | $275,000 |
| Bonus Prize | $150,000 |

In line with the spirit of the competition, participants eligible for a prize will be removed from the competition if they do not open source their solutions.

### Progress Prizes ($275,000)

| Place | Prize |
| --- | --- |
| First | $75,000 |
| Second | $50,000 |
| Third | $40,000 |
| Fourth | $35,000 |
| Fifth | $25,000 |
| Sixth | $20,000 |
| Seventh | $15,000 |
| Eighth | $15,000 |

### Grand Prize ($275,000)

Awarded to the highest scoring Solution Writeup. All artifacts must be open sourced and attached to an official competition Solution Writeup within seven days of the submission deadline to be eligible.

Submissions are evaluated equally across six criteria, each scored from 0 (lowest) to 5 (highest), with the final score the average of all six:

| Category | Description |
| --- | --- |
| Accuracy | How accurate is the submission based on its performance on the leaderboard? |
| Universality | How general and universal is the approach beyond the competition? |
| Progress | How much does the solution increase the overall chance of anyone achieving 85% on ARC-AGI-2? |
| Theory | How well do the artifacts describe *why* the submission works, not merely *how*? |
| Completeness | How thoroughly does the solution cover the submission to the leaderboard? |
| Novelty | How novel is the submission relative to existing public research? |

### Bonus Prize ($150,000)

Unlocked if a team achieves at least 85% accuracy on the competition leaderboard, then divided among the top 5 teams that reached 85%:

| Place | Prize |
| --- | --- |
| First | $75,000 |
| Second | $25,000 |
| Third | $20,000 |
| Fourth | $20,000 |
| Fifth | $10,000 |

If fewer than 5 teams achieve 85% accuracy, the prizes are divided proportionately among qualifying teams.

## Code Requirements

This is a **Code Competition**: submissions must be made through Kaggle Notebooks. For the "Submit to Competition" button to be active after a commit:

- CPU Notebook <= 12 hours run-time
- GPU Notebook <= 12 hours run-time
- No internet access enabled
- External data, freely and publicly available, is allowed, including pre-trained models
- Submission file must be named `submission.json`

Submission runtimes are obfuscated; repeating the exact same submission may show up to 10 minutes of variance before receiving a score. See the [Code Competition FAQ](https://www.kaggle.com/docs/competitions#kernels-only-FAQ) for details.

### Upgraded Accelerators

The competition has access to Kaggle's pool of L4x4 machines (96GB of GPU memory), enabling submissions with much larger models.

- **Quota usage** - L4x4 notebooks consume GPU quota at twice the rate of T4x2 and P100 machines.
- **Restricted use** - L4s are only available for notebooks attached to this competition; attempts to circumvent this may lead to team or account bans.
- **No internet** - all L4 sessions must have internet disabled.

### Open Source & Eligibility

All leading participants are expected to open source their solutions to be eligible for a prize. In order for a submission to be eligible, all code and methods authored by the submitter must be made open source under a permissive public domain license (e.g. CC0 or MIT-0). Any third-party code or methods not authored by the submitter must be available under at least an open source license that allows public sharing (e.g. Apache-2.0, GPLv3). Internet access is not available during Kaggle evaluation (no API-based systems such as GPT/Claude/etc.).

See the [ARC Prize 2026 rules](https://arcprize.org/competitions/2026) for full details.

## Quick Start

The sample notebook is designed for the Kaggle environment (no internet during evaluation, 12-hour limit). To run it locally:

```bash
git clone git@github.com:johnsonhk88/kaggle-ARC-Prize-2026-ARC-AGI-2.git
cd kaggle-ARC-Prize-2026-ARC-AGI-2
jupyter notebook src/sample-code/failed-in-aimo.ipynb
```

A Python environment with `transformers`, `torch`, `numpy`, and `jupyter` is required. A GPU is strongly recommended. Note that local runs do not reproduce Kaggle's evaluation environment exactly.

## Participation

As of September 2026: 9,181 entrants, 2,136 participants, 2,027 teams, and 18,506 submissions.

## Citation

Francois Chollet, Mike Knoop, Greg Kamradt, Walter Reade, María Cruz, and Addison Howard. ARC Prize 2026 - ARC-AGI-2. https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2, 2026. Kaggle.