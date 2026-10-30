# BioProBench → RL Environments: Summary

**Goal:** Build RL environments for LLM post-training from the protocols in BioProBench.

**Scope:** This document covers **PQA, ERR, ORD, and REA** only. REA means REA-ERR, the reasoning version of ERR. GEN and REA-GEN are left out on purpose.

**Repo:** Forked to https://github.com/alexberlaga/bioprobench. In the local clone, `origin` is the fork and `upstream` is `YuyangSunshine/bioprobench`.

---

## 1. What is in this repository

BioProBench (Liu et al., ICML 2026) is a benchmark that tests LLMs on biological lab protocols. It is built from about 22k protocols taken from protocols.io, Nature Protocols, and similar sources. The repo has prompts, inference scripts, and evaluation scripts. **It has no training code.**

| Path | Contents |
|---|---|
| `Data/` | **Test splits only**: `PQA_test.json` (1200), `ERR_test.json` (1200), `ORD_test.json` (1161). The train splits and the raw protocol corpus are on Hugging Face (`huggingface.co/BioProBench`). |
| `Scripts/prompt_format.py` | Prompt template for each task. Every task asks for the answer between `[ANSWER_START]...[ANSWER_END]`. |
| `Scripts/generate_response*.py` | Inference through an OpenAI-compatible API or a local HF model. |
| `Scripts/LLM-as-a-judge_for_REA-ERR.py` | Uses an LLM judge (DeepSeek) to check whether a model's reasoning finds the annotated error. |
| `Metrics/{PQA,ERR,ORD,REA-ERR}.py` | Evaluation for each task. Each script takes a whole output file and prints aggregate metrics. |
| `skills/evaluate-protocol-outputs/` | A cleaner, standalone rewrite of the metrics with a CLI (`scripts/evaluate.py`). |

---

## 2. Tasks

| Task | Input | Expected output | Test composition |
|---|---|---|---|
| **PQA** | A protocol sentence with a blank (`____`) and 5 choices | The chosen option and a confidence from 0 to 100, as `choice & confidence` | 1200 items; parameter 407, reagent 398, operation 395 |
| **ERR** | One protocol step plus context (purpose, prior step, next step) | `True` (step is correct) or `False` (step has an error) | 1200 items; 602 correct, 598 corrupted (parameter 200, reagent 199, operation 199) |
| **ORD** | A title and a shuffled list of steps | A list of the original indices in the correct order | 1161 items; 400 `top` (sections), 761 `child` (sub-steps); 3 to 30 steps, median about 5 |
| **REA** | Same as ERR, with a prompt that asks for explicit reasoning over operation, reagent, and parameter errors | Reasoning, then `True`/`False` | Same data as ERR. The reasoning is judged only on the 598 corrupted items, because only those have an `error_description`. |

---

## 3. What a "trainer" / "RL framework" is, and whether we need to pick one

An RL environment for LLMs has two parts: **prompts** and a **reward function** that scores each completion. A trainer runs the loop: it generates several completions per prompt, scores them with the reward, updates the model toward the higher-scoring ones, and repeats. The usual algorithms are GRPO and PPO.

| Trainer | Main difference |
|---|---|
| **TRL** (`GRPOTrainer`, Hugging Face) | Easiest to set up. Good on 1 to a few GPUs. Reward is `f(completions, **dataset_columns) -> list[float]`. |
| **verl / OpenRLHF** | Built for speed at scale (many GPUs or nodes, vLLM generation). Needs more configuration. |
| **verifiers / prime-rl** | The environment is its own installable package. Good for sharing and for multi-turn tasks. |

**We do not need to pick one now.** The environment is the same for all of them: one prompt builder and one per-sample reward per task. Plan:

1. Write the environments as plain Python with no framework dependency: prompt builders, `reward(sample, completion) -> float`, and tests.
2. Add a thin TRL adapter for a small single-GPU test that checks the rewards work end to end.
3. Add verl or prime-rl adapters later if needed. Each is a small job.

The data is all single-turn (one prompt, one answer, one score), so no multi-turn environment is needed.

---

## 4. Do we have a reward?

**Not a ready-made one.** `Metrics/*.py` contains evaluation scripts that read a file of outputs and report averages. RL needs a function that scores **one completion**. The parsing and comparison logic in those scripts can be reused, and the data contains the gold answers, so we can write the rewards ourselves:

| Task | Gold label in data | Proposed reward |
|---|---|---|
| **PQA** | `answer` | 1 if the parsed choice equals `answer`, else 0. Optional calibration term from confidence (Brier score). |
| **ERR** | `is_correct` | 1 if the parsed True/False equals `is_correct`, else 0. |
| **ORD** | `correct_steps` | 1 for an exact match, or partial credit from Kendall's τ rescaled to [0, 1]. |
| **REA** | `is_correct` + `error_description` | The True/False part works like ERR. On corrupted items, an optional bonus from an LLM judge if the reasoning finds the annotated error. |

For every task, unparsable output (missing tags, invalid index list, confidence above 100, and so on) gets reward 0 or a small negative value.

**Data split:** Train on the Hugging Face train splits. Keep `Data/*_test.json` held out for evaluation. Check how much train and test overlap before relying on test scores.

---

## 5. Note: potential false positives / false negatives in the rewards

A **false positive** means the reward goes to an answer that does not deserve it: a wrong answer, a format hack, or a lucky guess. A **false negative** means a correct answer gets no reward. Under RL, false positives matter most, because the policy will learn to exploit them. Everything below comes from reading the existing parsers in `Metrics/` and from the data checks on the test splits.

### PQA
- **FN, strict string match.** `Metrics/PQA.py` gives credit only if `answer == item['answer']` exactly. Answers like `0.3 mg/kg`, `"0.3"`, or `0.30` score 0 even though they are correct. *Mitigation:* normalize whitespace, quotes, and case, and map the answer to the nearest choice. Score as wrong anything that does not map to exactly one choice.
- **FN, first tag pair wins.** The parser uses `re.search`, so it takes the *first* `[ANSWER_START]...[ANSWER_END]`. If the model repeats the template text earlier in its reasoning, the real answer is ignored. *Mitigation:* use the last tag pair, as ORD does.
- **FN, confidence parsing.** The parser takes the first integer it finds, so a confidence written as `0.8` becomes `0`. This hurts only the calibration term.
- **FP, guessing.** With 5 choices, a random pick is right 20% of the time. This is normal noise, but the baseline reward is not 0.
- **FP, calibration hacking.** If a Brier term is added, always reporting confidence near the base rate can earn reward without any knowledge. Keep the calibration weight small next to correctness.
- Data checks passed: no choice contains `&`, there are no duplicate choices, and the gold answer is always among the choices.

### ERR
- **FP, substring matching (serious).** `extract_binary_answer` checks `'True' in answer or 'true' in answer` *before* checking for False. Consequences:
  - Copying the template `[ANSWER_START]True or False[ANSWER_END]` parses as **True**. That is right about 50% of the time with no reasoning at all.
  - `untrue`, `not true`, and answers that contain both words all parse as **True**.

  *Mitigation:* require the stripped content to be exactly `True` or `False` (case-insensitive). Anything else gets 0.
- **FP, fallback to the last line.** When the tags are missing, the parser uses the last line of the output. That rewards outputs that ignore the format and makes the substring problem worse. *Mitigation:* drop the fallback for training.
- **FP, constant-answer baseline.** The classes are balanced (602/598), so always answering one way gets about 50% reward. Watch for collapse to a single label. Track per-class accuracy (recall on corrupted vs. correct), not only mean reward.
- **Label noise (FN and FP).** In **66 of 598 corrupted items** (`is_correct=False`), `corrupted_text` is **identical** to `corrected_text`. The step shown to the model is the correct one, yet the label says it has an error (29 operation, 28 reagent, 9 parameter). A model that correctly answers "True" gets 0, and a model is rewarded for calling a correct step wrong. *Mitigation:* remove these items, or relabel them as correct.
- **Ambiguity.** Some corruptions may be plausible variants that are still valid in practice, such as alternative concentrations. Some "correct" steps may have problems that the context does not show. A binary label cannot capture this, so expect some irreducible noise.

### ORD
- **FN, multiple valid orders.** Only one order counts as correct. Steps that can run in either order (for example, preparing two independent reagents) are penalized when swapped. Kendall's τ partial credit softens this, but there is no ground truth for which steps are interchangeable.
- **FN, strict index parsing.** The answer must be a Python list literal that contains each index exactly once. 1-based indices, step text instead of indices, or a trailing comment all score 0. 1-based indices are an easy mistake despite the prompt. *Mitigation:* keep the strict check (it is honest), but use the parse-failure rate during training as a diagnostic.
- **Duplicate steps.** **16 test items** have identical step strings, such as two `Centrifuge at 100,000×g for 70 min.` steps. Exact match compares strings, so swapping identical steps is correctly accepted. However, `calculate_kendall_tau` builds `{step: rank}`, which merges duplicates and distorts τ on those items. *Mitigation:* compute τ over index positions, matching identical strings to each other.
- **FP, short items.** 249 items have only 3 steps, so random guessing gets an exact match 1 in 6 times. `top`-level items (section titles) can often be ordered from general knowledge of how lab protocols are structured, without reading the content.
- Data checks passed: no item is already in order, and `wrong_steps` is always a permutation of `correct_steps`.

### REA (reasoning + judge)
- **The True/False part inherits every ERR issue above**, including the substring false positives and the 66 mislabeled items.
- **FP, lenient judge.** The judge prompt asks whether the response "identifies the same problem… even if it finds other unrelated issues." A response that lists many possible errors can hit the right one by luck and get credit. Under RL this pushes the policy toward long, hedged lists of every possible error. *Mitigations:*
  - Give judge credit only when the final answer is also correct (`False`).
  - Penalize the number of distinct errors claimed.
  - Use a stricter judge prompt.
- **FP/FN, judge errors.** The judge is itself an LLM. It may reward vague text that mentions the right reagent or parameter without stating the actual problem (FP), or miss a correct explanation worded differently from `error_description` (FN). Audit a sample of judge decisions by hand before training.
- **FN, judge parsing.** The judge's output is parsed with the same substring-based `extract_binary_answer`, so the substring false positives above also affect the judge's verdict.
- **Coverage gap.** The 602 correct steps have no `error_description`, so the judge never sees them. On those items, the reasoning is not checked at all, only the final True/False.
- **Hacking the judge.** If the policy learns to quote the target step and list operation, reagent, and parameter concerns in a fixed template, it can get through a lenient judge. Track the judge-reward rate against the final-answer accuracy. If the first rises and the second does not, the judge is being exploited.
- Existing bug: `Metrics/REA-ERR.py` prints `results['LLM_Judge_Accuracy']`, but the function returns the key `Consistency`, so the script fails with a `KeyError` at the end.

---

## 6. Proposed next steps

1. Download the HF train splits and check train/test overlap.
2. Write strict per-sample reward functions for PQA, ERR, and ORD. Fix the parser issues above: exact True/False, last tag pair, normalized PQA matching, index-based τ.
3. Remove or relabel the 66 mislabeled ERR items, and check the train split for the same problem.
4. Write unit tests covering the cases above: template echo, `untrue`, 1-based indices, duplicate steps, unit suffixes.
5. Add the REA judge reward last, behind a correct-answer gate, after a manual audit of judge decisions.
6. Write a thin TRL adapter and run a small single-GPU test.
