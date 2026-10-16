# MLJ Contribution Information Sheet

**Manuscript:** Representation-Dependent Predictive Uncertainty in Autoregressive Language Models  
**Article type:** Regular research article

## Main claim and importance to machine learning

The main claim is that, on matched procedural reference trajectories, next-token predictive uncertainty in autoregressive language models varies systematically with the representation of the procedure. Representation should therefore be considered when interpreting token-level uncertainty measurements; otherwise, differences attributed to task content or model behavior may partly reflect the form of the target sequence. We are not claiming a universal ranking of representations or that lower predictive entropy implies a more accurate solution.

## Evidence supporting the claim

We use a frozen corpus of **100 procedural tasks from 50 families** and teacher-forced evaluation of complete matched reference trajectories. The primary measure is renormalized top-20 next-token entropy averaged at the task level. The matched natural-language-minus-Python contrast is positive in **100/100 tasks** for each of three model families: **Qwen3-4B: 0.3605** (99% bootstrap interval [0.3231, 0.3977]); **Phi-3-mini-4k-instruct: 0.6592** ([0.6079, 0.7109]); and **Mistral-7B-Instruct-v0.3: 0.6353** ([0.5831, 0.6870]). All 50 task families show the same sign in each baseline model comparison. Comparisons of absolute effect magnitudes across tokenizers are not interpreted.

Sequential representation conditions isolate more specific contrasts. Controlled natural language minus pseudocode and pseudocode minus canonical Python are positive in all three models. Opaque-identifier Python minus canonical Python is likewise positive in all three models, while the procedural-prose-minus-controlled-natural-language contrast is not supported in Mistral (mean −0.0058; 99% interval [−0.0320, 0.0199]). Valid-vocabulary entropy, semantic-step balancing, bootstrap inference, sign-flip testing, and family-level checks support the stated boundaries of the conclusions. The comparisons do not separately identify lexical, syntactic, or tokenization mechanisms.

## Closely related literature and distinction

Hindle et al. (2012) and Allamanis et al. (2018) establish statistical regularities and the naturalness of source code. Puerto et al. (2024), Chae et al. (2024), and Zhang et al. (2023) examine task performance under code-like or pseudocode-based prompting. Zhu et al. (2026) investigate token-level decoding difficulty and its structural localization within natural-language and code-generation settings. Miceli Barone et al. (2023) and Li et al. (2026) show sensitivity to identifier swaps and tokenization-changing code rewrites. We do not claim priority over these observations. Our distinguishing evidence comes from task-matched representations of the same procedural content, teacher-forced full reference trajectories, cross-model replication, and a sequence of intermediate representation contrasts that includes a negative/non-universal result.

## Previous publication and overlap

This manuscript was previously submitted to *Transactions on Machine Learning Research* and received an editorial rejection without external reviewer reports on **16 October 2026**; it is no longer under consideration there. No prior peer-reviewed conference or journal publication of this manuscript is identified in the supplied record. **Before submission, verify the status and identifier of any public preprint or ResearchGate posting of this first article, and disclose it accurately in the submission system and cover letter if applicable.** A public reproducibility repository accompanies this study.