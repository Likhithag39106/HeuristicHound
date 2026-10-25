# Heuristic Hound — Project Knowledge Base

Everything decided across planning so far. Java code smell detection, trained on developer ratings, evaluated on unseen projects.

---

## 1. What the project is

A code smell detector for Java that learns from professional developers' ratings instead of fixed numeric rules, and is tested on projects it has never seen.

**One line:** take 26 professional developers' ratings of real Java code, train two different kinds of model on it, and — unlike previous work — test on projects the model has never seen.

### Motivation
- A large share of code pushed to GitHub is now AI-generated, and it lands faster than humans can review it.
- AI writes code one prompt at a time without seeing the rest of the project, so it repeats logic, creates oversized classes, and ignores team conventions.
- Existing tools (SonarQube, PMD) use fixed thresholds — "flag any method over 50 lines" — so they miss real problems and raise false alarms. Developers stop trusting them and switch them off.
- A tool that *learns* what bad code looks like from real developer judgments would be more useful.

**Note:** the "50% of GitHub code is AI-written" figure needs a real source or should be softened to "a rapidly growing share." A reviewer will ask.

**Also note:** the motivation is about AI code, but the experiments run on human-written Java from 2019. That gap is visible. Either add a small AI-vs-human comparison, or frame the motivation around code volume outpacing review capacity.

### Problem statement
Code is being written faster than it can be reviewed, and a growing share of it is AI-generated without awareness of the wider project — so structural problems accumulate faster than teams catch them. Existing detectors use fixed numeric thresholds, so they raise false alarms and get ignored. Machine-learning alternatives report better numbers, but they are trained and tested on code from the same projects, meaning they partly memorise project style instead of learning what a code smell is.

This project builds a Java code smell detector trained on professional developers' ratings and evaluated honestly — trained on one set of projects, tested on entirely different ones. It covers four smells, compares two ways of representing code to the model, and measures how much prior accuracy was inflated by the flawed testing method.

### Objectives (final, after reviewer feedback)
1. Develop a class-level detector for God Class and Data Class using structural measurements such as size, complexity, and cohesion.
2. Develop a method-level detector for Long Method and Feature Envy, adding relationship features that capture how much a method depends on other classes' data.
3. Evaluate both detectors on Java projects never seen during training, comparing numerical metrics against learned code representations to identify the more effective approach.

*(Reviewer flagged that an earlier version's first two objectives were the same thing — "build a detector" and "train and test it." Fixed by splitting on granularity, with each level having its own angle.)*

### Paper title
**Repository-Level Evaluation of Metric-Based and Embedding-Based Code Smell Detection**

Alternatives: "Are We Testing on Unseen Code? Repository-Level Evaluation of Metric- and Embedding-Based Code Smell Detection" (echoes Di Nucci et al.'s "Are we there yet?"), or "Assessing Project Leakage and Annotator Agreement in Learned Code Smell Detection."

Don't put a number in the title — you don't have it yet and findings shift during review.

---

## 2. Novelty — what the paper actually claims

### Claim 1: Corrected evaluation protocol (the main one)

**Verified 2026-10-07 against primary sources. The claim holds.**

| Study | Split strategy | Evidence |
|---|---|---|
| **Madeyski & Lewowski 2023** (MLCQ authors' own follow-up) | Stratified 5-fold CV, 120 repetitions — **stratified by class label only** | Preprint: "Sampling during fold creation is stratified to ensure that each fold receives data from each class." No project grouping anywhere. |
| **Kovacevic et al. 2023** | Stratified random sampling, 80/20, repeated 51× | Reproducibility package states it directly |
| **EnseSmells 2025** (newest SOTA hybrid, all 4 smells) | Random 80/20 + stratified 5-fold | No project-aware stratification |

The highest-risk check was Madeyski & Lewowski 2023 — if the MLCQ authors themselves had used project-level validation, this claim collapsed. **They did not.** Their External Validity section only notes the data comes from "a single set of open source Java projects"; project independence, leakage, and cross-project generalization are not discussed anywhere in the paper.

Repeating a leaky split 51 or 120 times averages over 51 or 120 leaky splits — it produces tight confidence intervals around an inflated number, which reads as rigour.

**Essential scoping — do not overclaim.** Cross-project validation is *not* unknown in the wider code smell field. A 2024 survey of data preparation across 36 deep-learning code smell papers found 23 use within-project and 10 use cross-project evaluation. So the technique is established; it simply has not been applied to the MLCQ line of work.

The precise, defensible claim is therefore:
1. MLCQ-based studies consistently use row-level splits (three studies verified above),
2. cross-project evaluation is established elsewhere in the field,
3. **nobody has quantified the gap on MLCQ.** The same 2024 survey provides no performance comparison between within- and cross-project scenarios.

**Frame it as correction, not invention:**

> Cross-project validation is standard in defect prediction and used in roughly a third of deep-learning code smell studies. Despite this, MLCQ-based studies — including the dataset authors' own follow-up — evaluate with class-stratified splits that place samples from the same repository on both sides. We quantify the gap.

**Risk flag.** A public review of EnseSmells already states that "a natural stronger test is a project-level split of the MLCQ data, keeping all snippets from the same repository on one side." The idea is visible to reviewers in this area. It is not secret, and someone else may run it. This is an argument for moving quickly, not for abandoning the claim.

**Possible bonus finding.** ImprovMLCQ enriches MLCQ with features from CK, PMD, Organic and Designite *plus project characteristics*. If it also splits by row, including project identity as a feature would make leakage worse, not better — project membership becomes directly learnable. Worth verifying; it could be a sharp secondary result.

### Claim 2: Label reliability as a performance ceiling
No model can exceed the agreement of the humans who produced its labels. Report inter-rater agreement per smell and interpret performance against it.

This also gives a sanity check: an unexpectedly high F1 becomes evidence of leakage, not success.

### Claim 3: Representation comparison under the corrected protocol
Metrics vs embeddings vs fusion, four smells, two granularities.

**This is the vehicle, not the novelty** — Kovacevic et al. published metrics-vs-embeddings on MLCQ already.

### Claim 4: Documented negative result for Feature Envy
Supported by its inter-rater agreement rather than presented as a tuning failure.

### What is NOT novel (corrections made during planning)
- **Granularity is not a differentiator.** Kovacevic covered both: 2022 paper did Long Method + God Class, 2023 paper did Data Class + Feature Envy.
- **The AI-generated-code angle is occupied.** I earlier claimed it was unexplored without having searched — that was wrong. See §9 for the papers.
- **Hierarchical pooling is adjacent, not virgin.** Kovacevic computed per-line embeddings and pooled by sum/average. Method-level pooling is different but not a new idea.

### Due diligence status
1. ~~**Madeyski & Lewowski 2023 methodology**~~ — **DONE, 2026-10-07.** Class-stratified 5-fold CV, 120 repetitions, no project grouping. Claim 1 survives. See the table above.
2. **Zakeri-Nasrabadi et al. 2023** SLR on code smell datasets and validation mechanisms — still outstanding.
3. **ImprovMLCQ** — methodology not visible in the abstract; get the full paper. It enriches MLCQ with CK/PMD/Organic/Designite features plus project characteristics, so it is close to Phase A and may also be a bonus leakage finding.

---

## 3. The data

### MLCQ — the main dataset
- `MLCQCodeSmellSamples.csv`, **semicolon-delimited**
- 14,739 annotation rows → **4,770 distinct samples**
- 522 repositories, 26 reviewers
- **All 522 obtained** at their pinned commits (2026-10-13): 14 from their post-2019 GitHub address, 9 from the Software Heritage archive (same git hash). See PRD §4.3.
- Zenodo: https://zenodo.org/records/3666840 · CC-BY-4.0
- Cite: Madeyski & Lewowski, *MLCQ: Industry-relevant code smell data set*, EASE 2020, DOI 10.1145/3383219.3383264

**Critical property: MLCQ contains labels and addresses, no features.**

| Column group | Columns | What it is |
|---|---|---|
| Identifiers | `id`, `sample_id`, `reviewer_id` | Bookkeeping |
| Labels | `smell`, `severity`, `type` | The answer |
| Address | `code_name`, `repository`, `commit_hash`, `path`, `start_line`, `end_line`, `link` | Where the code lives |

Nothing describes the code. A row says: `org.apache.tez...UnorderedPartitionedKVWriter`, in `apache/tez` at commit `d5675c3`, lines 89–1427 → **critical Blob**. That's a verdict plus a street address.

The authors couldn't redistribute the source of 522 repos (licensing, size), so they shipped pointers. Every feature in this project is constructed by visiting the referenced repository at the pinned commit.

### Two paired annotation tasks — this is the version boundary

| Granularity | `type` | Smells | Samples | Positives (major+critical) |
|---|---|---|---|---|
| Class | `class` | Blob/God Class, Data Class | 2,340 | 286 / 311 |
| Method | `function` | Feature Envy, Long Method | 2,430 | 140 / 170 |

Blob and Data Class were annotated on the *same* samples (2,332 of 2,334 overlap). Feature Envy and Long Method likewise (2,415 of 2,415). Zero overlap across granularities.

So class-level gets two smells for the price of one pipeline. Method-level is a separate engineering effort because it needs method-level metrics and a signature join.

**Multi-label, not multi-class:** 63 class samples are both Blob and Data Class.

### Severity distribution
| Smell | critical | major | minor | none |
|---|---|---|---|---|
| blob | 127 | 312 | 535 | 3,045 |
| data class | 146 | 401 | 510 | 2,964 |
| feature envy | 24 | 142 | 288 | 2,883 |
| long method | 78 | 274 | 454 | 2,556 |

### Positive rates by binarization rule (max-severity aggregation)
| Smell | major+critical | minor+ |
|---|---|---|
| Blob | 12.3% | 21.7% |
| Data Class | 13.3% | 20.7% |
| Long Method | 7.0% | 14.0% |
| Feature Envy | 5.8% | 13.3% |

The `minor` decision roughly doubles the positive count. More consequential than the aggregation rule.

**Decision (D-1, 2026-10-13): `minor_plus` is the primary label rule** (Blob 506, Data Class 485, 221 both). Reason: reviewers agree more on *whether* a class smells than on *how badly* — Krippendorff's α is higher under `minor_plus` (table below) — and it gives about twice the positives. `major_critical_only` (286 / 311) is reported as a sensitivity analysis.

### Reviews per sample (Blob)
| Reviews | Samples | Share |
|---|---|---|
| 1 | 1,562 | 67% |
| 2 | 132 | 6% |
| 3 | 405 | 17% |
| 4 | 197 | 8% |
| 5 | 38 | 2% |

67% single-reviewed, so aggregation only affects a third of the data — decide it, don't agonize. Their reliability is also unmeasurable.

### Inter-rater agreement (measured, pairwise, binary at major+critical)
| Smell | Pairs | Raw agreement | Chance | Kappa | Reading |
|---|---|---|---|---|---|
| Long Method | 1,495 | 82.0% | 62.3% | **+0.52** | Moderate |
| Data Class | 2,903 | 74.2% | 65.4% | **+0.26** | Fair |
| Blob | 2,909 | 75.8% | 70.7% | **+0.17** | Slight |
| Feature Envy | 1,471 | 77.0% | 79.0% | **−0.10** | None — worse than chance |

Exact severity agreement (all four levels): Blob 46%, Data Class 45%, Long Method 52%, Feature Envy 33%. 67 Blob samples have maximum spread — one reviewer said `none`, another `critical`, same class.

**These are pairwise-averaged.** Recomputed properly as Krippendorff's α (handles 1–5 raters per sample; class level, 2,340 samples, 771 multi-reviewed — `phase_a_data/labels.py`):

| Smell | α binary, `minor_plus` (primary) | α binary, `major_critical_only` | α ordinal (4 levels) |
|---|---|---|---|
| Blob | **0.359** | 0.273 | 0.364 |
| Data Class | **0.431** | 0.339 | 0.451 |

α comes out higher than the pairwise kappa above (0.17 / 0.26), so quote α as the label-reliability ceiling and say which statistic it is. Method-level α is not computed yet (method level is parked).

Also: multi-reviewed samples may not be a random subset. If MLCQ deliberately double-reviewed ambiguous cases, agreement is biased downward. Check what the MLCQ paper reports.

**The pattern is objectivity, not granularity.** Long Method is countable. Data Class is checkable. Blob is judgment. Feature Envy requires holding a method *and* everything it touches in your head at once from a snippet in a review tool.

### Other datasets
| ID | Dataset | Role |
|---|---|---|
| D2 | `MLCQCodeSmellDevelopersSurvey.csv` | 20 reviewers' background; joins on `reviewer_id`. Has a "which code smells do you believe you can recognize" column — at least one reviewer rated Blob without claiming to recognize it |
| D3 | CrowdSmelling / Fontana, Zenodo 6555241 | Out-of-distribution test only, never trained on |
| D4 | 522 GitHub repos | Source for all feature extraction |

**Download list:** `MLCQCodeSmellSamples.csv` (required), `MLCQCodeSmellDevelopersSurvey.csv` (recommended, for annotator weighting), `MadeyskiLewowskiMLCQAppendix.pdf` (read it, documents annotation methodology). Skip both `.xlsx` files — redundant.

**Commit `datasets/` to git** — it's 7.5 MB, immutable (Zenodo DOI), CC-BY licensed, and reproducibility matters for a research project. Add `datasets/NOTICE.md` with the citation.

### Why not other datasets
Fontana's original data was deliberately balanced ~1:1 smelly/non-smelly, producing the famous >96% accuracy results. Di Nucci et al. replicated on realistic imbalanced data and found performance dropped up to 90%. MLCQ is imbalanced the way real repositories are.

---

## 4. Architecture

```
MLCQ labels + 522 Java repos
          ↓
   Clone at pinned commit
          ↓
   Single-pass extraction:
     • CK metrics
     • source snippets
     • ATFD / LAA / FDP
          ↓
   Labelled dataset
   (2,340 class · 2,430 method)
          ↓
   ┌──────┴──────┐
   ↓             ↓
Label analysis   REPOSITORY-LEVEL SPLIT  ← the contribution
(Fleiss' kappa)        ↓
                ┌──────┼──────┐
                ↓      ↓      ↓
         Rule baseline  Metrics arm  Embedding arm
         (Lanza &       (CK + ATFD,  (CodeBERT,
          Marinescu)     XGBoost)     frozen)
                       └──────┬──────┘
                              ↓
                           Fusion
                              ↓
              Calibrated detector + results table
              (isotonic · global threshold per smell
               · F1 · MCC · PR-AUC)
```

Validation kept separate: row-level split as contrast experiment only; CrowdSmelling inference-only.

### Why three arms
- **Rule Baseline** — no learning, fixed thresholds. The bar to beat. If trained models can't beat hardcoded numbers from 2006, there's no result.
- **Metrics Arm** — code as ~14 numbers you chose.
- **Embedding Arm** — code text to CodeBERT, which produces ~768 numbers it chose.
- **Fusion** — combine the two learned arms. The baseline doesn't feed into fusion; it's a reference line.

All three on the **identical split**, so any difference is real rather than an artifact of easier test data.

### Models: 4 pipelines, 8 classifiers
| Pipeline | Smells | Classifiers |
|---|---|---|
| Class-level, metrics | God Class, Data Class | 2 |
| Class-level, CodeBERT | God Class, Data Class | 2 |
| Method-level, metrics | Long Method, Feature Envy | 2 |
| Method-level, CodeBERT | Long Method, Feature Envy | 2 |

One classifier answers one yes/no question. Adding the second smell to a built pipeline is minutes of work.

---

## 5. Metrics vs representations

Both turn code into numbers, because a model can't read code. The difference is **who decides what the numbers mean**.

| | Metrics | CodeBERT embeddings |
|---|---|---|
| Where numbers come from | We count things | The model produces them |
| How many | ~14 | ~768 |
| Sees actual code? | No, only counts | Yes, the real text |
| Can explain a flag? | Yes — "too big, too tangled" | No |
| Handles very long classes? | Yes | Struggles (512-token cap) |
| Notices unanticipated things? | No | Yes — naming, style, patterns |

Analogy: metrics are "3 bedrooms, 1200 sq ft, built 1995." Embeddings are showing someone a photo.

### CK metrics — class level
WMC, LOC, NOM, NOF, CBO, RFC, LCOM, LCOM\*, TCC, LCC, DIT, NOC, fan-in, fan-out.

- *God Class* — high WMC, high LOC, high NOM, **high LCOM**. Big and incoherent.
- *Data Class* — **low** WMC, high NOF, mostly getters/setters.

They pull in opposite directions on complexity, which is why one classifier per smell.

### CK metrics — method level
LOC, WMC (cyclomatic), parametersQty, variablesQty, maxNestedBlocks, returnsQty, loopQty.

### Relational features — from JavaParser, not CK
| Metric | Meaning |
|---|---|
| **ATFD** | Access To Foreign Data — distinct foreign attributes/accessors touched |
| **LAA** | Locality of Attribute Accesses — own-class accesses ÷ total |
| **FDP** | Foreign Data Providers — distinct external classes reached into |

These exist for **Feature Envy**. That smell is a *relationship* between a method and another class. No scalar metric of the method in isolation can express it — CBO says a method is coupled, not whether it's more interested in others than itself.

### About CK
- **Chidamber & Kemerer metrics** (1994) is the metric suite; **CK** (`mauricioaniche/ck`) is the Java tool implementing it.
- Crucially: **it parses without compiling.** 522 arbitrary 2019-era repos cannot be built — dependencies are gone, build tools stale.
- LCOM has several incompatible definitions. CK emits `lcom` and `lcom*`. Note which you use; a reviewer will ask.
- **TCC/LCC need CK's "variables and fields" option on.** CK derives them from per-method field usage, which it only records with that flag; with it off, every class with 2+ methods gets TCC = 0. Caught in the pilot; the run uses the flag (`extraction.ck_variables_and_fields`).
- CK crashes outright on a few repos (JDT bugs on single files: SapMachine, openj9, n4js, ceylon, error-prone-javac, reactor-core). Those were measured per module/folder instead, so DIT / NOC / fan-in only see that unit; listed in `_scoped_repos.json`. Two folders still failed: SapMachine `javac/comp` (JDT NPE triggered by two files together; re-measured without `ArgumentAttr.java`, a non-sample file, PRD Step 2) and n4js's generated ANTLR parser (out of memory; method samples only, left `failed` until method level resumes).
- **javalang gaps:** it cannot parse `Type[]::new` or `Map<K, @Nullable V>` (both Java 8). `relational.py` retries with them blanked out; neither feeds a metric. Worth a sentence in the writeup.
- Needs JDK on PATH.

### The 512-token problem
CodeBERT caps at 512 tokens. A God Class runs 800–1,300 lines. **The very property that makes it a Blob prevents it fitting in the encoder.**

Two strategies, both implemented and compared:
- **Truncation** — first 512 tokens. Baseline, expected weak for Blob.
- **Hierarchical pooling** — embed each method, pool over methods.

Report token-truncation rate per smell as a paper artifact.

---

## 6. The 13 steps

### Phase A — Data

**Step 1 — Resolve samples to source.** 522 unique `(repository, commit_hash)` pairs. `git init` → `git remote add` → `git fetch --depth 1 origin <sha>` → `git checkout FETCH_HEAD`. Write `unresolved.json` with reasons. Resumable via progress manifest. **Pilot on 20 repos and measure** disk, wall-clock, failure rate before the full run.

**Step 2 — Single-pass extraction.** THE critical step. Per repo, before deleting:
1. CK metrics (class + method)
2. Source snippets — exact `start_line`–`end_line` per sample
3. Relational features (ATFD/LAA/FDP) via JavaParser

Then delete the clone. For repos over a size threshold, scope CK to the enclosing Maven/Gradle module and log which were scoped.

**Step 3 — Class-level join [GATE].** Exact join on `code_name` + `commit_hash`. Output `join_report.json` with match rate, unmatched breakdown, and **30 random matched rows dumped for manual verification**. Stop here.

**Step 4 — Method-level join [GATE].** FQ class + method signature + commit. Handle overloads, generics, varargs explicitly. Own report, own 30-row check.

**Step 5 — Labels and agreement.** Four outputs: primary labels (`minor_plus`, D-1); Krippendorff's α per smell (D-2); annotator-filtered label set (using D2); high-confidence subset (multi-reviewed samples where reviewers agreed).

**Step 6 — Splits.** `GroupShuffleSplit(groups=repository)`, 80/20. Repo lists committed. `StratifiedGroupKFold` (5-fold) for training-time use. Also generate a **row-level split labelled as contrast experiment only** — used exactly once, in Step 12.

### Phase B — Metrics arm

**Step 7 — Rule baseline.** Lanza & Marinescu detection strategies. **Decision (D-4, 2026-10-13): thresholds as implemented in PMD 7**, whose `GodClassRule` cites the book (p. 80). The book gives FEW / MANY as ranges (FEW = 2–5), so tools pick different values (PMD 3/5, an NDepend reimplementation 5/8); PMD's are reproducible and widely used.
- God Class: `ATFD > 5 AND WMC ≥ 47 AND TCC < 1/3` (all sources agree).
- Data Class: `WOC < 1/3 AND ((NOPA+NOAM > 3 AND WMC < 31) OR (NOPA+NOAM > 5 AND WMC < 47))`.
- WMC = summed cyclomatic complexity in both the book/PMD and CK. Definition differences to report: our NOPA excludes static fields (PMD counts them); our WOC divides by public methods + NOPA (Marinescu's definition; PMD divides by public methods only). All thresholds in `config.yaml`.

**Step 8 — Metrics models [GATE].** Logistic Regression (L2, balanced) → Random Forest (balanced_subsample) → **XGBoost** (`scale_pos_weight` from actual ratio) → optionally LightGBM. One binary classifier per smell. Feature Envy ablation: with and without relational features. Must beat Step 7.

### Phase C — Embedding arm

**Step 9 — Encode snippets.** Read cached snippets from Step 2 — no repo access needed. CodeBERT or UniXcoder, frozen, CPU, batched, cached, resumable. Both pooling strategies.

**Step 10 — Embedding models.** Frozen encoder, small head (logistic regression and shallow MLP). Don't fine-tune — 286 Blob and 140 Feature Envy positives make overfitting near-certain.

### Phase D — Fusion and results

**Step 11 — Fusion.** Early (concatenate; watch the 14-vs-768 dimensionality imbalance — scale metrics, PCA the embedding to ~50 dims, or both) and late (stacker trained on validation folds).

**Step 12 — Threshold, evaluation, OOD [GATE].** Freeze one global threshold per smell, selected on validation folds. Isotonic calibration. Produce the results table. Re-run under the row-level split and report the delta. Apply winning arm to CrowdSmelling unmodified.

**Step 13 — Paper.** Written from the results files.

### Gates
| Gate | Blocks |
|---|---|
| Step 3 join manually verified | All class-level work |
| Step 4 join manually verified | All method-level work |
| Step 8 beats Step 7 | Phase C |
| Step 12 complete | Step 13 |

---

## 7. Calibration and thresholds

**Calibration** makes the number honest. XGBoost's raw output isn't a real probability — it might say 0.9 for something genuinely smelly 60% of the time. Isotonic regression learns the mapping so 0.8 means roughly 8 in 10.

Matters because the tool shows confidence to a developer. Flag something at 90% and be wrong half the time, and people stop trusting it — the same trust problem that makes existing tools get switched off.

**Threshold** turns a score into a decision. 0.5 is wrong here — only ~12% of samples are smelly, so 0.5 is far too strict. Find it by maximizing F1 on validation folds; it usually lands well below 0.5.

**One global threshold per smell:** per smell because positive rates differ (12.3% vs 5.8%); global because it's applied uniformly across repos, not tuned per repository.

**The rule:** chosen on validation folds carved out of training projects, never on test. Picking on test means peeking at answers.

---

## 8. Expected results — calibrate before starting

- **Long Method will score high.** Raw line count alone separates it at **AUC 0.981** (positive median 58 LOC vs negative 8). Near-definitional. Report it, don't lead with it.
- **Feature Envy will score poorly.** Kappa −0.10, 140 positives. Line count alone gives AUC 0.713, which is a length confound not signal. Report as a documented limitation with the agreement evidence. Run 5+ seeds, mean ± std.
- **Blob (+0.17) and Data Class (+0.26)** are the real results, both label-bounded.
- **Metrics vs embeddings may show no clear winner** — published MLCQ work found exactly that. A null result is publishable under the corrected protocol.
- **An unexpectedly high F1 is evidence of leakage, not success.** Check the split before celebrating.

### Reference numbers from literature
- Madeyski & Lewowski 2023 MCC: Long Method 0.81, Data Class 0.57, Blob 0.51, Feature Envy 0.31.
- CuBERT embeddings on MLCQ: F-measure 0.53 God Class, 0.75 Long Method.

---

## 9. Related work

### Dataset
- **Madeyski & Lewowski (2020)** — *MLCQ: Industry-relevant code smell data set*, EASE. Required citation.
- **Madeyski & Lewowski (2023)** — *Detecting code smells using industry-relevant data*, IST 155:107112. 6,720 models. **Read the methodology section — highest-risk check on the split claim.**

### Metrics vs embeddings
- **Kovacevic et al. (2022)** — Long Method and God Class via neural source code embeddings.
- **Kovacevic et al. (2023)** — Data Class and Feature Envy, ScienceDirect. Found no clear winner.
- **EnseSmells (2025)** — hybrid of embeddings plus design metrics, all four smells.
- **ImprovMLCQ** — MLCQ enriched with CK, PMD, Organic, Designite features plus project characteristics.
- `milica-skipina/ML-code-smell-detection` — publishes CuBERT and CodeT5 embeddings for MLCQ Feature Envy and Data Class. Use to validate your extraction. **Do not use their splits** — row-level, which is what you're correcting.

### Methodology — strongest citations
- **Di Nucci et al. (2018)** — *Are we there yet?*, SANER. The precedent for the whole argument.
- **Pecorelli et al. (2020)** — data balancing, JSS 169:110693.
- **Zakeri-Nasrabadi et al. (2023)** — SLR on code smell datasets and validation mechanisms, ACM Computing Surveys.
- **Tuning Code Smell Prediction Models: A Replication Study** — ICPC 2024.
- **Santos & Choren (2025)** — data preprocessing SLR, IST 184:107752.
- **Code-Centric Detection of Vulnerability-Fixing Commits** — arXiv 2605.13138. Runs your exact experiment on a related task: group-stratified splits cut F1 by ~14–17% vs random, with t-SNE showing repository-specific clusters. Your methodological template — and evidence the effect is real.

### LLM-generated code smells (motivation, not contribution)
- **Paul, Zhu & Bayley (2025)** — *Investigating The Smells of LLM Generated Code*, arXiv 2510.03029. Measures smells in LLM-generated **Java** vs professionally written baselines.
- **Causal Analysis of LLM Code Smells** — arXiv 2511.15817. "Propensity Smelly Score."
- **Khati et al., ICSE 2026** — smell propensity in LLM-generated code.
- *Copilot-in-the-Loop* (ASE 2024), *Can LLMs Produce Better Object-Oriented Designs?* (2026), Moratis et al. on the DevGPT dataset.

### LLMs as detectors
- **Beyond Strict Rules: LLMs for Code Smell Detection** — arXiv 2601.09873.
- **Smelly-shot is all you need** — EMSE 2026.
- **Alomari et al. (2026)** — SLR on LLMs for code quality, IST 190:107960. LLM refactoring is not reliable.

### For the deferred Paper 2
- **Calibrating Deep Learning-based Code Smell Detection using Human Feedback** — SCAM 2023. Close to the memory layer.

---

## 10. Hardware and practical constraints

**Target machine:** Windows, Intel i7-1255U (12 threads), 16 GB RAM, Intel Iris Xe (**no CUDA**), ~317 GB free.

Everything runs on this machine. Encoding ~4,700 snippets through a 125M-parameter encoder on CPU is a one-time ~1–3 hour cost; cache the vectors. Colab only if fine-tuning (not recommended).

### Disk is the binding constraint
The 522 repos include SapMachine (a JDK fork), oracle/graal, apache/hadoop, apache/hive, apache/lucene-solr. **Median repo yields only 4 samples**, so most bytes downloaded are waste.

Top repos by sample count: SapMachine 138, apache/hive 111, eclipse/birt 78, oracle/graal 74, apache/camel 66, apache/hadoop 65.

### Why snippets are tiny
| | Size |
|---|---|
| 522 cloned repos | tens of GB (sparse checkout of `*.java` + build files kept it to ~11 MB per repo on average) |
| All snippets (4,692 extracted) | **~11 MB** (measured) |

335,902 lines total across every sample. You keep roughly 0.1% of each repo. Class snippets dominate — 298K of 336K lines (median 51 lines, max 4,111); method snippets are tiny (median 8 lines).

**Snippets are not regenerable without re-cloning. Back up `data/processed/snippets/`.**

### Non-negotiable: single-pass extraction
Clone → CK + snippets + relational → delete. Deleting before caching snippets means re-cloning 522 repos for the embedding arm. **The single most expensive available mistake in this project.**

---

## 11. Rules

### Modelling
- **No SMOTE or synthetic oversampling.** Interpolating in a code-metrics space invents classes that could not exist. Class weights only.
- **Same model family across all four smells.** Mixing families makes a performance gap unattributable to smell difficulty vs model choice.
- **Multi-label, not multi-class.** One binary classifier per smell.
- **One global threshold per smell**, on validation folds, frozen, applied uniformly. Never per-repository.
- **Calibrate** with `CalibratedClassifierCV(method='isotonic')`.
- **Never take the max of two models' scores per sample.** That implicitly uses the test label and inflates results in a way that won't survive review.
- Under ~200 positives: 5+ seeds, mean ± std.

### Evaluation
- Headline metrics: **F1, MCC, PR-AUC.** Never accuracy — predicting "clean" everywhere scores ~88%.
- PR-AUC over ROC-AUC — at 12% positives, ROC-AUC flatters everything.
- Splits by repository, committed in `eval/splits/`. Row-level split appears **only** as the Step 12 contrast experiment.
- Every arm compared against the rule baseline and the previous arm, same split.
- One results file per run, never overwritten.
- Report negative results.

### Out of scope
Graph representation, GNNs, LLM explanation layer, memory/adaptive-feedback layer, CI/CD and PR bots, reinforcement learning, temporal modeling, vulnerability scoring, data flow graphs, any language besides Java (filter to `*.java`, skip others silently), fine-tuning an encoder.

### Terminology (matters for the writeup)
Code *embeddings* are vectors from a pretrained encoder. Graph *representation* is a data structure. A *GNN* is a model that consumes one. Paper 1 uses embeddings and no graphs. Conflating these is what makes a reviewer suspicious — the original project brief labelled the whole pipeline "Graph Neural Networks" including parts that were just graph construction.

---

## 12. Tech stack

Python 3.11 · pandas · scikit-learn · XGBoost (+ optional LightGBM) · a kappa/alpha implementation · JavaParser or `javalang` · CK JAR (needs JDK) · HuggingFace `transformers` + PyTorch CPU · Git. Windows-compatible throughout.

---

## 13. Optional additions worth considering

**Nearly free (data already in hand)**
- Per-project performance breakdown across the 88 class-level test repos — pooled numbers hide wide variation.
- Which projects are hardest, correlated with size, age, domain.
- Feature importance as a finding: do the models rely on different metrics than Lanza & Marinescu's rules assume?
- Severity as an ordinal target rather than binary.
- Learning curve — does more data help, or are you at the label-noise ceiling?

**Cheap, strengthens the paper**
- **Bootstrap confidence intervals over repositories + paired tests between arms.** Most code-smell papers report a single F1 with no error bars.
- Agreement-stratified evaluation — performance on unanimous vs contested samples.
- Threshold sensitivity, and false positives translated into concrete review cost.

**Moderate effort**
- **Error analysis** — hand-inspect 30 false positives and 30 false negatives, categorise failure modes. The section reviewers most often find missing.
- Reproducibility package — the joined dataset, splits, and code.

**Not optional:** you're arguing prior work's evaluation was flawed. If yours has no confidence intervals and no error analysis, the first reviewer will say so, and fairly.

---

## 14. Paper 2 — reserved

Context and adaptation: repository graph representation (AST + call/access graph), retrieval API, GNN experiment, LLM explanation layer with a measured unsupported-claim rate, adaptive memory layer with a CI/CD surface for collecting developer feedback.

Not designed until Paper 1 reports — their shape depends on Paper 1's numbers. If relational features already close the Feature Envy gap, the graph layer's job is narrower; if calibration is poor, the explanation layer's grounding contract changes.

**Known dependency:** Paper 2 needs whole-repository graphs, which the snippet cache doesn't provide. Requires a second clone pass — acceptable, since by then only the repos that matter need visiting.
