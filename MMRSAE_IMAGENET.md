**MPSAE / CSR: selected ImageNet benchmark**

The main Docker experiment now benchmarks `csr`, `csrv2`, `mpsae` and
`mpsaev2`. The single-method Python default is `mpsae`; templates and the
launcher select all four methods. Existing commands, Docker service, scope
`mmrsae`, `MMRSAE_*` environment names, output/data paths, probes and retrieval
remain the same. Rebuild the image to include changed source:

```bash
docker compose up -d --build
docker compose logs -f --tail 100 imagenet
```

For a plan that starts no downloads, GPU work or fitting:

```bash
python mmrsae_imagenet.py --plan --variant mpsae
python mmrsae_imagenet.py --plan --variant mpsaev2
ENV_FILE=mmrsae.env.example bash run_mmrsae_imagenet.sh --plan
```

This is an exploratory ImageNet comparison. One target-K model is fitted per
method/backbone and evaluated at multiple inference budgets. The strict historical
CSR/MPSAE studies and transfer recipes remain separate. DLMMSAE and landmark-first
MMRSAE remain library methods; the active CLI and reports exclude them, and saved
DLMMSAE results are not relabeled or modified.

**Shared plain SAE and MPSAE objective.** All selected methods use:

```text
a = (x - b_pre) D.T + b_enc
z_k = TopK_then_ReLU(a)
x_hat_k = z_k D + b_pre
R_K = MSE(x_hat_K, x) + MSE(x_hat_4K, x)/8 + dead_auxiliary_loss/32
```

The dictionary is the learned SAE matrix D, not a bank of input landmarks.
There is no context synthesis, activation masking, sparse KL or context KL in
these selected models. Code values are nonnegative, with at most k positives.
The original dead-neuron objective, its budget 512, activity updates and threshold
30 are unchanged. Auxiliary dead-neuron reconstruction and wide reconstruction
are separate branches.

| Method | Objective | Main/reconstruction sparsity | OT views |
|---|---|---|---|
| `csr` | `R_K + gamma*NCL` | Fixed K | None |
| `csrv2` | `R_K(t) + gamma*NCL` | Existing half-cosine main-K schedule | None |
| `mpsae` | `R_K + 2*L_OT` | Fixed K and wide 4K | K,2K,4K |
| `mpsaev2` | `R_K* + 2*L_OT` | Fixed K* and wide 4K* throughout | K*,2k_a(t),4k_a(t) |

CSR NCL uses two deterministic copies of normalized K codes, temperature 0.2,
and gamma 0.1 for RN50 or 1.0 for RN18. MPSAE uses the existing angular partial
matching gap, with default weight 2, transported mass .9, entropy .2, OT groups
of at most 32, initial/refinement iteration bounds 100/2000 and tolerance 1e-4.
The solver is detached; the cost receives the matching-gap gradient. The 2K view
has no additional reconstruction loss.

**MPSAEv2 implements the supplied fixed-main, linear auxiliary schedule.** Let
K* be the target inference sparsity. At every step:

$$
L_t = R_{K_*} + 2 L_{OT}(z_{K_*}, z_{2k_a(t)}, z_{4k_a(t)}).
$$

All views share one encoder preactivation. Main reconstruction always uses K*,
wide reconstruction always uses 4K*, and dead-neuron loss/activity stay fixed to
the original target-K MPSAE recipe. Only the two auxiliary OT views anneal.
With zero-based update t and total update count T, define
`T_a=max(1,ceil(anneal_fraction*T))` and:

$$
k_a(t) =
\begin{cases}
\operatorname{round}\left(K_0-(K_0-K_*)t/T_a\right), & t<T_a,\\
K_*, & t\ge T_a.
\end{cases}
$$

The rounding is Python `round`; budgets are capped by SAE width when encoding.
`MMRSAE_ANNEAL_START_K` sets K0 (default `min(h,max(64,4K*))` in Python; template
sets 64). `MMRSAE_ANNEAL_FRACTION` defaults to .7. At T=500, K*=8, K0=64, Ta=350:

| Step t | Main K* | k_a(t) | Second OT budget | Third OT budget | Wide reconstruction |
|---:|---:|---:|---:|---:|---:|
| 0 | 8 | 64 | 128 | 256 | 32 |
| 175 | 8 | 36 | 72 | 144 | 32 |
| 350 onward | 8 | 8 | 16 | 32 | 32 |

At ka=K*, the loss and gradients match fixed-K MPSAE for identical model state.
CSRv2 retains its existing cosine schedule; MPSAEv2 does not anneal the main gate.
When fraction=1, the mathematical endpoint is t=T (just after the final zero-based
training update); when T=1, its sole update uses K0. `mpsae-annealed` aliases
`mpsaev2`. Historical MPSAEv2 implementations in other runners retain their original
recipes and must not be conflated with this fixed-main version.

Training logs keep `training_k=K*`, report `annealing_k`, `ot_view_k`,
`ot_view_2k`, `ot_view_4k` and fixed `auxiliary_budget`. Diagnostics measure the
actual auxiliary OT views, and plots distinguish main K, ka, OT views and wide
reconstruction. New protocol `mmrsae-imagenet-mpsae-fixed-main-v5` plus source
identity prevents resuming DLMMSAE or earlier incompatible v2 checkpoints as this
recipe. Data allocation retains the previous held-out anchor reservation for
identical split settings across methods; plain models do not load a synthesis or
context bank for serving. Changing the benchmark does not establish an accuracy
or compute improvement.

**All-zero codes.** The final Top-K/ReLU can produce fewer than K positive
coefficients or an empty code. A zero final code receives no gradient through
that gate. Identical nonzero views have zero diagonal angular cost, but this
does not make the full OT matching gap zero; three all-zero views have angular
cost 2/3 under the implemented normalization. As in
the original MPSAE runner, `--allow-empty-codes` (default; `MMRSAE_ALLOW_EMPTY_CODES=1`)
does not stop on such codes and does not change any objective: they keep a zero
direction in the OT cost and stay zero rows in the
retrieval index, and an empty query is recorded as a miss. Training logs
`final_nnz_min`, `final_nnz_below_k_fraction`, `final_zero_code_fraction`,
`raw_kth_preactivation_nonpositive_fraction`, `tail_ratio_2k_mean`,
`tail_ratio_4k_mean` and `encoder_bias_mean`; retrieval reports
`empty_gallery_codes`, `empty_query_codes` and `empty_query_fraction`.
`--no-allow-empty-codes` restores the strict abort. The separate repair profile
keeps its strict check.

**Reports and figures.** After every fitting invocation (one method or
`MMRSAE_VARIANTS`, dev or official, complete or failed),
[mmrsae_reporting.py](mmrsae_reporting.py) writes
`<output>/reports/<evaluation>_k<K>_<id>/`: `results.csv`, `results.md`, and PNG
and PDF figures. The overview contains:

| Figure | Measurement |
|---|---|
| `top1_by_k` | 1-NN accuracy vs every measured inference K, one panel per backbone |
| `top1_by_k_backbones` | RN18/RN50 curves and full-representation baselines, one panel per method |
| `classification_top1_by_k` | Independently trained linear-probe Top-1 classification vs K |
| `accuracy_by_active_dim` | Classification and 1-NN side by side, against measured mean positive coordinates (queries/gallery respectively) |
| `top1_vs_retrieval_ms` | 1-NN accuracy vs warm GPU search time per query batch; K labels at markers; excludes encoding |
| `top1_vs_gpu_hours_<backbones>` | 1-NN vs measured representation-training GPU-hours at each K; combined panels use macro accuracy and summed hours over the same backbones |
| `top1_by_method` | Endpoint accuracy at the configured training target K |
| `training_loss`, `sparsity_schedule`, `support_health` | Reconstruction/OT/NCL losses, fixed main K, auxiliary OT budgets and sparse-code health |

Full-representation retrieval references require `MMRSAE_DENSE_REFERENCE=1`
(`--dense-reference`) and enough GPU memory for the dense gallery. Full-parent
classification is always measured when linear probes are enabled and does not
need that index. Missing classification, timing or GPU-hour measurements in
older outputs are omitted; neither paper values nor zeros fill those gaps.
`results.csv` exports per-K numbers, `compute_tradeoff.csv` exports compute
comparisons, and `plot_coverage.json` plus `overview.md` identify available plots
and missing measurements. A synthetic plot-layout test is not an experimental
accuracy result.

Trials are grouped by settings except the method, implementation identity and
GPU hardware/software. Timing curves additionally separate different query
batches and gallery/query identities. This keeps target K, data sizes,
evaluation pools and code versions distinct.
`reports/index.json` lists the groups. `--plan`, `--preflight-only` and
`--prepare-only` produce no report; `MMRSAE_REPORT=0` disables it. Regenerate at
any time without a GPU: `python mmrsae_reporting.py --output-dir runs/mmrsae`.

All enabled MMRSAE/MPSAE transport objectives use angular cost only.
`MMRSAE_MIXTURE_WEIGHT=0` is enforced; nonzero overrides fail before data work.
Landmark teacher-to-student KL stays separate from OT. CSR/CSRv2 and
`mmrsae-rel` do not use transport. The staged repair profile also uses angular
cost under its new `mpsae-repair-angular-v2` protocol; its historical O/N labels
now select identical loss components.

**Accuracy at every inference K.** `MMRSAE_EVAL_TOPK=auto` evaluates
2,4,8,16,32,64,128,256,512 for ResNet-18, and adds 1024,2048 for ResNet-50,
capped by dictionary width. The training target K is always included. A custom
list such as `MMRSAE_EVAL_TOPK=2,4,8,16,32,64,128,256,512` is accepted, with
every value required to be between 1 and h. Each point re-encodes both gallery
and queries at that budget, using the plain SAE for all selected methods.

These curves evaluate one checkpoint per method/backbone; they do not train
independent models for each K. `top1_by_k.png/.pdf` plots inference K against
top-1 with a distinct curve for every method and a panel per backbone. Tables
include both training K and inference K. Unmeasured budgets are never filled
with copied accuracy. Completed points survive an interrupted evaluation.
Intermediate epoch evaluations still measure only the training target K.

CSRv2 retains the half-cosine **main-K** schedule from 64 to the target over
70% of updates by default. Its reconstruction budget follows 4K(t). MPSAEv2
instead uses the **linear auxiliary-OT** schedule specified above; its main K*
and wide reconstruction 4K* remain fixed. Both use a fixed compressor learning
rate. Inference uses the requested evaluation budget, independent of training
schedule. The plain encoder needs no input landmark dictionary at inference.
The dead-feature auxiliary keeps weight 1/32 and budget k_aux=512. The existing
wide reconstruction keeps weight 1/8; no separate 2K MSE is added.

`cost0_min/max` and the legacy-named `costmix_min/max` now both report the
angular cost range. Solver feasibility/refinement and negative-gap checks stay
enabled. Neighborhood affinity kernels are retained for historical mathematical
reference checks; active training objectives cannot enable them with an override.

**Launching and setting parameters.** For Docker, copy
[mmrsae.env.example](mmrsae.env.example) to `.env`, set the Hugging Face token
and optional W&B credentials, then use the existing entry point:

```bash
bash run_all_experiments_docker.sh
```

`run_full_pipeline.sh` is an alias. Both default to `EXPERIMENT_SCOPE=mmrsae`.
An existing `.env` supplies defaults; exported variables override them. Select
`ENV_FILE=mmrsae.env` to use a separate configuration file. `imagenet` selects the historical
strict runner; **`mmrsae` selects this new ImageNet-only workflow**. The
`all` and `transfer` scopes continue to be opt-in legacy workflows.
Adding `--plan` to the Docker launcher prints the configuration on the host
without building a container or requiring data credentials.

The Docker image installs the declared dependencies and includes the CUDA
compiler needed by the sparse retrieval extension. Data, parent weights and
compiled extensions persist in the named data volume; reports/checkpoints
persist under `runs/mmrsae/` on the host. Paths configured as `/data/...` or
`/output/...` refer to the container. Local files on the host require a mount
or placement in the persistent data volume before they can be used there.

In an already prepared environment, the direct launcher consumes exported
variables or an explicit `ENV_FILE`, and accepts ordinary CLI overrides:

```bash
bash run_mmrsae_imagenet.sh --plan
ENV_FILE=mmrsae.local.env bash run_mmrsae_imagenet.sh --plan
bash run_mmrsae_imagenet.sh --preflight-only
bash run_mmrsae_imagenet.sh --prepare-only
bash run_mmrsae_imagenet.sh --backbones resnet18 --variant mpsaev2 \
  --k 16 --lr 4e-5 --max-steps 500 --run-name rn18_mpsaev2_k16
```

These are usage examples, not commands executed during implementation. The
direct shell launcher reads only an explicitly selected `ENV_FILE`; Docker
loads `.env` by default. Use local paths in a direct-launch configuration.
Explicit CLI arguments override exported values, which override file defaults. `--plan` prints
configuration without GPU initialization or data preparation; the other modes
require the CUDA retrieval check. No CPU execution mode is provided.

| Manual setting | Default | Interpretation |
|---|---|---|
| `MMRSAE_BACKBONES` / `--backbones` | `resnet50,resnet18` | Comma-separated selected architectures |
| `MMRSAE_VARIANT` / `--variant` | `mpsae` | Single selected method; CLI override runs exactly once |
| `MMRSAE_VARIANTS` | `csr,csrv2,mpsae,mpsaev2` | Launcher comparison list, run sequentially |
| `MMRSAE_K` / `--k` | 8 | Training target K; included in the inference evaluation grid |
| `MMRSAE_EVAL_TOPK` / `--eval-topk` | auto | Inference budget grid; each point uses the same trained checkpoint |
| `MMRSAE_ANNEAL_START_K` / `--anneal-start-k` | min(h,max(64,4K)) | CSRv2 initial main K / MPSAEv2 initial auxiliary ka; templates select 64 |
| `MMRSAE_ANNEAL_FRACTION` / `--anneal-fraction` | 0.7 | Fraction of updates for annealing: CSRv2 cosine main K, MPSAEv2 linear auxiliary OT K |
| `MMRSAE_HIDDEN_DIM` / `--hidden-dim` | 8192 | Dictionary width |
| `MMRSAE_REPAIR_RANK` / `--dictionary-rank` (alias `--repair-rank`) | 32 | Historical library query-adapter rank; inactive for selected methods |
| `MMRSAE_DICTIONARY_TEMPERATURE` / `--dictionary-temperature` | 0.1 | Softmax temperature for input landmark lookup |
| `MMRSAE_DICTIONARY_BLEND` / `--dictionary-blend` | 0.1 | Initial trainable mixture of landmark context into x; strictly between 0 and 1 |
| `MMRSAE_CONTEXT_WEIGHT` / `--context-weight` | 0 for selected methods | Historical library context KL control; inactive here |
| `MMRSAE_CONTEXT_REFERENCE_COUNT` / `--context-reference-count` | 1024 | Fixed fitting reference rows Q, disjoint from A; at least 3 |
| `MMRSAE_CONTEXT_TEACHER_TEMPERATURE` / `--context-teacher-temperature` | 0.1 | Clean-input cosine profile temperature |
| `MMRSAE_CONTEXT_STUDENT_TEMPERATURE` / `--context-student-temperature` | 0.1 | Raw-context cosine profile temperature |
| `MMRSAE_CONTEXT_NOISE_STD` / `--context-noise-std` | 0.02 | Auxiliary lookup Gaussian noise relative to row RMS; 0 disables |
| `MMRSAE_LINEAR_PROBE` / `--linear-probe`, `--no-linear-probe` | 1 / enabled | Separate supervised classification evaluation at every K and full-parent features |
| `MMRSAE_PROBE_EPOCHS` / `--probe-epochs` | 10 | Epochs for each frozen-representation classifier |
| `MMRSAE_PROBE_BATCH_SIZE` / `--probe-batch-size` | 1024 | Classifier fitting batch |
| `MMRSAE_PROBE_LR` / `--probe-lr` | 0.1 | Classifier learning rate |
| `MMRSAE_PROBE_WEIGHT_DECAY` / `--probe-weight-decay` | 0.0001 | Classifier weight decay |
| `MMRSAE_MAX_STEPS` / `--max-steps` | 500 | Per-backbone update cap; zero removes the cap |
| `MMRSAE_EPOCHS` / `--epochs` | 10 | Epoch limit, also enforced when a step cap exists |
| `MMRSAE_BATCH_SIZE` / `--batch-size` | 1024 | Compressor fitting batch |
| `MMRSAE_TRAIN_SIZE` / `--train-size` | 128000 | Fitting/gallery rows; zero selects every eligible row after reserving anchors and dev queries |
| `MMRSAE_DEV_SIZE` / `--dev-size` | 10000 | Training-derived development queries |
| `MMRSAE_ANCHOR_COUNT` / `--anchor-count` | 1024 | Reserved training rows retained for split compatibility; unused by plain models |
| `MMRSAE_BANK_REFRESH` / `--bank-refresh` | 25 | Historical library bank interval; no bank is refreshed by selected methods |
| `MMRSAE_SEED` / `--seed` | 1227 | Model initialization, masking and fitting |
| `MMRSAE_DATA_SEED` / `--data-seed` | 42 | Stable allocation/extraction across model-seed changes |
| `MMRSAE_EVAL_EVERY` / `--eval-every` | 0 | Endpoint only; a positive completed-epoch interval adds gallery evaluations |
| `MMRSAE_CHECKPOINT_EVERY` / `--checkpoint-every` | 25 | Updates between resumable checkpoints |
| `MMRSAE_EMBEDDING_SOURCE` / `--embedding-source` | `precomputed` | Local/Hub NPZ+JSON pairs; `legacy` enables the older preparation workflow |
| `MMRSAE_EMBEDDINGS_REPO` / `--embeddings-repo` | `quangnguyenopt/NSFW` | Prepared embedding repository |
| `MMRSAE_EMBEDDINGS_REPO_TYPE` / `--embeddings-repo-type` | `model` | Hub repository type, independent of ImageNet's dataset repository |
| `MMRSAE_EMBEDDINGS_REVISION` / `--embeddings-revision` | `main` | Resolved once to an immutable commit in the cache |

One method has a ceiling of 1,000 fitting updates across two backbones; the
four-method templates therefore allow 8 representation fits and 4,000 fitting updates. This count
excludes linear classifiers, data preparation, encoding, search and initialization. Smaller steps
or one backbone reduce fitting cost; `--no-timed-retrieval` avoids repeated
timed searches while retaining endpoint accuracy. Full-parent dense reference
search is an explicit `--dense-reference` option because its gallery can be
large. It is not required for every manual trial.

**Data preparation and split boundaries.** Both backbones use prepared
archives from [quangnguyenopt/NSFW](https://huggingface.co/quangnguyenopt/NSFW/tree/main).
This is a private Hub **model** repository. Set `HF_TOKEN` to a token with
read access to it; accepting ImageNet terms alone does not grant that access.
Existing local pairs take precedence and require no network or token:

```text
<MMRSAE_EMBEDDINGS>/
  1K_train_ff2048.npz   1K_train_ff2048.json
  1K_train_rn18.npz     1K_train_rn18.json
  1K_val_ff2048.npz     1K_val_ff2048.json
  1K_val_rn18.npz       1K_val_rn18.json
```

The loader downloads only missing files for the requested backbone/split,
directly into this folder. It never downloads `weights.zip`, raw images,
parent checkpoints or an entire repository snapshot. `dev` uses train only;
`official` also opens val. Downloads use the Hub client's resumable cache.
Revisions are pinned once, and the archive LFS hash must agree with its JSON
before an archive download begins. Original receipts retain the FF2048
publisher/parent identity or RN18's parent, transform and image provenance;
the mirror is not relabelled as a new pretrained encoder.

The RN18 files use uncompressed NPZ members, which are mapped directly in
place, including ZIP64 and Fortran/byte-order handling. FF2048 files are
compressed: their feature member is streamed in 8 MiB chunks once into a
read-only NPY disk map. Repeated methods reuse it. Archive checksums are cached
against the absolute path, size and nanosecond `mtime`/`ctime` values;
a changed file or a different expected hash requires checking again or fails.
Feature validation uses batches bounded to about 8 MiB, and training converts
only the consumed batches to FP32. The feature matrix is never loaded whole
into RAM. The legacy FF2048 comparison now uses this bounded-memory reader too.

All four archives total about 8.1 GB. Compressed FF2048 train needs another
10.5 GB of cache disk when first opened (validation adds about 0.41 GB).
RN18 needs no second feature copy. Data/cache paths remain persistent under
Docker's `/data`; keep the cache when resuming or comparing methods.

For an offline server, copy the needed NPZ **and** JSON files to
`MMRSAE_EMBEDDINGS` and set `MMRSAE_DOWNLOAD=0`. To reuse this checkout's
existing `imagenet_embedding_weight` directory with Compose, set in `.env`:

```dotenv
DATA_HOST_DIR=/absolute/server/path/to/imagenet_embedding_weight
MMRSAE_EMBEDDINGS=/data
MMRSAE_DOWNLOAD=0
```

The directory must permit cache writes. For direct local execution, set
`MMRSAE_EMBEDDINGS` to that local directory instead. Explicit
`--ff2048-train/val` and `--rn18-train/val` paths still take precedence.

`MMRSAE_EMBEDDING_SOURCE=legacy` retains the original FF2048 publisher download
and automatic RN18 export from raw images. Missing default RN18 pairs are
exported **after** GPU retrieval preflight, using the resolved CLI/environment
settings. Existing archives still skip extraction. The standalone
`export_rn18_embeddings.py` remains available, including `--data-backend
imagefolder`; raw ImageNet permissions are needed only for those image paths.

Default `--evaluation dev` uses disjoint training-derived queries and does
not download/open official validation. Labels allocate/evaluate the pools;
compressor reconstruction/OT/NCL objectives do not use class labels. The separate linear-probe evaluation does use fitting labels. Landmarks
come from training and are reserved outside fitting/gallery even with unlimited
training size. The sparse KL implementation still removes the minibatch's union
of row IDs from its profiles if used outside this experiment recipe. The selected plain-SAE methods do not construct Q or synthesize from A.

Use `--evaluation official` explicitly for an official-validation measurement
after choosing settings. Source/parent/recipe fingerprints remain in the
results. Tuning on those results must be described as validation-informed
selection. A subset-gallery score is not the full-gallery published benchmark;
the 71.19% FF2048 reference applies only to its original full-gallery setting.
No published MRL values are treated as local measurements in this workflow.

`--no-download` requires compatible cached resources or explicitly supplied
local resources with the required provenance. Optional `--ff2048-train` and
`--ff2048-val` paths must refer to the intended released archives; filename
and shape alone do not authenticate a release. See the environment template
for revision/hash overrides. Changing a data or extraction recipe uses a
different cache identity rather than silently reusing incompatible features.

**GPU retrieval and cost reporting.** Before dataset work, the runner checks
the selected CUDA device, GPU FAISS and the custom sparse CUDA extension. It
performs small actual searches to check their operation; compilation or
backend failures raise immediately. CPU-only FAISS and unavailable CUDA are
errors. The full sparse index is also subject to a device-memory budget and
fails if it cannot fit. Runtime CUDA or search failures propagate; there is
no silent CPU fallback. These checks establish backend availability at run
time, not a guaranteed throughput on every GPU.

Retrieval is exact normalized 1-NN. Sparse indices retain coordinates and
FP32 values without expanding the full gallery to width 8192. Chunked scores
limit working memory while all gallery postings remain GPU-resident. Default
query batch is 32, score budget 256 MiB, gallery chunk ceiling 65,536 rows and
index memory fraction 0.8. CUDA-event and synchronized-wall search times are
reported separately from SAE encoding and index construction. Timed search
uses three warmups and five repetitions; reported time per query is amortized
over the configured batch, not isolated single-query latency. A padded K=8
indexed FP32 code uses 68 bytes before file metadata.

Representation `training_seconds` measures synchronized wall time around batch
transfer, forward/loss, backward, optimizer updates and training KL-bank refresh.
`training_gpu_hours = training_seconds / 3600` because each fit uses one GPU.
Diagnostics, data preparation, model/reference initialization, logging,
checkpoint I/O, retrieval and linear probes are excluded. The checkpoint retains
this counter across resume. The broader `fit_seconds` includes diagnostics and
other fitting-session work and is never substituted for training GPU-hours.
Hardware, timing scope and evaluation workload must match for speed comparisons.

**Separate classification evaluation.** Linear probes are enabled by default
after endpoint retrieval. Each inference K receives its own supervised linear
classifier trained on frozen sparse codes from fitting rows. Full-parent
features receive a separate classifier under the same declared settings. The
SAE and backbone remain frozen; validation/development labels only score the
finished classifiers, with no accuracy-based checkpoint selection. This is a
locally specified linear-probe protocol, not the backbone's original classifier
or a reproduction of classification numbers in the example figures.
Each classifier uses zero initialization, row-L2-normalized inputs, SGD with
momentum 0.9 and a cosine learning-rate schedule for the declared epoch count.
Sparse codes are cached as coordinates/values and expanded only for the current
minibatch. Classifier checkpoints are saved each epoch. The full-parent probe
streams features without a dense FAISS index and runs even when the optional
full-parent retrieval measurement is disabled.
`MMRSAE_LINEAR_PROBE=0` or `--no-linear-probe` skips it. Classification metrics
live under `summary.json:classification` and remain distinct from retrieval
`top1`, which means 1-NN label accuracy. Classifier fitting and encoding costs
are separate from representation training GPU-hours.

**Monitoring, resume and outputs.** Set `WANDB_API_KEY`, optionally
`WANDB_ENTITY`, and `WANDB_PROJECT=MMRSAE` for online monitoring. With no key,
the launcher defaults to offline W&B logging; `WANDB_MODE=disabled` disables
it explicitly. Each backbone has its own run, grouped by the experiment.
Loss components, training progress and endpoint retrieval metrics are logged;
local artifacts remain available independently of the dashboard. Configured
online monitoring failures are surfaced rather than silently disabling it.

Resolved configuration, provenance, split identities, fitting history,
checkpoints, retrieval metrics and per-query predictions are kept per run and
backbone. A compatible restart restores model/optimizer, mask RNG, data order,
landmark refresh state, training-time counter and update position. Selected plain-SAE methods need no Q bank or context RNG. Diagnostics do not mutate
activity counters. The shared checkpoint format retains inactive compatibility fields.
Reuse the same command/output
directory to resume. A different recipe gets a different derived name unless
one was specified; incompatible state under an explicit name is rejected.
Only one writer should own a run directory.

Each evaluated budget writes `retrieval/endpoint/k_<K>/result.json`, metrics
and predictions. `retrieval/endpoint/result.json` and the backbone summary
contain `per_topk`, `evaluation_topk` and `budget_evaluation=single_checkpoint_multi_k`.
Top-level retrieval fields remain the training-target-K point for older readers.
This revision uses `mmrsae-imagenet-mpsae-fixed-main-v5`; changed architecture,
code and recipe identities prevent resuming a sparse-start or auxiliary-only
annealing checkpoint as this method. Existing runs must keep their original
implementation and identities; new experiments receive new derived run names.

Source-level savings include bounded preparation/extraction, reusable parent
features, indexed sparse reductions, batched independent OT solves, shared pure Top-K gates and endpoint-only evaluation by default. These are design
choices visible in the code; no speedup or accuracy result is claimed before
execution. [VALIDATION.md](VALIDATION.md) records that limitation.
