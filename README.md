# CommCDSR (anonymous artifact)

## Setup
- Python 3.10+, PyTorch, scikit-learn; `pip install -r requirements-free.txt` is not needed beyond torch/pyaml/requests.
- LLM access (community profiles + rerank): export `LLM4CDSR_API_KEY=<key>`
  (an OpenAI-compatible endpoint; base_url/models configured in
  `configs/methods/comm_cdsr_best.yaml`).

## Run
Datasets are not bundled (Amazon review data; the paper's six 2-domain pair
configs and prepared canonical artifacts are added separately):

```bash
python -c "from cdsr.quick_start.quick_start_best import run_cdsr; ..."
```
with `config_file_list = [configs/overall.yaml, <dataset>.yaml, configs/methods/comm_cdsr_best.yaml]`,
`run_mode="train_test"`, and `community_memory` / `community_rerank` settings
per the paper (tau per target-domain tail severity; rerank theta_H=0.5, K=5).

## Layout
- `cdsr/quick_start/quick_start_best.py` — entry (`run_cdsr`)
- `cdsr/data`, `cdsr/protocol`, `cdsr/evaluation` — frozen data/protocol/eval
- `cdsr/validation` — post-training quality audits (optional `comm_validation.enabled`)
