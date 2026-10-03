*AI-generated draft (Claude, Anthropic) — for review. All numbers are computed by `scripts/gate_scene_classifier.py` from version-controlled labels and the trained model.*

# Scene-classifier gate — blind sort: wednesday_validation

**Verdict: PASS — strong agreement** (Cohen's kappa = 0.709).

- Paired recordings scored: **96**
- Raw agreement: **85.4%** (Wilson 95% CI 77.0%–91.1%)
- Cohen's kappa: **0.709**
- Human called Scene-1 on 41; model called Scene-1 on 49
- Disagreements: **14**
    - `CAMHDA301-20210922T091500`: human=not_scene1 model=scene1 (p=0.917)
    - `CAMHDA301-20220112T151500`: human=scene1 model=not_scene1 (p=0.5606)
    - `CAMHDA301-20220202T121500`: human=not_scene1 model=scene1 (p=0.9422)
    - `CAMHDA301-20220209T061500`: human=not_scene1 model=scene1 (p=0.9445)
    - `CAMHDA301-20220406T181500`: human=not_scene1 model=scene1 (p=0.9864)
    - `CAMHDA301-20221102T061500`: human=not_scene1 model=scene1 (p=0.9622)
    - `CAMHDA301-20230301T121500`: human=not_scene1 model=scene1 (p=0.9842)
    - `CAMHDA301-20230329T061500`: human=not_scene1 model=scene1 (p=0.9927)
    - `CAMHDA301-20230426T001500`: human=not_scene1 model=scene1 (p=0.9997)
    - `CAMHDA301-20231220T031500`: human=not_scene1 model=scene1 (p=0.9444)
    - `CAMHDA301-20240424T151500`: human=not_scene1 model=scene1 (p=0.9934)
    - `CAMHDA301-20240515T031500`: human=not_scene1 model=scene1 (p=0.997)
    - `CAMHDA301-20240619T121500`: human=scene1 model=not_scene1 (p=0.4229)
    - `CAMHDA301-20240717T031500`: human=scene1 model=not_scene1 (p=0.895)

- Timing (recordings both call Scene-1, n=38): **33/38** within one 30 s tile of the human's marked time.

Caveats (Defensible Statistics): kappa is sensitive to the Scene-1/not base rate; the agreement CI reflects n. The human sort is the reference, so model-Scene-1/human-not disagreements may be genuine model errors OR borderline cases where the human was stricter — eyeball the high-confidence ones. Tile-level blurry recall is weaker than clear (see metrics.json).
