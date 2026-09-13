# Open Model Evaluation Results

Post-competition evaluation of the purple agent on open LLM models deployed via Azure AI Foundry.
All runs use the same purple agent codebase (feature/nemotron-edge branch) with the deterministic
question-asking heuristic. Green agent is gpt-4o-mini on Azure OpenAI in all cases.

## Kimi K2.5 (n=6)

Endpoint: agentwhettersmodeltests.services.ai.azure.com/models
Rate limits: 20 RPM, 20k TPM
Date: 2026-05-28

| Run | Transcript | Score | Correct | Accuracy |
|-----|-----------|-------|---------|----------|
| 1 | 20260528_110503 | +600 | 134/160 | 83.7% |
| 2 | 20260528_120030 | +740 | 141/160 | 88.1% |
| 3 | 20260528_125304 | +560 | 132/160 | 82.5% |
| 4 | 20260528_141400 | +580 | 133/160 | 83.1% |
| 5 | 20260528_150716 | +620 | 135/160 | 84.3% |
| 6 | 20260528_174702 | +640 | 136/160 | 85.0% |

Mean score: +623 (std dev 58)
Mean accuracy: 84.5% (135.2/160)
Range: +560 to +740

Transcripts located at: pragmatic_builder/logs/transcripts/<timestamp>/

## DeepSeek V3.2 (n=6)

Endpoint: agentwhettersmodeltests.services.ai.azure.com/models
Rate limits: 20 RPM, 20k TPM
Date: 2026-05-28

| Run | Transcript | Score | Correct | Accuracy |
|-----|-----------|-------|---------|----------|
| 1 | 20260528_185613 | +1000 | 154/160 | 96.3% |
| 2 | 20260528_194115 | +980 | 153/160 | 95.6% |
| 3 | 20260528_202707 | +1000 | 154/160 | 96.3% |
| 4 | 20260528_211259 | +1000 | 154/160 | 96.3% |
| 5 | 20260528_215851 | +980 | 153/160 | 95.6% |
| 6 | 20260528_224443 | +980 | 153/160 | 95.6% |

Mean score: +990 (std dev 10)
Mean accuracy: 95.9% (153.5/160)
Range: +980 to +1000

Transcripts located at: pragmatic_builder/logs/transcripts/<timestamp>/

## Llama 4 Maverick 17B (n=6)

Endpoint: agentwhettersmodeltests.services.ai.azure.com/openai/v1/
Model name: Llama-4-Maverick-17B-128E-Instruct-FP8
Rate limits: 20 RPM, 20k TPM
Date: 2026-05-28 to 2026-05-29
Note: api-version query param must NOT be set for this endpoint (returns "API version not supported")

| Run | Transcript | Score | Correct | Accuracy |
|-----|-----------|-------|---------|----------|
| 1 | 20260528_234853 | +840 | 146/160 | 91.3% |
| 2 | 20260529_003406 | +840 | 146/160 | 91.3% |
| 3 | 20260529_012029 | +840 | 146/160 | 91.3% |
| 4 | 20260529_020651 | +840 | 146/160 | 91.3% |
| 5 | 20260529_025313 | +840 | 146/160 | 91.3% |
| 6 | 20260529_033935 | +840 | 146/160 | 91.3% |

Mean score: +840 (std dev 0)
Mean accuracy: 91.3% (146/160)
Range: +840 to +840

All 6 runs produced identical results. At temperature=0.1 this model is fully deterministic
on this task, making the same 14 errors in every run.

Transcripts located at: pragmatic_builder/logs/transcripts/<timestamp>/

## Comparison (all models, full pipeline)

| Model | n | Mean Score | Accuracy | Std Dev |
|-------|---|-----------|----------|---------|
| DeepSeek V3.2 | 6 | +990 | 95.9% | 10 |
| GPT-4o-mini | 12 | +947 | 94.6% | — |
| Nemotron-3 Super (edge) | 6 | +943 | 94.5% | — |
| Llama 4 Maverick 17B | 6 | +840 | 91.3% | 0 |
| GPT-4o | 6 | +810 | 90.3% | — |
| Kimi K2.5 | 6 | +623 | 84.5% | 58 |

Note: the Nemotron-3 Super edge figure above, +943 and 94.5 percent, is
from an earlier run on an older vLLM revision. Later runs on an updated
vLLM achieved better performance, +993 and 96.0 percent, which is the
figure reported in the paper.
