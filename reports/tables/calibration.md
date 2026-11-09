| Model | Variant | ECE (95% subject CI) | NLL | Brier | PR-AUC |
|---|---|---|---|---|---|
| EfficientNet-B0 (seed 42) | raw | 0.0425 [0.026, 0.109] | 0.2753 | 0.0802 | 0.9710 |
| EfficientNet-B0 (seed 42) | temperature scaled (cross-fit) | 0.0292 [0.020, 0.112] | 0.2658 | 0.0796 | 0.9692 |
| MobileNetV3-L | raw | 0.0646 [0.034, 0.145] | 0.3094 | 0.0926 | 0.9704 |
| MobileNetV3-L | temperature scaled (cross-fit) | 0.0635 [0.027, 0.164] | 0.3020 | 0.0919 | 0.9692 |
| Ensemble of 5 B0 seeds | raw | 0.0210 [0.019, 0.104] | 0.2575 | 0.0792 | 0.9737 |
| Ensemble of 5 B0 seeds | temperature scaled (cross-fit) | 0.0232 [0.022, 0.107] | 0.2621 | 0.0798 | 0.9722 |
| EfficientNet-B0 (seed 42) | MC dropout (20 passes) | 0.0398 [0.023, 0.108] | 0.2747 | 0.0801 | 0.9710 |
