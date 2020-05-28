# Fine-tune MNLI

## BERT

|ex_no|train acc|dev matched acc|dev mismatched acc|model_version|epoch|lr|batch|beta1|beta2|eps|l2 weight decay|linear schedular|warm up step|dropout|seed|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|0|0|0|0|bert-base-cased|3|3e-5|32|0.9|0.999|1e-8|0.01|False|0|0.1|777|
|1|0|0|0|bert-base-cased|3|3e-5|32|0.9|0.999|1e-8|0.01|True|10000|0.1|777|
|2|0|0|0|bert-base-cased|3|3e-5|32|0.9|0.999|1e-8|0.01|True|10000|0.1|777|
