# Fine-tune MNLI

## BERT

|ex_no|train acc|dev max matched acc|dev max matched acc checkpoint|dev max mismatched acc|dev max mismatched acc checkpoint|model_version|epoch|lr|batch|beta1|beta2|eps|l2 weight decay|remove weight decay on bias and LayerNorm|linear schedular|warm up step|dropout|seed|
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
|0|0.856631755828857|0.7968415690269995|25000|0.8091944670463792|29000|bert-base-cased|3|3e-5|32|0.9|0.999|1e-8|0.01|False|False|0|0.1|777|
|1|0.903957724571228|0.8266938359653592|35000|0.8332994304312449|33000|bert-base-cased|3|3e-5|32|0.9|0.999|1e-8|0.01|False|True|10000|0.1|777|
|2|0|0|0|0|0|bert-base-cased|3|4e-5|32|0.9|0.999|1e-8|0.01|False|True|10000|0.1|777|
|3|0|0|0|0|0|bert-base-cased|3|3e-5|32|0.9|0.999|1e-8|0.01|True|True|10000|0.1|777|
