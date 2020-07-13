# Fine-tune BoolQ 
- [official github link](https://github.com/google-research-datasets/boolean-questions)

## Get Data
```sh
# download
wget https://dl.fbaipublicfiles.com/glue/superglue/data/v2/BoolQ.zip

# move to data folder
mv BoolQ.zip ./data/fine_tune/BoolQ.zip

# extract: 'it will create a new folder: ./data/fine_tune/BoolQ/'
unzip ./data/fine_tune_data/BoolQ.zip -d ./data/fine_tune/ 

# remove redundant files
rm ./data/fine_tune/BoolQ.zip
```


