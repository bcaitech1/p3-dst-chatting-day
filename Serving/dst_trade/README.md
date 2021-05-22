### 1) torch-model-archiver

`torch-model-archiver --model-name TRADE --version 1.0 --serialized-file "./serialized-file/model-27.bin" --export-path "./model_store" --extra-files data_utils.py,model.py,preprocessor.py,exp_config.json,slot_meta.json --handler Dst_custom_handler.py`

### 2) torchserve

* `torchserve --start --model-store ./model_store --ts-config config.properties`

### 3) inference
* `curl http://127.0.0.1:8080/predictions/TRADE -T test.json`

* `curl -X POST http://ec2-18-116-93-203.us-east-2.compute.amazonaws.com:8080/predictions/TRADE -T test.json`