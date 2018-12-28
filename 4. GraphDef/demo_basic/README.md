

meta_graph与graph的区别

## save graph & checkpoint

```bash
python save
```


## freeze & restore

freeze graph

```bash
# 或者采用
TRAIN_DIR=model
freeze_graph \
    --input_graph=${TRAIN_DIR}/graph.pb \
    --input_binary=true \
    --input_checkpoint=${TRAIN_DIR}/ckpt \
    --output_graph=${TRAIN_DIR}/frozen_graph.pb \
    --output_node_names=res
```

optinal: convert pb to pbtxt
```bash
python converter
```


## js

```bash
TRAIN_DIR=model
tensorflowjs_converter \
    --input_format=tf_frozen_model \
    --output_node_names='res' \
    --saved_model_tags=demo \
    ${TRAIN_DIR}/frozen_graph.pb \
    ${TRAIN_DIR}/web_model
```

```bash
node restore.js
```