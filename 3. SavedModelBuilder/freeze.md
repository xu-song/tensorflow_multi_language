



# freeze from pbtxt

```bash
freeze_graph --input_graph=model/saved_model.pbtxt \
  --input_checkpoint=model/variables/variables.data-00000-of-00001 \
  --input_binary=false \
  --output_graph=model/frozen_model.pb \
  --output_node_names=softmax_linear
```


报错

```bash
google.protobuf.text_format.ParseError: 1:1 : 
Message type "tensorflow.GraphDef" has no field named "saved_model_schema_version".
```

1. 可能是版本的问题
2. 可能只需要包含graphdef，不用meta graph



# freeze from pb

```
google.protobuf.message.DecodeError: Error parsing message
```
