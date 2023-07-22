Usage

```bash
python generator_interface.py --model [model_name which may include local path] --tokenizer [tokenizer to use] --port [port] --embedder [embedding model to use]
```

When setting this generator up with a model, keep in mind the following:
1. Upstream services use the path of the model as the model name - so ideally, the model name should be reflected in:

```python
model_name = model_name.split("/")[-1]
```

2. 