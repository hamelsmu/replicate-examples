
Install special version of cog

```bash
sudo curl -o /usr/local/bin/cog -L "https://github.com/replicate/cog/releases/download/v0.10.0-alpha5/cog_$(uname -s)_$(uname -m)"
sudo chmod +x /usr/local/bin/cog
```


Build and run container

```bash
cog build -t cog-vllm
cog run -e CUDA_VISIBLE_DEVICES=0 -p 5000 /bin/bash
# in the container run this for debugging
python predict.py
```

or

```bash
cog predict -e CUDA_VISIBLE_DEVICES=0 -i prompt="Write a blogpost about SEO directed at a technical audience" -i max_new_tokens=512 -i temperature=0.6 -i top_p=0.9 -i top_k=50 -i presence_penalty=0.0 -i frequency_penalty=0.0 -i prompt_template="<s>[INST] {prompt} [/INST] "
```
