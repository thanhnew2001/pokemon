from flask import Flask, request, jsonify
import json
import sys
import time
import warnings
from functools import partial
from pathlib import Path
from typing import Literal

import lightning as L
import torch
from lightning.fabric.strategies import FSDPStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from generate.base import generate
from lit_gpt import Tokenizer
from lit_gpt.adapter import Block
from lit_gpt.adapter import GPT, Config
from lit_gpt.adapter_v2 import add_adapter_v2_parameters_to_linear_layers
from lit_gpt.utils import lazy_load, check_valid_checkpoint_dir, quantization
from scripts.prepare_alpaca import generate_prompt


app = Flask(__name__)

prompt = "What food do lamas eat?"
user_input = ""
adapter_path = Path("out/adapter_v2/alpaca/lit_model_adapter_finetuned.pth")
checkpoint_dir = Path(f"checkpoints/stabilityai/stablelm-base-alpha-3b")
quantize = None
max_new_tokens = 100
top_k = 200
temperature = 0.8
strategy = "auto"
devices = 1
precision = "bf16-true"

if strategy == "fsdp":
    auto_wrap_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
    strategy = FSDPStrategy(auto_wrap_policy=auto_wrap_policy, cpu_offload=False)
fabric = L.Fabric(devices=devices, precision=precision, strategy=strategy)
fabric.launch()

check_valid_checkpoint_dir(checkpoint_dir)

with open(checkpoint_dir / "lit_config.json") as fp:
    config = Config(**json.load(fp))

if quantize is not None and devices > 1:
    raise NotImplementedError
if quantize == "gptq.int4":
    model_file = "lit_model_gptq.4bit.pth"
    if not (checkpoint_dir / model_file).is_file():
        raise ValueError("Please run `python quantize/gptq.py` first")
else:
    model_file = "lit_model.pth"
checkpoint_path = checkpoint_dir / model_file

fabric.print(f"Loading model {str(checkpoint_path)!r} with {config.__dict__}", file=sys.stderr)
t0 = time.time()
with fabric.init_module(empty_init=True), quantization(quantize):
    model = GPT(config)
    add_adapter_v2_parameters_to_linear_layers(model)
fabric.print(f"Time to instantiate model: {time.time() - t0:.02f} seconds.", file=sys.stderr)

t0 = time.time()
with lazy_load(checkpoint_path) as checkpoint, lazy_load(adapter_path) as adapter_checkpoint:
    checkpoint.update(adapter_checkpoint.get("model", adapter_checkpoint))
    model.load_state_dict(checkpoint, strict=quantize is None)
fabric.print(f"Time to load the model weights: {time.time() - t0:.02f} seconds.", file=sys.stderr)

model.eval()
model = fabric.setup(model)

tokenizer = Tokenizer(checkpoint_dir)


@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.get_json()

    prompt = data.get('prompt', 'What food do lamas eat?')
    user_input = data.get('input', '')

    sample = {"instruction": prompt, "input": user_input}
    prompt = generate_prompt(sample)
    encoded = tokenizer.encode(prompt, device=model.device)
    prompt_length = encoded.size(0)
    max_returned_tokens = prompt_length + max_new_tokens

    t0 = time.perf_counter()
    y = generate(
        model,
        encoded,
        max_returned_tokens,
        max_seq_length=max_returned_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=tokenizer.eos_id,
    )
    t = time.perf_counter() - t0

    model.reset_cache()
    output = tokenizer.decode(y)
    output = output.split("### Response:")[1].strip()

    # Convert the output to a JSON response
    response = {
        'output': output,
        'time': t,
        'tokens_generated_per_second': (y.size(0) - prompt_length) / t
    }

    return jsonify(response)


if __name__ == '__main__':
    torch.set_float32_matmul_precision("high")
    warnings.filterwarnings(
        # Triggered internally at ../aten/src/ATen/EmptyTensor.cpp:31
        "ignore",
        message="ComplexHalf support is experimental and many operators don't support it yet",
    )

    app.run(debug=True)
