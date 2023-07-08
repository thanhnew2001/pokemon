# -*- coding: utf-8 -*-
"""Inference

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1G002cboizRn6fwuVSU9sC9HFKBi8NDu8

# DICA-Inference

## Import libraries
"""

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# pip install -Uqqq pip --progress-bar off
# pip install -qqq bitsandbytes==0.39.0 --progress-bar off
# pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
# pip install -qqq -U git+https://github.com/huggingface/transformers.git@e03a9cc --progress-bar off
# pip install -qqq -U git+https://github.com/huggingface/peft.git@42a184f --progress-bar off
# pip install -qqq -U git+https://github.com/huggingface/accelerate.git@c9fbb71 --progress-bar off
# pip install -qqq datasets==2.12.0 --progress-bar off
# pip install -qqq loralib==0.1.1 --progress-bar off
# pip install -qqq einops==0.6.1 --progress-bar off
# pip install -qqq accelerate==0.20.3 --progress-bar off
# pip install scikit-learn absl-py nltk rouge_score deepspeed wandb scipy jsonargparse

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# git clone https://github.com/Lightning-AI/lit-parrot
# cd lit-parrot
# pip install .

# %%bash
# cd lit-parrot
# python scripts/download.py --repo_id tiiuae/falcon-7b
# python scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/tiiuae/falcon-7b

# %%bash
# cd lit-parrot
# python generate/base.py --prompt "How can I create an account?" --checkpoint_dir checkpoints/tiiuae/falcon-7b --quantize llm.int8 --max_new_tokens 200

"""## Classical Inference"""

import json
import os
from pprint import pprint
import bitsandbytes as bnb
import pandas as pd
import torch
import torch.nn as nn
import transformers
from datasets import load_dataset, Dataset
from huggingface_hub import notebook_login
from peft import (
LoraConfig,
PeftConfig,
PeftModel,
get_peft_model,
prepare_model_for_kbit_training,
)
from transformers import (
AutoConfig,
AutoModelForCausalLM,
AutoTokenizer,
BitsAndBytesConfig,
)
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

"""# Load the finetune model"""

PEFT_MODEL = "nhat117/test5"

config = PeftConfig.from_pretrained(PEFT_MODEL)
bnb_config = BitsAndBytesConfig(load_in_4bit= True,
bnb_4bit_use_double_quant=True,
bnb_4bit_quant_type="nf4",
bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
	config.base_model_name_or_path,
	return_dict=True,
	quantization_config=bnb_config,
	device_map="auto",
	trust_remote_code=True,
	)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
model = PeftModel.from_pretrained(model, PEFT_MODEL)

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.temperature = 0.7
generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id

from transformers import pipeline
pipeline = pipeline(model = model, tokenizer = tokenizer, task = 'text-generation')

"""## Perform inference

* `Question`: Chào bác sĩ, Mẹ em 42 tuổi gần đây hay mệt mỏi, khó thở, đo nhịp tim 52 nhịp/ phút chậm so với người bình thường. Đi chụp tim không sao nhưng chụp thấy có vết mờ ở phổi. Vậy bác sĩ cho em hỏi chụp X quang phổi thấy có vết mờ là dấu hiệu bệnh gì? Em cảm ơn bác sĩ.
* `Expected Answer` : 'Chào bạn, Với câu hỏi “Chụp X quang phổi thấy có vết mờ là dấu hiệu bệnh gì?”, bác sĩ xin giải đáp như sau: Theo như các triệu chứng bạn mô tả bệnh của người nhà, tốt nhất bạn nên đưa người nhà khám chuyên khoa Tim mạch và Hô hấp. Các bác sĩ cần thăm khám bệnh nhân, kết hợp xem phim, điện tim,.... từ đó mới có thể đưa ra chẩn đoán bệnh cho người nhà. Nếu bạn còn thắc mắc về chụp X quang phổi thấy có vết mờ, bạn có thể đến bệnh viện để kiểm tra và tư vấn thêm. Cảm ơn bạn đã đặt câu hỏi cho DICA. Chúc bạn có thật nhiều sức khỏe. Trân trọng!'
"""

prompt = f"""
<human>  Chào bác sĩ, Mẹ em 42 tuổi gần đây hay mệt mỏi, khó thở, đo nhịp tim 52 nhịp/ phút chậm so với người bình thường. Đi chụp tim không sao nhưng chụp thấy có vết mờ ở phổi. Vậy bác sĩ cho em hỏi chụp X quang phổi thấy có vết mờ là dấu hiệu bệnh gì? Em cảm ơn bác sĩ.
<assistant>:
""".strip()
print(prompt)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# pipeline(prompt,generation_config = generation_config)

# Commented out IPython magic to ensure Python compatibility.
# %%time
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# encoding =tokenizer(prompt, return_tensors = 'pt').to(device)
# 
# with torch.inference_mode():
#     outputs = model.generate(input_ids = encoding.input_ids, attention_mask = encoding.attention_mask,
#                             generation_config = generation_config)
# 
# print(tokenizer.decode(outputs[0],skip_special_token = True))

# Commented out IPython magic to ensure Python compatibility.
# %%bash
# pip install fastapi nest-asyncio pyngrok uvicorn

def ask(message):
  prompt = f"""
<human>Xin chào tôi tên là {message.sender}.Câu hỏi của tôi là {message.message}
<assistant>:
""".strip()
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  encoding =tokenizer(prompt, return_tensors = 'pt').to(device)

  with torch.inference_mode():
    outputs = model.generate(input_ids = encoding.input_ids, attention_mask = encoding.attention_mask,
                            generation_config = generation_config)

    message = tokenizer.decode(outputs[0],skip_special_token = True)
  return message
ask("Chào bác sĩ, Mẹ em 42 tuổi gần đây hay mệt mỏi, khó thở, đo nhịp tim 52 nhịp/ phút chậm so với người bình thường. Đi chụp tim không sao nhưng chụp thấy có vết mờ ở phổi. Vậy bác sĩ cho em hỏi chụp X quang phổi thấy có vết mờ là dấu hiệu bệnh gì? Em cảm ơn bác sĩ.")


