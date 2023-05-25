import accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

MODEL_NAME = 'timdettmers/guanaco-33b-merged'

n_gpus = torch.cuda.device_count()

max_memory = {i: '10GB' for i in range(n_gpus)}

model = AutoModelForCausalLM.from_pretrained(
    pretrained_model_name_or_path=MODEL_NAME,
    load_in_4bit=True,
    device_map='sequential',
    max_memory=max_memory,
    torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4'
    ),
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


if __name__ == '__main__':
    input_ids = tokenizer.encode("Once upon a time", return_tensors="pt").to('cuda:1') 
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            do_sample=True,
            min_length=0,
            max_length=100,
            top_p=0.9,
            temperature=0.7,
        )

    generated_text = tokenizer.decode(
        [el.item() for el in generated_ids[0]], skip_special_tokens=True)

    print(generated_text)
