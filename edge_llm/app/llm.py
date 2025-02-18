from transformers import pipeline,LlamaForCausalLM, LlamaTokenizer
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline

#Initialize selected model and tokenizer
model_name = "meta-llama/Llama-2-7b-chat-hf"
model = LlamaForCausalLM.from_pretrained(model_name)
tokenizer =  LlamaTokenizer.from_pretrained(model_name)

#Develop a HuggingFace pipeline to download model
pipe = pipeline(
    task="text-generation",
    model=model,
    tokenizer=tokenizer,
    trust_remote_code=True,
    max_new_tokens=100,
    repetition_penalty=1.1,
    model_kwargs={"max_length": 1200, "temperature": 0.01}
)
llm_pipeline = HuggingFacePipeline(pipeline=pipe)
