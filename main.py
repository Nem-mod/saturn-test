from langchain.llms import LlamaCpp

model_path = r'llama-2-70b-chat.ggmlv3.q3_K_L.bin'

llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=84,
    n_ctx=512,
    temperature=0,
    n_gqa=8
)

# time.sleep(10)
output = llm("Q: Name the planets in the solar system? A: ", max_tokens=32