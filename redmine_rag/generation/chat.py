from langchain import hub

from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

import torch
from redmine_rag.core import INDEX_STORAGE


def _load_vectorstore():
    # 埋め込みモデルの読み込み
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    return FAISS.load_local(
        folder_path=INDEX_STORAGE,
        embeddings=embedding_model,
        allow_dangerous_deserialization=True,
    )


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_model():
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        GPTQConfig,
        pipeline,
    )

    model_id = "elyza/ELYZA-japanese-Llama-2-7b-instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    quantization_config = GPTQConfig(
        bits=4, dataset="c4", tokenizer=tokenizer, use_exllama=False, use_cuda_fp16=True
    )
    my_device_map = {
        "model.embed_tokens": "cpu",
        "model.layers": "cpu",
        "model.norm": "cpu",
        "lm_head": "cpu",
    }

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        device_map=my_device_map,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
    ).eval()

    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
    )
    return HuggingFacePipeline(
        pipeline=pipe,
        # model_kwargs=dict(temperature=0.1, do_sample=True, repetition_penalty=1.1)
    )


def get_quantize_model():
    model_path = "./ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf"

    return LlamaCpp(model_path=model_path, n_gpu_layers=-1, n_ctx=2048, max_tokens=512)


_chain = None


def _get_chain():
    global _chain
    if _chain is None:
        retriever = _load_vectorstore().as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        llm = get_quantize_model()

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        _chain = rag_chain

    return _chain


def generate(question: str):
    # _get_chain().invoke(question)
    response = _get_chain().invoke(question)
    print(response)


if __name__ == "__main__":
    generate("富士山の高さは？")
