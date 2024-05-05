from langchain import hub
from langchain_community.llms.llamacpp import LlamaCpp
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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


_chain = None


def _get_chain():
    if _chain is None:
        retriever = _load_vectorstore().as_retriever()
        prompt = hub.pull("rlm/rag-prompt")

        # モデルのパス
        model_path = "./ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf"

        # モデルの設定
        llm = LlamaCpp(
            model_path=model_path,
            temperature=0.2,
            n_ctx=4096,
            top_p=1,
            n_gpu_layers=25,  # gpuに処理させるlayerの数
        )  # type: ignore

        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        global _chain
        _chain = rag_chain

    return _chain


def generate(question: str):
    _get_chain().invoke(question)
