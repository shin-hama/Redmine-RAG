from typing import Sequence
import bs4
from langchain_community.document_loaders.web_base import WebBaseLoader
from langchain_community.document_loaders.directory import DirectoryLoader
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from redmine_rag.core import INDEX_STORAGE


def _load_webpages(urls: Sequence[str]):
    loader = WebBaseLoader(
        web_paths=urls, bs_kwargs=dict(parse_only=bs4.SoupStrainer("p"))
    )
    return loader.load()


def _load_directory(directory: str):
    loader = DirectoryLoader(directory)
    return loader.load()


def vectorize_webpages(urls: Sequence[str]):
    docs = _load_webpages(urls)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_texts = text_splitter.split_documents(docs)

    # 埋め込みモデルの読み込み
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

    # インデックスの作成
    vectorstore = FAISS.from_documents(
        documents=split_texts,
        embedding=embedding_model,
    )

    # インデックスの保存
    vectorstore.save_local(folder_path=INDEX_STORAGE)
