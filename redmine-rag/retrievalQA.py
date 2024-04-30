from langchain_community.llms.llamacpp import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA

# インデックスのパス
index_path = "./storage"

# モデルのパス
model_path = "./ELYZA-japanese-Llama-2-7b-fast-instruct-q4_K_M.gguf"

# 埋め込みモデルの読み込み
embedding_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large")

# インデックスの読み込み
index = FAISS.load_local(
    folder_path=index_path,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,
)

# プロンプトテンプレートの定義
question_prompt_template = """

{context}

Question: {question}
Answer: """

# プロンプトの設定
QUESTION_PROMPT = PromptTemplate(
    template=question_prompt_template,  # プロンプトテンプレートをセット
    input_variables=["context", "question"],  # プロンプトに挿入する変数
)

# モデルの設定
llm = LlamaCpp(
    model_path=model_path,
    n_gpu_layers=25,  # gpuに処理させるlayerの数
    stop=["Question:", "Answer:"],  # 停止文字列
)

# （RAG用）質問回答chainの設定
chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=index.as_retriever(
        search_kwargs={"k": 2}  # indexから上位いくつの検索結果を取得するか
    ),
    chain_type_kwargs={"prompt": QUESTION_PROMPT},  # プロンプトをセット
    chain_type="stuff",  # 検索した文章の処理方法
    return_source_documents=True,  # indexの検索結果を確認する場合はTrue
)

# 質問文
question = "{some context}"

# LLMの回答生成
response = chain.invoke(question)

# indexの検索結果を確認
for i, source in enumerate(response["source_documents"], 1):
    print(f"\nindex: {i}----------------------------------------------------")
    print(f"{source.page_content}")
    print("---------------------------------------------------------------")

# 回答を確認
response_result = response["result"]
print(f"\nAnswer: {response_result}")
