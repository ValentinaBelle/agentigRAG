from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SummaryIndex, VectorStoreIndex
from llama_index.core.tools import QueryEngineTool
from llama_index.core.query_engine.router_query_engine import RouterQueryEngine
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.schema import MetadataFilters, FilterCondition
from llama_index.core.tools import FunctionTool

def get_router_query_engine(file_path: str, llm=None, embed_model=None, book=None):
    # Создаем router
    llm = llm or OpenAI(model="gpt-3.5-turbo")
    embed_model = embed_model or OpenAIEmbedding(model="text-embedding-ada-002")

    Settings.llm = llm
    Settings.embed_model = embed_model

    # Загружаем документ
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

    # Делим на куски текст
    splitter = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
    nodes = splitter.get_nodes_from_documents(documents)

    # Общий обзор документа
    summary_index = SummaryIndex(nodes)
    
    # Поиск конкретной информации
    vector_index = VectorStoreIndex(nodes, embed_model=embed_model)

    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True
    )
    vector_query_engine = vector_index.as_query_engine(
        similarity_top_k=3,
        response_mode="compact",
    )

    summary_tool = QueryEngineTool.from_defaults(
        query_engine=summary_query_engine,
        description=(
            "Используйте ТОЛЬКО для общего обзора документа"
        ),
    )

    vector_tool = QueryEngineTool.from_defaults(
        query_engine=vector_query_engine,
        description=(
            "Используйте для выделения запрашиваемой информации документа"
        ),
    )
def vector_query_with_pages(query: str, page_numbers=None):
        # Фильтры по страницам
        page_numbers = page_numbers or []
        if page_numbers:
            metadata_dicts = [{"key": "page_label", "value": p} for p in page_numbers]
            filters = MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
            qe = vector_index.as_query_engine(
                similarity_top_k=3,
                filters=filters,
                response_mode="compact"
            )
        else:
            qe = vector_index.as_query_engine(
                similarity_top_k=3, 
                response_mode="compact"
            )
        return qe.query(query)
    
    vector_advanced_tool = FunctionTool.from_defaults(
        name="vector_page_search",
        fn=vector_query_with_pages,
        description="Используйте для точного поиска по страницам"
    )

# LLM селектор, который на выходе выдает JSON для парсинга => вызывает summary_tool()=>Возвращает ответ
    query_engine = RouterQueryEngine(
        selector=LLMSingleSelector.from_defaults(),
        query_engine_tools=[
            summary_tool,
            vector_tool,
            vector_advanced_tool
        ],
        verbose=True
    )
    return query_engine

# тестируем
if __name__ == "__main__":
    pdf_files = []
    all_files = os.listdir(".")  # ['script.py', 'book.pdf', ...]
    for f in all_files:
        if f.endswith('.pdf'):  # проверка
            pdf_files.append(f)  # добавляем
    if pdf_files:
        engine = get_router_query_engine(pdf_files[0])
        return query_engine
