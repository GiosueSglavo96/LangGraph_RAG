import os
import numpy as np

from langchain_core.tools import tool
from src.multi_agent_system.models.embedding_model import get_embedding_model
from src.multi_agent_system.config.rag_settings import Settings

from typing import List, Tuple, Any, Iterable

from langchain.schema import Document
from langchain_community.document_loaders import TextLoader, CSVLoader, PyPDFLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    HnswConfigDiff,
    OptimizersConfigDiff,
    ScalarQuantization,
    ScalarQuantizationConfig,
    PayloadSchemaType,
    FieldCondition,
    MatchValue,
    MatchText,
    Filter,
    SearchParams,
    PointStruct,
)

def load_documents() -> List[Document]:
    docs = []
    input_files_path = "../data/input_files/"

    for filename in os.listdir(input_files_path):
        print(filename)
        complete_file_path = os.path.join(input_files_path, filename)

        if filename.endswith(".txt"):
            loader = TextLoader(complete_file_path, encoding="utf-8")
        
        elif filename.endswith(".csv"):
            loader = CSVLoader(complete_file_path)
        
        elif filename.endswith(".pdf"):
            loader = PyPDFLoader(complete_file_path)
        
        else:
            print(f"❌ Unsupported file format: {filename}")
            continue
        
        try:
            loaded_doc = loader.load()
            docs.extend(loaded_doc)
            print(len(docs), f"📄 Loaded {filename} with {len(loaded_doc)} documents.")
        except Exception as e:
            print(f"⚠️ Error loading {filename}: {e}")
        
    return docs

def get_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(url=settings.qdrant_url)

def split_documents(docs: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", "? ", "! ", "; ", ": ", ", ", " ", ""],
    )
    return splitter.split_documents(docs)

def recreate_collection_for_rag(client: QdrantClient, collection_name: str, vector_size: int):
    
    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        hnsw_config=HnswConfigDiff(
            m=32,             # grado medio del grafo HNSW (maggiore = più memoria/qualità)
            ef_construct=256  # ampiezza lista candidati in fase costruzione (qualità/tempo build)
        ),
        optimizers_config=OptimizersConfigDiff(
            default_segment_number=2  # parallelismo/segmentazione iniziale
        ),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type="int8", always_ram=False)  # on-disk quantization dei vettori
        ),
    )

    # Indice full-text sul campo 'text' per filtri MatchText
    client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT
    )

    # Indici keyword per filtri esatti / velocità nei filtri
    for key in ["doc_id", "source", "title", "lang"]:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=key,
            field_schema=PayloadSchemaType.KEYWORD
        )

def build_points(chunks: List[Document], embeds: List[List[float]]) -> List[PointStruct]:
    pts: List[PointStruct] = []
    for i, (doc, vec) in enumerate(zip(chunks, embeds), start=1):
        payload = {
            "doc_id": doc.metadata.get("id"),
            "source": doc.metadata.get("source"),
            "title": doc.metadata.get("title"),
            "lang": doc.metadata.get("lang", "en"),
            "text": doc.page_content,
            "chunk_id": i - 1
        }
        pts.append(PointStruct(id=i, vector=vec, payload=payload))
    return pts

def upsert_chunks(client: QdrantClient, collection_name: str, chunks: List[Document], embeddings: OpenAIEmbeddings):
    vecs = embeddings.embed_documents([c.page_content for c in chunks])
    points = build_points(chunks, vecs)
    client.upsert(collection_name=collection_name, points=points, wait=True)

def qdrant_semantic_search(
    client: QdrantClient,
    collection_name: str,
    query: str,
    embeddings: OpenAIEmbeddings,
    limit: int,
    with_vectors: bool = False
):
    qv = embeddings.embed_query(query)
    res = client.query_points(
        collection_name=collection_name,
        query=qv,
        limit=limit,
        with_payload=True,
        with_vectors=with_vectors,
        search_params=SearchParams(
            hnsw_ef=256,  # ampiezza lista in fase di ricerca (recall/latency)
            exact=False   # True = ricerca esatta (lenta); False = ANN HNSW
        ),
    )
    return res.points

def qdrant_text_prefilter_ids(
    client: QdrantClient,
    collection_name: str,
    query: str,
    max_hits: int
) -> List[int]:
    # Scroll con filtro MatchText per ottenere id dei match testuali
    # (nota: scroll è paginato; qui prendiamo solo i primi max_hits per semplicità)
    matched_ids: List[int] = []
    next_page = None
    while True:
        points, next_page = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[FieldCondition(key="text", match=MatchText(text=query))]
            ),
            limit=min(256, max_hits - len(matched_ids)),
            offset=next_page,
            with_payload=False,
            with_vectors=False,
        )
        matched_ids.extend([p.id for p in points])
        if not next_page or len(matched_ids) >= max_hits:
            break
    return matched_ids

def mmr_select(
    query_vec: List[float],
    candidates_vecs: List[List[float]],
    k: int,
    lambda_mult: float
) -> List[int]:
    
    V = np.array(candidates_vecs, dtype=float)
    q = np.array(query_vec, dtype=float)

    def cos(a, b):
        na = (a @ a) ** 0.5 + 1e-12
        nb = (b @ b) ** 0.5 + 1e-12
        return float((a @ b) / (na * nb))

    sims = [cos(v, q) for v in V]
    selected: List[int] = []
    remaining = set(range(len(V)))

    while len(selected) < min(k, len(V)):
        if not selected:
            # pick the highest similarity first
            best = max(remaining, key=lambda i: sims[i])
            selected.append(best)
            remaining.remove(best)
            continue
        best_idx = None
        best_score = -1e9
        for i in remaining:
            max_div = max([cos(V[i], V[j]) for j in selected]) if selected else 0.0
            score = lambda_mult * sims[i] - (1 - lambda_mult) * max_div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(best_idx)
        remaining.remove(best_idx)
    return selected

def hybrid_search(
    client: QdrantClient,
    collection_name: str,
    settings: Settings,
    query: str,
    embeddings: OpenAIEmbeddings
):
    
    # (1) semantica
    sem = qdrant_semantic_search(
        client, collection_name, query, embeddings,
        limit=settings.top_n_semantic, with_vectors=True
    )
    if not sem:
        return []

    # (2) full-text prefilter (id)
    text_ids = set(qdrant_text_prefilter_ids(client, collection_name, query, settings.top_n_text))

    # Normalizzazione score semantici per fusione
    scores = [p.score for p in sem]
    smin, smax = min(scores), max(scores)
    def norm(x):  # robusto al caso smin==smax
        return 1.0 if smax == smin else (x - smin) / (smax - smin)

    # (3) fusione con boost testuale
    fused: List[Tuple[int, float, Any]] = []  # (idx, fused_score, point)
    for idx, p in enumerate(sem):
        base = norm(p.score)                    # [0..1]
        fuse = settings.alpha * base
        if p.id in text_ids:
            fuse += settings.text_boost         # boost additivo
        fused.append((idx, fuse, p))

    # ordina per fused_score desc
    fused.sort(key=lambda t: t[1], reverse=True)

    # MMR opzionale per diversificare i top-K
    if settings.mmr:
        qv = embeddings.embed_query(query)
        # prendiamo i primi N dopo fusione (es. 30) e poi MMR per final_k
        N = min(len(fused), max(settings.final_top_k * 5, settings.final_top_k))
        cut = fused[:N]
        vecs = [sem[i].vector for i, _, _ in cut]
        mmr_idx = mmr_select(qv, vecs, settings.final_top_k, settings.mmr_lambda)
        picked = [cut[i][2] for i in mmr_idx]
        return picked

    # altrimenti, prendi i primi final_k dopo fusione
    return [p for _, _, p in fused[:settings.final_top_k]]

def format_docs_for_prompt(points: Iterable[Any]) -> str:
    blocks = []
    for p in points:
        pay = p.payload or {}
        src = pay.get("source", "unknown")
        blocks.append(f"[source:{src}] {pay.get('text','')}")
    return "\n\n".join(blocks)

def execute_rag(query:str):
    """
    This funcition, given a query written by the user in natural language, 
    performs a RAG (Retrieval-Augmented Generation) process to retrieve relevant documents from a Qdrant vector store 
    and format them for further use.

    Args:
        query (str): The user's natural language query.

    Returns:
        str: Formatted string containing the retrieved documents' content and sources.
    """
    settings = Settings()

    embedding_model = get_embedding_model()

    vector_store = get_qdrant_client(settings)
    vector_size = 1536

    COLLECTION_NAME = "medicine_collection"

    docs = load_documents()
    print("Docs loaded:", len(docs))
    chunks = split_documents(docs, settings)
    recreate_collection_for_rag(vector_store, COLLECTION_NAME, vector_size)
    upsert_chunks(vector_store, COLLECTION_NAME, chunks, embedding_model)

    points = hybrid_search(vector_store, COLLECTION_NAME, settings, query, embedding_model)
    print("=" * 80)
    print("Q:", query)
    if not points:
        print("No results found in vector store.")
    for p in points:
        print(f"- id={p.id} score={p.score:.4f} src={p.payload.get('source')}")
    
    context = format_docs_for_prompt(points)

    return context
