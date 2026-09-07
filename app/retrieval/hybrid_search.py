from typing import List, Dict, Any, Optional
from app.retrieval.query_router import QueryRouter
from app.retrieval.query_rewriter import QueryRewriter
from app.retrieval.reranker import DocumentReranker
from app.retrieval.glossary import expand_query
from app.generation.citation import clean_source_filename
from pymongo import MongoClient
from app.config import settings
from functools import lru_cache
from collections import defaultdict
from app.services.embedding_client import embed_query
from app.services.qdrant_store import search_similar_blocks
from qdrant_client.http import models as qmodels
from app.services.mongo_client import get_chunks_collection, get_parent_chunk, refresh_known_tickers
from collections import OrderedDict, defaultdict

import bm25s
import numpy as np
import unicodedata
import re

mongo_client = MongoClient(getattr(settings, "MONGO_URI", "mongodb://mongo:27017"))

VI_STOPWORDS = {"là", "của", "và", "các", "một", "những", "cho", "về", "trong", "đã", "này", "được"}  # bổ sung thêm nếu cần

# TICKER_MAPPING viết tay + REPORT_SCOPE_QUERY_KEYWORDS (dò keyword rule-
# based) đã bị LOẠI BỎ khỏi đây. Ticker và report_scope giờ do LLM
# (gpt-4o-mini) trích xuất/chuẩn hoá trực tiếp từ câu hỏi ngay tại
# QueryRewriter.rewrite_and_extract_metadata() (nhánh financial_search) và
# CalculationService.extract_intent() (nhánh calculation) -- cả 2 đều đối
# chiếu với danh sách ticker THẬT đang có trong hệ thống
# (app.services.mongo_client.get_known_tickers()/known_tickers_prompt_text()),
# thay vì 1 dict tĩnh phải sửa code mỗi khi ingest thêm công ty/biến thể tên
# gọi mới. Xem process_user_query() bên dưới và app/calculation/calculation_service.py.

REPORT_SCOPE_FILENAME_HINTS: Dict[str, List[str]] = {
    "parent": ["congtyme", "cty me", "rieng", "Congtyme"],
    "consolidated": ["hopnhat", "hop nhat", "Hopnhat"],
}

K_RRF = 60                     # hằng số k trong công thức RRF: 1 / (k + rank)
BM25_TOP_K = 30                # số ứng viên BM25 lấy mỗi query
DENSE_TOP_K = 30               # số ứng viên Dense lấy mỗi query
FUSION_TOP_K = 30              # số chunks sau RRF đưa vào reranker
TOP_K = 10                     # số chunks cuối cùng sau reranker

# ----------------------------------------------------------------------------
# Ablation toggle -- BẬT/TẮT từng thành phần của Hybrid Search để đánh giá
# riêng từng bước (dùng trong evaluation/*.py, vd ragas_eval.py --stage).
#
#   "bm25"     : CHỈ chạy BM25   -- tắt Dense, RRF, Reranker.
#   "dense"    : CHỈ chạy Dense  -- tắt BM25, RRF, Reranker.
#   "rrf"      : chạy BM25 + Dense + RRF -- tắt Reranker.
#   "reranker" : chạy BM25 + Dense, BỎ RRF -- gộp thẳng (union) candidate
#                của cả 2 rồi đưa nguyên vào Reranker.
#   "full"     : hành vi ĐẦY ĐỦ như code gốc (BM25 + Dense + RRF + Reranker),
#                mặc định, KHÔNG đổi hành vi cũ.
# ----------------------------------------------------------------------------
STAGE_BM25 = "bm25"
STAGE_DENSE = "dense"
STAGE_RRF = "rrf"
STAGE_RERANKER = "reranker"
STAGE_FULL = "full"
VALID_STAGES = (STAGE_FULL, STAGE_BM25, STAGE_DENSE, STAGE_RRF, STAGE_RERANKER)

_MAX_TRACKED_SESSIONS = 500  # tránh self._session_state phình vô hạn theo thời gian chạy
 
def _strip_diacritics(s: str) -> str:
    s = unicodedata.normalize("NFD", s or "")
    return "".join(c for c in s if unicodedata.category(c) != "Mn")
 
 
def extract_report_scope(metadata_filter: Optional[dict]) -> Optional[str]:
    scope = (metadata_filter or {}).get("report_scope")
    return scope if scope in REPORT_SCOPE_FILENAME_HINTS else None
 
 
def _source_file_matches_scope(source_file: Optional[str], scope: Optional[str]) -> bool:
    """True nếu source_file khớp đúng scope yêu cầu, HOẶC nếu không xác
    định được scope của chính source_file đó (an toàn hơn là lỡ loại bỏ
    nhầm dữ liệu hợp lệ khi tên file không theo đúng quy ước đặt tên).
 
    QUAN TRỌNG: cả Qdrant lẫn Mongo hiện đều lưu 'source_file' là ĐƯỜNG DẪN
    ĐẦY ĐỦ (không chỉ tên file) -- nếu so hint trực tiếp trên cả path, một
    thư mục cha (vd tên ticker/tên công ty) tình cờ chứa chuỗi trùng hint
    ("rieng", "hopnhat"...) có thể gây match sai. Dùng clean_source_filename()
    (đã có sẵn trong app/generation/citation.py, cùng hàm mà tầng citation
    dùng để hiển thị tên file cho người dùng) để CHỈ lấy phần tên file cuối
    cùng trước khi so khớp, tách biệt hẳn khỏi cấu trúc thư mục phía trước."""
    if not scope or not source_file:
        return True
    fname = _strip_diacritics(clean_source_filename(str(source_file)).lower())
    if any(h in fname for h in REPORT_SCOPE_FILENAME_HINTS.get(scope, [])):
        return True
    other_hints = [
        h for s, hs in REPORT_SCOPE_FILENAME_HINTS.items() if s != scope for h in hs
    ]
    if any(h in fname for h in other_hints):
        return False
    return True
 
 
def _report_scope_mongo_regex(scope: str) -> Optional[str]:
    """Build regex cho $regex trên field 'source_file' của Mongo (đường dẫn
    ĐẦY ĐỦ, không phải tên file trần). Neo hint vào ĐOẠN CUỐI CÙNG của path
    (sau dấu '/' hoặc '\\' gần nhất, tới hết chuỗi) bằng [^/\\\\]*$, để hint
    chỉ được tính là khớp khi nó nằm trong chính TÊN FILE, không phải trong
    tên thư mục cha (vd thư mục theo ticker) đứng trước nó trong path."""
    hints = REPORT_SCOPE_FILENAME_HINTS.get(scope, [])
    if not hints:
        return None
    alt = "|".join(re.escape(h) for h in hints)
    return rf"({alt})[^/\\]*$"

def _normalize_cid(cid: Any) -> str:
    """Chuẩn hoá chunk_id/parent_id về dạng hex thô không gạch ngang (đồng bộ với MongoDB),
    loại bỏ hoàn toàn dấu gạch ngang do Qdrant tự động ép sang định dạng UUID chuẩn.
    """
    if not cid:
        return ""
    return str(cid).replace("-", "").strip()

def ticker_year_filter(metadata_filter: Optional[dict]) -> tuple[dict, Optional[qmodels.Filter]]:
    """
    Chuẩn hóa ticker/year và trả về đồng thời:
    - mongo_query: Dict điều kiện lọc cho MongoDB (dùng trong BM25)
    - qdrant_filter: Object qmodels.Filter cho Qdrant (dùng trong Dense)
    """
    if not metadata_filter:
        return {}, None

    mongo_query = {}
    must_qdrant = []

    # 1. Chuẩn hóa Ticker -- ticker đã được LLM (QueryRewriter/CalculationService,
    # dùng gpt-4o-mini, đối chiếu known_tickers_prompt_text()) chuẩn hoá
    # đúng theo danh sách ticker thật trong hệ thống ngay từ bước trích
    # xuất. Ở đây chỉ strip/upper cho an toàn (phòng khi gọi trực tiếp
    # hàm này với 1 ticker chưa qua LLM, vd truyền tay lúc test/debug).
    raw_ticker = metadata_filter.get("ticker")
    if raw_ticker:
        clean_ticker = str(raw_ticker).strip().upper()

        mongo_query["ticker"] = clean_ticker
        must_qdrant.append(
            qmodels.FieldCondition(
                key="ticker", match=qmodels.MatchValue(value=clean_ticker)
            )
        )

    # 2. Chuẩn hóa Year
    raw_year = metadata_filter.get("year")
    if raw_year:
        try:
            clean_year = int(raw_year)
        except (ValueError, TypeError):
            clean_year = raw_year
            
        mongo_query["year"] = clean_year
        must_qdrant.append(
            qmodels.FieldCondition(
                key="year", match=qmodels.MatchValue(value=clean_year)
            )
        )

    qdrant_filter = qmodels.Filter(must=must_qdrant) if must_qdrant else None
    return mongo_query, qdrant_filter

def _load_bm25_corpus(ticker: Optional[str] = None, year: Optional[int] = None, report_scope: Optional[str] = None):
    mongo_query: dict = {}
    if ticker:
        mongo_query["ticker"] = ticker
    if year:
        mongo_query["year"] = year
    if report_scope:
        pattern = _report_scope_mongo_regex(report_scope)
        if pattern:
            mongo_query["source_file"] = {"$regex": pattern, "$options": "i"}
    docs = list(
        get_chunks_collection().find(
            mongo_query, {"chunks": 1, "ticker": 1, "year": 1, "source_file": 1}
        )
    )
    print(f"[BM25] Đã lọc ra {len(docs)} file khớp với công ty và năm {mongo_query}:")
    for doc in docs:
        print(f"File ID: {doc['_id']}")
    corpus_ids: List[str] = []
    corpus_texts: List[str] = []
    lookup: Dict[str, dict] = {}

    for doc in docs:
        chunks = doc.get("chunks", [])
        for local_idx, chunk in enumerate(chunks):            
            raw_cid = str(chunk.get("_id") or chunk.get("chunk_id"))
            cid = _normalize_cid(raw_cid)
            text = (chunk.get("text") or "").strip()
            if text:
                corpus_ids.append(cid)
                corpus_texts.append(text)
                lookup[cid] = {
                    "content": text, 
                    "parent_id": _normalize_cid(chunk.get("parent_id")) or cid,
                    "ticker": chunk.get("ticker"),
                    "year": chunk.get("year"),
                    "order_index": local_idx
                }
                    
    return corpus_ids, corpus_texts, lookup

@lru_cache(maxsize=64)
def _get_bm25_index(ticker: Optional[str] = None, year: Optional[int] = None, report_scope: Optional[str] = None):
    corpus_ids, corpus_texts, lookup = _load_bm25_corpus(ticker=ticker, year=year, report_scope=report_scope)
    if not corpus_ids or not corpus_texts:
            print("[BM25] Không tìm thấy tài liệu phù hợp trong MongoDB")
            return None, [], {}
    corpus_tokens = bm25s.tokenize(corpus_texts, stopwords=list(VI_STOPWORDS))
    retriever = bm25s.BM25()
    retriever.index(corpus_tokens)
    
    return retriever, corpus_ids, lookup

def refresh_bm25_index() -> None:
    """Gọi hàm này sau khi ingest/xoá tài liệu (vd cuối
    app/routers/documents.py) -- lru_cache ở trên không tự biết Mongo vừa
    đổi nên phải invalidate thủ công, nếu không BM25 sẽ tìm trên corpus cũ.
    Đồng thời invalidate luôn cache get_known_tickers() (mongo_client.py):
    danh sách ticker dùng làm ngữ cảnh cho LLM chuẩn hoá ticker cũng cần
    cập nhật ngay khi có công ty mới được ingest, nếu không LLM sẽ không
    "thấy" được ticker vừa thêm."""
    _get_bm25_index.cache_clear()
    refresh_known_tickers()


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = K_RRF) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, corpus_id in enumerate(ranking, start=1):
            scores[corpus_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: -x[1])

class HybridSearchPipeline:
    def __init__(self):
        self.router = QueryRouter()
        self.rewriter = QueryRewriter()
        self.reranker = DocumentReranker()
        self._session_state: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    # ------------------------------------------------------------------
    # Quản lý session để chatbot ghi nhớ công ty đang thảo luận 
    # ------------------------------------------------------------------

    def _get_session_state(self, session_id: Optional[str]) -> Dict[str, Any]:
        if not session_id:
            return {}
        return dict(self._session_state.get(session_id, {}))

    def _update_session_state(self, session_id: Optional[str], current_filter: Dict[str, Any]) -> None:
        """Ghi lại NHỮNG GIÁ TRỊ MỚI thực sự trích xuất được từ câu hỏi
        HIỆN TẠI (không ghi giá trị đã kế thừa từ trước đó -- tránh việc 1
        giá trị kế thừa tự "xác nhận lại" chính nó vô nghĩa)."""
        if not session_id:
            return
        state = self._session_state.setdefault(session_id, {})
        for key, value in current_filter.items():
            if value:
                state[key] = value
        self._session_state.move_to_end(session_id)
        while len(self._session_state) > _MAX_TRACKED_SESSIONS:
            evicted_id, _ = self._session_state.popitem(last=False)
            print(f"[session_state] Vượt quá {_MAX_TRACKED_SESSIONS} session đang theo dõi, "
                  f"loại bỏ session cũ nhất khỏi bộ nhớ: {evicted_id!r}")

    def clear_session_state(self, session_id: str) -> None:
        """Cho phép chủ động xoá ngữ cảnh 1 session (vd người dùng bấm
        'cuộc trò chuyện mới' trên frontend)."""
        self._session_state.pop(session_id, None)

    def get_session_context(self, session_id: Optional[str]) -> Dict[str, Any]:
        """Đọc lại {ticker, year, report_scope} đã ghi nhận của 1 session."""
        return self._get_session_state(session_id)

    def update_session_context(self, session_id: Optional[str], **fields: Any) -> None:
        """Ghi nhận lại ticker/year/report_scope MỚI của 1 session. Chỉ
        truyền field nào THỰC SỰ trích xuất được ở câu hỏi hiện tại (giá trị
        rỗng/None sẽ bị bỏ qua, không ghi đè mất giá trị cũ -- xem
        _update_session_state())."""
        self._update_session_state(session_id, fields)

    # ------------------------------------------------------------------
    # Xử lý query 

    def process_user_query(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        # 1. Routing câu hỏi
        route = self.router.route(query)
        if route == "chitchat":
            return {"type": "chitchat", "queries": [query], "metadata": {}}
        if route == "term_definition":
            return {"type": "term_definition", "original_query": query}
        if route == "calculation":
            # Không đi qua rewrite/RRF ở đây -- xử lý riêng bởi
            # nhánh app.calculation.calculation_service.
            return {"type": "calculation", "original_query": query, "session_id": session_id}
        
        # 2. Rewrite & Extract Metadata -- ticker/year/report_scope giờ đều
        # do LLM (gpt-4o-mini) trích xuất trong CÙNG 1 lần gọi
        # rewrite_and_extract_metadata() (xem app/retrieval/query_rewriter.py),
        # không còn dò keyword rule-based riêng cho report_scope nữa.
        #
        # MULTI-ENTITY: rewriter giờ trả về "entities" = list[{ticker, year,
        # report_scope}] -- câu hỏi so sánh/tổng hợp NHIỀU công ty (vd "So
        # sánh doanh thu FPT năm 2023 với MWG năm 2024") sẽ có >= 2 phần tử,
        # MỖI công ty giữ đúng ticker/năm/report_scope riêng của nó. Câu
        # hỏi thông thường (1 công ty) vẫn chỉ có 1 phần tử -- xử lý giống
        # hệt bản cũ ở nhánh đó.
        extracted_info = self.rewriter.rewrite_and_extract_metadata(query)
        entities = extracted_info.get("entities") or [{
            "ticker": extracted_info.get("ticker"),
            "year": extracted_info.get("year"),
            "report_scope": extracted_info.get("report_scope"),
        }]

        session_state = self._get_session_state(session_id)

        if len(entities) <= 1:
            # 3a. CHỈ 1 công ty -- áp dụng kế thừa từ session như bản cũ
            # cho field nào câu hỏi HIỆN TẠI không đề cập (follow-up
            # question, vd "Còn năm 2023 thì sao?" sau khi đã hỏi về FPT).
            current_filter = dict(entities[0]) if entities else {"ticker": None, "year": None, "report_scope": None}
            entity = dict(current_filter)
            inherited_keys = []
            for key in ("ticker", "year", "report_scope"):
                if not entity.get(key) and session_state.get(key):
                    entity[key] = session_state[key]
                    inherited_keys.append(key)
            if inherited_keys:
                print(f"[process_user_query] session={session_id!r} kế thừa {inherited_keys} "
                      f"từ lượt hỏi trước: { {k: entity[k] for k in inherited_keys} }")
            entities = [entity]
            self._update_session_state(session_id, current_filter)
        else:
            # 3b. NHIỀU công ty -- mỗi công ty đã được LLM nêu rõ trong câu
            # hỏi hiện tại, KHÔNG áp dụng kế thừa session cho từng entity
            # (tránh trộn nhầm ticker/năm cũ của session vào 1 trong các
            # công ty vừa hỏi). Vẫn cập nhật session bằng công ty CUỐI CÙNG
            # được nhắc tới, để câu hỏi tiếp theo (vd "còn ROE thì sao?")
            # có 1 công ty "đang nói tới" mặc định hợp lý.
            print(f"[process_user_query] Câu hỏi nhiều công ty ({len(entities)}): "
                  f"{[e.get('ticker') for e in entities]}")
            self._update_session_state(session_id, entities[-1])

        return {
            "type": "financial_search",
            "original_query": query,
            "search_queries": extracted_info.get("rewritten_queries", [query]),
            "entities": entities,
            # tương thích ngược: field số ít = entity đầu tiên, cho bất kỳ
            # chỗ nào (nếu có) còn đọc "metadata_filter" trực tiếp.
            "metadata_filter": entities[0],
        }
    
    def bm25_search(self, query: str, k: int = BM25_TOP_K, metadata_filter: Optional[dict] = None):
        mongo_query, _ = ticker_year_filter(metadata_filter)
        ticker, year = mongo_query.get("ticker"), mongo_query.get("year")
        report_scope = extract_report_scope(metadata_filter)

        retriever, corpus_ids, lookup = _get_bm25_index(ticker=ticker, year=year, report_scope=report_scope)        
        if not retriever or not corpus_ids:
            return []
        
        print(f"[BM25] Đang phân tách chuỗi (split strings) và tokenize query người dùng: "
              f"(filter ticker={ticker}, year={year}, report_scope={report_scope})")
        query_tokens = bm25s.tokenize([query], stopwords=list(VI_STOPWORDS))

        print("[BM25] Đang truy vấn BM25 index")
        indices, scores = retriever.retrieve(query_tokens, k=min(k, len(corpus_ids)))
        hits = [(_normalize_cid(corpus_ids[idx]), float(score)) for idx, score in zip(indices[0], scores[0])]

        bm25_mongo_indices = [lookup[cid]["order_index"] for cid, _ in hits[:k] if cid in lookup]
        print(f"[BM25] Đã lọc ra {len(hits)} chunk trong MongoDB có các index: {bm25_mongo_indices}")
        return hits[:k]

    def dense_search(
        self,
        query: str,
        k: int = DENSE_TOP_K,
        metadata_filter: Optional[dict] = None,
    ) -> List[tuple[str, float, dict]]:
        _, qdrant_filter = ticker_year_filter(metadata_filter)
        report_scope = extract_report_scope(metadata_filter)

        print("[Dense] Đang tạo Vector Embedding cho câu hỏi")
        vector = embed_query(query, return_sparse=False)["dense"]

        hits = search_similar_blocks(vector, limit=k, filter_conditions=qdrant_filter)
        if report_scope:
            before = len(hits)
            hits = [
                h for h in hits
                if _source_file_matches_scope((h.payload or {}).get("source_file"), report_scope)
            ][:k]
            print(f"[Dense] Lọc report_scope={report_scope!r} theo source_file: {before} -> {len(hits)} điểm.")
        result = [(_normalize_cid(hit.id), float(hit.score), hit.payload or {}) for hit in hits]

        dense_order_indices = [
            hit.payload.get("order_index")
            for hit in hits if hit.payload
        ]
        print(f"[Dense] Đã lọc ra top {len(result)} vector point tốt nhất trên Qdrant với order_index: {dense_order_indices}")
        return result
         
    def RRF_fuse(
        self,
        search_queries: List[str],
        metadata_filter: Optional[dict] = None,
        stage: str = STAGE_FULL,
    ) -> tuple[List[str], Dict[str, dict]]:
        """Chạy BM25 + Dense cho từng rewritten query, gộp toàn bộ ranking
        (2 x số rewritten_queries ranking) bằng 1 lần RRF duy nhất -- RRF hỗ
        trợ fuse nhiều ranking cùng lúc nên không cần fuse riêng từng cặp.

        stage: dùng để ĐÁNH GIÁ RIÊNG từng thành phần của pipeline (ablation),
        xem VALID_STAGES ở đầu file:
          - "full" / "rrf": chạy CẢ BM25 và Dense, fuse bằng RRF như bản gốc
            (khác nhau ở chỗ "rrf" thì retrieve()/_retrieve_for_entity() sẽ
            BỎ QUA bước rerank ngay sau khi gọi hàm này).
          - "reranker": chạy CẢ BM25 và Dense nhưng KHÔNG fuse bằng RRF --
            gộp thẳng (union, giữ nguyên thứ tự phát hiện) candidate id của
            2 nhánh để đưa nguyên vào Reranker chấm điểm lại.
          - "bm25": CHỈ chạy BM25 (tắt hẳn Dense) -- xếp hạng theo điểm BM25
            gốc, không qua RRF.
          - "dense": CHỈ chạy Dense (tắt hẳn BM25) -- xếp hạng theo điểm
            cosine similarity gốc, không qua RRF.

        Trả về:
          fused_ids     : list[chunk_id] top FUSION_TOP_K, đã sắp xếp theo
                          tiêu chí phù hợp với stage (rrf_score / điểm gốc /
                          union không xếp hạng)
          content_lookup: dict[chunk_id -> {"content", "parent_id"}]
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"stage không hợp lệ: {stage!r} (chọn 1 trong {VALID_STAGES})")

        run_bm25 = stage != STAGE_DENSE
        run_dense = stage != STAGE_BM25

        mongo_query, _ = ticker_year_filter(metadata_filter)
        ticker, year = mongo_query.get("ticker"), mongo_query.get("year")
        report_scope = extract_report_scope(metadata_filter)

        corpus_ids: List[str] = []
        bm25_lookup: Dict[str, dict] = {}
        if run_bm25:
            _, corpus_ids, bm25_lookup = _get_bm25_index(ticker=ticker, year=year, report_scope=report_scope)

        rankings: List[List[str]] = []
        content_lookup: Dict[str, dict] = {}
        qdrant_cid_to_order_idx: Dict[str, Any] = {}  # Lưu order_index của các chunk xuất hiện từ Qdrant
        raw_scores: Dict[str, float] = {}             # dùng cho stage="bm25"/"dense" (xếp hạng theo điểm gốc, KHÔNG RRF)
        candidate_order: "OrderedDict[str, None]" = OrderedDict()  # dùng cho stage="reranker" (union, giữ thứ tự phát hiện)

        for q in search_queries:
            q_expanded = expand_query(q)          # xử lí từ viết, thuật ngữ tài chính

            if run_bm25:
                bm25_hits = self.bm25_search(q_expanded, metadata_filter=metadata_filter)
                rankings.append([_normalize_cid(cid) for cid, _ in bm25_hits])
                for cid, score in bm25_hits:
                    norm_cid = _normalize_cid(cid)
                    if norm_cid not in content_lookup and norm_cid in bm25_lookup:
                        content_lookup[norm_cid] = bm25_lookup[norm_cid]
                    raw_scores[norm_cid] = max(raw_scores.get(norm_cid, float("-inf")), score)
                    candidate_order.setdefault(norm_cid, None)

            if run_dense:
                dense_hits = self.dense_search(q_expanded, metadata_filter=metadata_filter)
                rankings.append([_normalize_cid(cid) for cid, _, _ in dense_hits])
                for cid, score, payload in dense_hits:
                    norm_cid = _normalize_cid(cid)
                    if norm_cid not in content_lookup:
                        text_content = payload.get("text") or payload.get("content", "")
                        content_lookup[norm_cid] = {
                            "content": text_content,
                            "parent_id": _normalize_cid(payload.get("parent_id")) if payload.get("parent_id") else None,
                            "order_index": payload.get("order_index")
                        }
                    if norm_cid not in qdrant_cid_to_order_idx:
                        qdrant_cid_to_order_idx[norm_cid] = payload.get("order_index")
                    raw_scores[norm_cid] = max(raw_scores.get(norm_cid, float("-inf")), score)
                    candidate_order.setdefault(norm_cid, None)

        # -------------------- stage="bm25" hoặc "dense" --------------------
        # Đánh giá riêng 1 thành phần: KHÔNG qua RRF, xếp hạng thẳng theo
        # điểm gốc (BM25 score / cosine similarity) đã gộp qua các rewritten
        # query (lấy điểm cao nhất nếu 1 chunk khớp nhiều query).
        if stage in (STAGE_BM25, STAGE_DENSE):
            ranked_ids = sorted(raw_scores.keys(), key=lambda cid: -raw_scores[cid])
            result_ids = ranked_ids[:FUSION_TOP_K]
            print(f"[{stage.upper()}-only] Đã TẮT {'Dense' if stage == STAGE_BM25 else 'BM25'}, RRF và Reranker. "
                  f"Xếp hạng trực tiếp theo điểm gốc. Top {len(result_ids)} chunk ID: {result_ids}")
            return result_ids, content_lookup

        # -------------------- stage="reranker" --------------------
        # Chạy đủ BM25 + Dense nhưng BỎ QUA RRF: gộp thẳng (union) candidate
        # của 2 nhánh, không xếp hạng lại ở đây -- để nguyên cho Reranker
        # (cross-encoder) tự chấm điểm.
        if stage == STAGE_RERANKER:
            result_ids = list(candidate_order.keys())[:FUSION_TOP_K]
            print(f"[RERANKER-only] Đã TẮT RRF. Gộp thẳng (union) {len(result_ids)} candidate "
                  f"từ BM25 + Dense để đưa vào Reranker: {result_ids}")
            return result_ids, content_lookup

        # -------------------- stage="full" hoặc "rrf" --------------------
        # Logic gốc, không đổi hành vi: RRF fuse toàn bộ ranking BM25 + Dense.
        fused = reciprocal_rank_fusion(rankings)
        RRF_ids = [_normalize_cid(cid) for cid, _ in fused][:FUSION_TOP_K]

        cid_to_mongo_idx = {cid: idx for idx, cid in enumerate(corpus_ids)} if corpus_ids else {}
        RRF_mongo_indices = [cid_to_mongo_idx[cid] for cid in RRF_ids if cid in cid_to_mongo_idx]

        RRF_qdrant_indices = [qdrant_cid_to_order_idx[cid] for cid in RRF_ids if cid in qdrant_cid_to_order_idx and qdrant_cid_to_order_idx[cid] is not None]
        
        print(f"[RRF_Fusion] Đã chạy RRF trên quy trình BM25 song song Dense Vector (filter ticker={ticker}, year={year}). Top {len(RRF_ids)} chunk ID chọn ra: {RRF_ids}")
        if RRF_mongo_indices:
            print(f"[RRF_Fusion] Vị trí index tương ứng của top {len(RRF_mongo_indices)} chunk trong MongoDB: {RRF_mongo_indices}")
        if RRF_qdrant_indices:
            print(f"[RRF_Fusion] Vị trí order_index tương ứng của top {len(RRF_qdrant_indices)} chunk từ Qdrant: {RRF_qdrant_indices}")
        return RRF_ids, content_lookup

    def rerank(
        self,
        original_query: str,
        candidate_ids: List[str],
        content_lookup: Dict[str, dict],
        top_k: int = TOP_K,
    ) -> List[str]:
        valid_ids = [_normalize_cid(cid) for cid in candidate_ids if content_lookup.get(cid, {}).get("content")]
        if not valid_ids:
            return []
        documents = [content_lookup[cid]["content"] for cid in valid_ids]
        print(f"[Reranker] Đang tính điểm Rerank cho {len(documents)} tài liệu")
        ranked_indices = self.reranker.rerank(original_query, documents, top_k=top_k)
        result = [valid_ids[i] for i in ranked_indices]

        print(f"[Reranker] Top {len(result)} chunk ID cuối cùng được giữ lại: {result}")
        return result

    def _retrieve_for_entity(
        self,
        user_query: str,
        search_queries: List[str],
        entity: Dict[str, Any],
        top_k: int,
        stage: str = STAGE_FULL,
    ) -> tuple[List[str], List[Dict[str, Any]]]:
        """Chạy TRỌN 1 lượt Hybrid Search (RRF -> fallback bỏ year -> rerank
        -> mở rộng parent) cho ĐÚNG 1 entity (ticker/year/report_scope).

        Tách riêng thành hàm này để retrieve() gọi LẶP LẠI cho câu hỏi so
        sánh nhiều công ty (mỗi công ty 1 lượt retrieval độc lập, không bị
        trộn filter với nhau) -- xem retrieve() bên dưới. Đây chính là toàn
        bộ logic cũ của retrieve() (bản chỉ hỗ trợ 1 công ty), giữ nguyên
        không đổi hành vi cho trường hợp 1 công ty.

        stage: xem VALID_STAGES / RRF_fuse() -- "full" (mặc định) giữ nguyên
        hành vi gốc (BM25+Dense+RRF+Reranker). Các stage ablation khác
        ("bm25", "dense", "rrf") sẽ BỎ QUA bước rerank() ngay bên dưới và
        cắt thẳng top_k từ danh sách RRF_fuse() đã trả về.
        """
        metadata_filter = entity
        mongo_query, qdrant_filter = ticker_year_filter(metadata_filter)
        report_scope = extract_report_scope(metadata_filter)

        print(f"[retrieve] entity trích được: {metadata_filter}")
        print(f"[retrieve] Áp dụng lọc công ty={metadata_filter.get('ticker')}, năm={metadata_filter.get('year')}, report_scope={report_scope}"
              f"cho cả BM25 (mongo_query={mongo_query}) và Dense Vector "
              f"(qdrant_filter={qdrant_filter.model_dump(exclude_none=True) if qdrant_filter else None})")
        RRF_ids, content_lookup = self.RRF_fuse(search_queries, metadata_filter=metadata_filter, stage=stage)
        if not RRF_ids and metadata_filter.get("ticker"):
            print("[*] Lần 1 không thấy data. Tiến hành Fallback: Bỏ lọc Year, BẮT BUỘC giữ Ticker...")
            fallback_filter = {"ticker": metadata_filter["ticker"], "report_scope": metadata_filter.get("report_scope")}
            RRF_ids, content_lookup = self.RRF_fuse(search_queries, metadata_filter=fallback_filter, stage=stage)

        if stage in (STAGE_FULL, STAGE_RERANKER):
            top_chunk_ids = self.rerank(user_query, RRF_ids, content_lookup, top_k=top_k)
        else:
            # stage="bm25"/"dense"/"rrf": KHÔNG qua Reranker -- cắt thẳng
            # top_k từ danh sách RRF_fuse() đã xếp hạng sẵn (theo điểm gốc
            # hoặc theo RRF score, tuỳ stage).
            top_chunk_ids = [
                _normalize_cid(cid) for cid in RRF_ids
                if content_lookup.get(_normalize_cid(cid), {}).get("content")
            ][:top_k]
            print(f"[{stage.upper()}] Đã TẮT Reranker. Cắt thẳng top {len(top_chunk_ids)} chunk ID: {top_chunk_ids}")
 
        seen_parents = set()
        contexts: List[Dict[str, Any]] = []
        for cid in top_chunk_ids:
            norm_cid = _normalize_cid(cid)
            chunk_info = content_lookup.get(norm_cid, {})
            target_parent_id = _normalize_cid(chunk_info.get("parent_id")) or norm_cid            
            if target_parent_id in seen_parents:
                continue
            seen_parents.add(target_parent_id)

            # Truy vấn Parent Document từ Mongo
            parent_doc = get_parent_chunk(target_parent_id)
            if parent_doc:
                text_content = parent_doc.get("text") or parent_doc.get("content", "")
                citation_info = {
                    "chunk_id": target_parent_id,
                    "matched_chunk_id": norm_cid,
                    # LƯU Ý: document_id của FILE NGUỒN, KHÔNG PHẢI chunk_id.
                    "doc_id": parent_doc.get("doc_id"),
                    "source_file": parent_doc.get("source_file"),
                    "page_start": parent_doc.get("page_start"),
                    "page_end": parent_doc.get("page_end"),
                    "section": parent_doc.get("section") or parent_doc.get("heading"),
                    "ticker": parent_doc.get("ticker"),
                    "year": parent_doc.get("year"),
                }
            else:
                # Fallback lấy trực tiếp nội dung từ content_lookup nếu Mongo chưa bóc tách xong
                text_content = chunk_info.get("content", "")
                citation_info = {
                    "chunk_id": norm_cid,
                    "matched_chunk_id": norm_cid,
                    "doc_id": None,
                    "source_file": "Báo cáo tài chính",
                    "page_start": None,
                    "page_end": None,
                    "section": None,
                }

            if text_content and not any(c["content"] == text_content for c in contexts):
                contexts.append({
                    "content": text_content,
                    "citation": citation_info
                })

        return top_chunk_ids, contexts

    def retrieve(
        self, user_query: str, 
        top_k: int = TOP_K,
        session_id: Optional[str] = None,
        prep: Optional[Dict[str, Any]] = None,
        stage: str = STAGE_FULL,
    ) -> Dict[str, Any]:
        """Entry point chính -- gọi hàm NÀY từ RAGController.execute_search()
        thay vì tự làm dense-search riêng như code cũ.

        MULTI-ENTITY: nếu prep["entities"] có NHIỀU HƠN 1 công ty (câu hỏi
        so sánh/tổng hợp, xem process_user_query()), retrieval được chạy
        RIÊNG cho từng công ty (_retrieve_for_entity()) rồi GỘP LẠI, chia
        đều ngân sách top_k cho mỗi công ty -- tránh tình trạng công ty có
        nhiều chunk khớp hơn "lấn át" hoàn toàn công ty còn lại trong 1 lần
        RRF/rerank dùng chung. Câu hỏi thông thường (1 công ty) chạy y hệt
        bản cũ, không đổi hành vi.

        stage: BẬT/TẮT từng thành phần của Hybrid Search để đánh giá riêng
        (ablation study) -- xem VALID_STAGES ở đầu file / RRF_fuse():
          - "full"     (mặc định, KHÔNG đổi hành vi cũ): BM25 + Dense + RRF
                       + Reranker.
          - "bm25"     : CHỈ BM25 -- tắt Dense, RRF, Reranker.
          - "dense"    : CHỈ Dense -- tắt BM25, RRF, Reranker.
          - "rrf"      : BM25 + Dense + RRF -- tắt Reranker.
          - "reranker" : BM25 + Dense, BỎ RRF -- gộp thẳng candidate rồi
                       đưa nguyên vào Reranker.
 
        Trả về:
          {
            "is_chitchat": bool,
            "chunk_ids": list[str],   # top_k chunk_id sau rerank (gộp)
            "context": list[dict],    # nội dung parent tương ứng, đã dedup
            "entities": list[dict],   # các công ty/năm/scope đã dùng để lọc
            "stage": str,             # stage thực sự đã dùng để retrieve
          }
        """
        if stage not in VALID_STAGES:
            raise ValueError(f"stage không hợp lệ: {stage!r} (chọn 1 trong {VALID_STAGES})")

        prep = prep if prep is not None else self.process_user_query(user_query, session_id=session_id)
        if prep["type"] == "chitchat":
            return {"is_chitchat": True, "chunk_ids": [], "context": []}
        if prep["type"] == "term_definition":
            return {"is_chitchat": False, "is_definition": True, "chunk_ids": [], "context": []}
        if prep["type"] == "calculation":
            return {"is_chitchat": False, "is_definition": False, "is_calculation": True, "chunk_ids": [], "context": []}

        entities = prep.get("entities") or [prep.get("metadata_filter", {})]
        search_queries = prep["search_queries"]

        if len(entities) <= 1:
            entity = entities[0] if entities else {}
            top_chunk_ids, contexts = self._retrieve_for_entity(user_query, search_queries, entity, top_k, stage=stage)
        else:
            per_entity_k = max(3, top_k // len(entities))
            print(f"[retrieve] Câu hỏi nhiều công ty ({len(entities)}) -- chạy retrieval riêng cho từng "
                  f"công ty, mỗi công ty tối đa {per_entity_k} chunk (stage={stage}).")
            top_chunk_ids = []
            contexts = []
            seen_parents_global: set = set()
            for entity in entities:
                ent_chunk_ids, ent_contexts = self._retrieve_for_entity(user_query, search_queries, entity, per_entity_k, stage=stage)
                top_chunk_ids.extend(ent_chunk_ids)
                for ctx in ent_contexts:
                    parent_id = ctx["citation"].get("chunk_id")
                    if parent_id in seen_parents_global:
                        continue
                    seen_parents_global.add(parent_id)
                    # Đánh dấu context này match theo entity nào -- hữu ích
                    # cho generation/citation khi hiển thị câu trả lời so
                    # sánh nhiều công ty (biết đoạn nào thuộc công ty nào).
                    ctx["citation"]["matched_entity"] = {
                        "ticker": entity.get("ticker"),
                        "year": entity.get("year"),
                        "report_scope": entity.get("report_scope"),
                    }
                    contexts.append(ctx)

        return {
            "is_chitchat": False,
            "is_definition": False,
            "chunk_ids": top_chunk_ids,
            "context": contexts,
            "entities": entities,
            "stage": stage,
        }