"""
app/calculation/calculation_service.py

Quy trình:
  1. LLM (gpt-4o-mini) đọc câu hỏi -> xác định MỘT HOẶC NHIỀU chỉ số tài
     chính (metric_keys, khớp METRIC_REGISTRY) + ticker/year/quarter/
     report_scope (CalculationIntent). Chỉ số có thể được người dùng gọi
     tên TRỰC TIẾP ("ROE", "biên lợi nhuận"...) hoặc mô tả GIÁN TIẾP qua ý
     nghĩa/ngữ cảnh câu hỏi (vd "khả năng trả nợ ngắn hạn" -> current_ratio)
     -- prompt yêu cầu LLM suy luận ngữ nghĩa, không chỉ khớp từ khoá, và có
     thể trả về NHIỀU metric_key cùng lúc (câu hỏi nêu rõ nhiều chỉ số, hoặc
     câu hỏi tổng quát về "tình hình/sức khỏe tài chính" ngụ ý cần đánh giá
     nhiều khía cạnh). Ticker được đối chiếu với danh sách ticker THẬT đang
     có trong hệ thống (known_tickers_prompt_text(), lấy trực tiếp từ
     Mongo), report_scope cũng do LLM tự suy luận -- xem
     app/retrieval/query_rewriter.py để rõ cơ chế tương tự dùng ở nhánh
     financial_search. Ticker/year/report_scope KHÔNG trích được ở câu hỏi
     hiện tại sẽ được KẾ THỪA từ session_state dùng CHUNG với
     HybridSearchPipeline (xem extract_intent()).
  2. Với TỪNG chỉ số trong metric_keys, với TỪNG input field công thức cần,
     tái sử dụng trực tiếp HybridSearchPipeline.RRF_fuse() + .rerank() (lọc
     theo ticker/năm/report_scope) để lấy context liên quan. Field nào được
     NHIỀU chỉ số dùng chung (vd 'revenue' xuất hiện trong cả gross_margin
     lẫn net_margin) chỉ retrieve + trích xuất MỘT LẦN DUY NHẤT trong cùng
     1 câu hỏi (xem operand_cache trong calculate()).
  3. Một LLM THỨ HAI đọc context, trích xuất ĐÚNG 1 con số cho field đó
     (không suy diễn/làm tròn).
  4. compute_metric() (code Python thuần) thực hiện phép TÍNH -- không để
     LLM tự nhẩm, đúng nguyên tắc README.
  5. calculate() ghép kết quả của TỪNG chỉ số thành 1 câu trả lời duy nhất
     (giữ NGUYÊN định dạng cũ khi câu hỏi chỉ có 1 chỉ số, chỉ thêm khối
     phân tách khi có từ 2 chỉ số trở lên).

ABLATION TOGGLE (đồng bộ với app/retrieval/hybrid_search.py):
  calculate() và _fetch_operand() nhận thêm tham số `stage` (mặc định
  STAGE_FULL, KHÔNG đổi hành vi cũ) để BẬT/TẮT từng thành phần của Hybrid
  Search khi tra cứu operand -- dùng cho đánh giá riêng từng bước, xem
  VALID_STAGES trong app/retrieval/hybrid_search.py:
    - "full"     : BM25 + Dense + RRF + Reranker (mặc định).
    - "bm25"     : CHỈ BM25 -- tắt Dense, RRF, Reranker.
    - "dense"    : CHỈ Dense -- tắt BM25, RRF, Reranker.
    - "rrf"      : BM25 + Dense + RRF -- tắt Reranker.
    - "reranker" : BM25 + Dense, BỎ RRF -- gộp thẳng candidate rồi vào
                   Reranker.
  `stage` được truyền thẳng vào RRF_fuse(..., stage=stage); bước rerank()
  chỉ được gọi khi stage in ("full", "reranker") -- y hệt logic đã áp dụng
  trong HybridSearchPipeline._retrieve_for_entity().
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

from app.config import settings
from app.calculation.metrics import METRIC_FORMULAS, compute_metric_from_operands, FormulaError, get_metric_spec
from app.models.calculation_schema import CalculationIntent, CalculationResponse, OperandDetail, MetricFormulaSpec
from app.generation.citation import format_citation_label, clean_source_filename
from app.services.mongo_client import get_parent_chunk, known_tickers_prompt_text
from app.calculation.calculation_formatter import format_calculation_answer
from app.retrieval.hybrid_search import STAGE_FULL, STAGE_RERANKER, VALID_STAGES


FIELD_LABELS: dict[str, str] = {
    "gross_profit": "Lợi nhuận gộp",
    "revenue": "Doanh thu thuần",
    "net_income": "Lợi nhuận sau thuế",
    "current_assets": "Tài sản ngắn hạn",
    "current_liabilities": "Nợ ngắn hạn",
    "total_liabilities": "Tổng nợ phải trả",
    "total_equity": "Vốn chủ sở hữu",
    "current_value": "Giá trị kỳ hiện tại",
    "prior_year_value": "Giá trị cùng kỳ năm trước",
    "prior_quarter_value": "Giá trị quý trước",
    "beginning_value": "Giá trị đầu kỳ",
    "ending_value": "Giá trị cuối kỳ",
    "num_years": "Số năm",
    "shareholders_equity_t": "Vốn chủ sở hữu kỳ hiện tại",
    "shareholders_equity_t_minus_1": "Vốn chủ sở hữu kỳ trước",
    "total_assets_t": "Tổng tài sản kỳ hiện tại",
    "total_assets_t_minus_1": "Tổng tài sản kỳ trước",
    "average_shareholders_equity": "Vốn chủ sở hữu bình quân",
    "average_total_assets": "Tổng tài sản bình quân",
}

MAX_METRICS_PER_QUERY = 5

def _normalize_cid(cid: Any) -> str:
    """Chuẩn hoá chunk_id/parent_id về dạng hex thô không gạch ngang (đồng bộ với MongoDB)."""
    if not cid:
        return ""
    return str(cid).replace("-", "").strip()

def _clean_filename(source_file: str) -> str:
    """Chỉ lấy tên file (không path), khớp docstring của OperandDetail.source."""
    return os.path.basename(str(source_file or "Báo cáo tài chính"))

class CalculationService:
    def __init__(self, search_pipeline):
        """search_pipeline: instance HybridSearchPipeline đã khởi tạo sẵn
        trong app/main.py (dùng lại luôn RRF_fuse/rerank/_qdrant_filter,
        không tạo pipeline riêng). Đồng thời dùng CHUNG session-context của
        chính pipeline này (get_session_context/update_session_context) để
        ticker/year/report_scope nhất quán giữa nhánh financial_search và
        nhánh calculation cho cùng 1 session -- xem extract_intent()."""
        self.search_pipeline = search_pipeline

        self.intent_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini",
                                      openai_api_key=settings.OPENAI_API_KEY)
        # self.extract_llm = ChatGroq(temperature=0, model_name="openai/gpt-oss-120b",
        #                              groq_api_key=settings.GROQ_API_KEY)
        self.extract_llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini",
                                      openai_api_key=settings.OPENAI_API_KEY)

        self.intent_prompt = PromptTemplate(
            template="""Bạn là chuyên gia phân tích tài chính. Nhiệm vụ: đọc câu hỏi của người
dùng và xác định MỘT HOẶC NHIỀU chỉ số tài chính (metric) người dùng muốn
tính, cùng công ty (ticker)/năm/quý/report_scope liên quan.

QUAN TRỌNG -- suy luận chỉ số GIÁN TIẾP: người dùng có thể KHÔNG gọi tên chỉ
số trực tiếp (không nói "ROE", "biên lợi nhuận"...) mà chỉ mô tả Ý NGHĨA của
chỉ số đó qua ngữ cảnh câu hỏi. Hãy suy luận theo Ý NGHĨA, không chỉ khớp từ
khoá. Một vài ví dụ suy luận gián tiếp (mỗi ví dụ chỉ minh hoạ, không phải
danh sách đầy đủ):
  - "Khả năng trả nợ ngắn hạn của X có tốt không?", "X có đủ tài sản ngắn hạn
    để trả nợ không?" -> ["current_ratio"]
  - "X có đang dùng nhiều nợ vay để tài trợ hoạt động không?", "mức độ đòn
    bẩy tài chính của X" -> ["debt_to_equity"]
  - "Mỗi đồng vốn cổ đông bỏ ra thì X sinh lời bao nhiêu?", "hiệu quả sinh
    lời trên vốn chủ sở hữu của X" -> ["roe"]
  - "X sử dụng tài sản hiệu quả đến đâu để tạo ra lợi nhuận?" -> ["roa"]
  - "Lợi nhuận của X có tăng trưởng tốt so với cùng kỳ năm ngoái không?" -> ["yoy_growth"]
  - "Tốc độ tăng trưởng bình quân của X qua các năm" -> ["cagr"]

NHIỀU CHỈ SỐ CÙNG LÚC: câu hỏi có thể cần TRẢ VỀ NHIỀU metric_key, ví dụ:
  - Nêu rõ nhiều chỉ số: "Tính cả ROE và ROA của X năm 2024" -> ["roe", "roa"]
  - Câu hỏi TỔNG QUÁT về "tình hình/sức khỏe/năng lực tài chính" của X, không
    chỉ rõ 1 khía cạnh cụ thể -> liệt kê NHIỀU chỉ số liên quan để đánh giá
    toàn diện, ví dụ ["current_ratio", "debt_to_equity", "roe"].

DANH SÁCH CHỈ SỐ HỖ TRỢ (chỉ được chọn trong các key sau, không tự bịa key mới):
{metric_list}

DANH SÁCH MÃ CỔ PHIẾU (TICKER) HIỆN CÓ TRONG HỆ THỐNG (đối chiếu tên công
ty/ngân hàng/biệt danh/tên viết tắt trong câu hỏi với danh sách này để trả
về ĐÚNG ticker viết hoa tương ứng; nếu công ty được hỏi không có trong danh
sách vẫn cứ suy đoán mã ticker hợp lý thay vì trả null, TRỪ khi không xác
định được công ty nào):
{known_tickers}

report_scope xác định như sau:
  - "parent"       nếu hỏi về công ty mẹ / báo cáo riêng lẻ (không hợp nhất công ty con).
  - "consolidated" nếu hỏi về báo cáo hợp nhất / toàn hệ thống / toàn ngân hàng.
  - null           nếu câu hỏi không phân biệt rõ phạm vi báo cáo.

Trả về DUY NHẤT 1 JSON object, không thêm giải thích, đúng cấu trúc sau:
{{
    "metric_keys": ["<1 hoặc nhiều key trong danh sách trên>"] (mảng RỖNG [] nếu không xác định được chỉ số nào),
    "ticker": "<mã cổ phiếu viết hoa, hoặc null>",
    "year": <năm dạng số, hoặc null>,
    "quarter": <quý 1-4 dạng số, hoặc null>,
    "compare_year": <năm dùng để so sánh YoY/CAGR nếu có, hoặc null>,
    "compare_quarter": <quý dùng để so sánh QoQ nếu có, hoặc null>,
    "report_scope": "parent" | "consolidated" | null
}}

Câu hỏi: {query}
JSON Output:""",
            input_variables=["query", "metric_list", "known_tickers"],
        )

    def _metric_list_text(self) -> str:
        return "\n".join(
            f"- {key}: {spec.name_vi} (aliases: {', '.join(spec.aliases)})"
            for key, spec in METRIC_FORMULAS.items()
        )

    @staticmethod
    def _safe_json(raw: str) -> Optional[dict]:
        try:
            start, end = raw.find("{"), raw.rfind("}") + 1
            return json.loads(raw[start:end])
        except Exception:
            return None

    def extract_intent(self, query: str, session_id: Optional[str] = None) -> CalculationIntent:
        chain = self.intent_prompt | self.intent_llm
        raw = chain.invoke({
            "query": query,
            "metric_list": self._metric_list_text(),
            "known_tickers": known_tickers_prompt_text(),
        }).content.strip()
        data = self._safe_json(raw) or {}

        # metric_keys: LLM giờ có thể trả về NHIỀU chỉ số cùng lúc (câu hỏi
        # nêu rõ nhiều chỉ số, hoặc suy luận gián tiếp ra nhiều khía cạnh
        # liên quan) -- validate lại từng key so với METRIC_FORMULAS (LLM
        # đôi khi vẫn bịa key không có trong registry dù đã được yêu cầu
        # không làm vậy), bỏ trùng lặp, và GIỚI HẠN số lượng để tránh 1 câu
        # hỏi mơ hồ kéo theo quá nhiều phép tính cùng lúc.
        raw_metric_keys = data.get("metric_keys")
        if not raw_metric_keys:
            # Tương thích ngược: LLM (hoặc bản prompt cũ hơn) lỡ trả field
            # "metric_key" đơn thay vì "metric_keys" dạng mảng.
            single = data.get("metric_key")
            raw_metric_keys = [single] if single else []
        if isinstance(raw_metric_keys, str):
            raw_metric_keys = [raw_metric_keys]

        metric_keys: list[str] = []
        for mk in raw_metric_keys:
            if not mk or mk in metric_keys:
                continue
            if mk not in METRIC_FORMULAS:
                print(f"[Calculation] Bỏ qua metric_key không có trong registry do LLM trả về: {mk!r}")
                continue
            metric_keys.append(mk)
            if len(metric_keys) >= MAX_METRICS_PER_QUERY:
                print(f"[Calculation] Câu hỏi ngụ ý quá nhiều chỉ số cùng lúc, chỉ giữ lại "
                      f"{MAX_METRICS_PER_QUERY} chỉ số đầu tiên: {metric_keys}")
                break

        # ticker/report_scope giờ đều do CHÍNH gpt-4o-mini trích xuất ngay
        # trong intent_prompt ở trên (đối chiếu known_tickers_prompt_text()
        # thay TICKER_MAPPING viết tay, và tự suy luận report_scope thay vì
        # dò keyword rule-based/detect_report_scope() cũ) -- dùng chung ý
        # tưởng chuẩn hoá với QueryRewriter/HybridSearchPipeline, chỉ khác
        # là nhánh calculation gọi LLM riêng để gộp chung với metric_keys.
        raw_ticker = data.get("ticker") or None
        if raw_ticker:
            raw_ticker = str(raw_ticker).strip().upper()
        raw_year = data.get("year")
        raw_report_scope = data.get("report_scope")
        if raw_report_scope not in ("parent", "consolidated"):
            raw_report_scope = None

        ticker, year, report_scope = raw_ticker, raw_year, raw_report_scope
        inherited_keys: list[str] = []

        # ------------------------------------------------------------
        # Kế thừa ticker/year/report_scope từ session nếu câu hỏi hiện tại
        # không đề cập -- đọc từ CHÍNH session-context của HybridSearchPipeline
        # (đã được nhánh financial_search cập nhật, nếu trước đó user từng
        # hỏi qua nhánh đó), để 2 nhánh dùng chung 1 ngữ cảnh hội thoại.
        # ------------------------------------------------------------
        session_context: dict = {}
        if self.search_pipeline is not None and hasattr(self.search_pipeline, "get_session_context"):
            session_context = self.search_pipeline.get_session_context(session_id)

        if not ticker and session_context.get("ticker"):
            ticker = session_context["ticker"]
            inherited_keys.append("ticker")
        if not year and session_context.get("year"):
            year = session_context["year"]
            inherited_keys.append("year")
        if not report_scope and session_context.get("report_scope"):
            report_scope = session_context["report_scope"]
            inherited_keys.append("report_scope")

        if inherited_keys:
            print(f"[Calculation] session={session_id!r} kế thừa {inherited_keys} từ lượt hỏi trước: "
                  f"ticker={ticker!r}, year={year!r}, report_scope={report_scope!r}")

        intent = CalculationIntent(
            metric_key=metric_keys[0] if metric_keys else None,
            metric_keys=metric_keys,
            ticker=ticker,
            year=year,
            quarter=data.get("quarter"),
            compare_year=data.get("compare_year"),
            compare_quarter=data.get("compare_quarter"),
            report_scope=report_scope,
            raw_query=query,
        )
        print(f"[Calculation] Intent trích được: metrics={intent.metric_keys}, ticker={intent.ticker}, "
              f"year={intent.year}, quarter={intent.quarter}, report_scope={intent.report_scope}")

        # Cập nhật lại session_state bằng CHÍNH giá trị MỚI trích xuất được
        # từ câu hỏi HIỆN TẠI (raw_ticker/raw_year/raw_report_scope gốc,
        # KHÔNG phải giá trị đã merge kế thừa) -- nhất quán với cách
        # HybridSearchPipeline.process_user_query() cập nhật session, tránh
        # 1 giá trị kế thừa tự "xác nhận lại" chính nó vô nghĩa.
        if self.search_pipeline is not None and hasattr(self.search_pipeline, "update_session_context"):
            self.search_pipeline.update_session_context(
                session_id,
                ticker=raw_ticker,
                year=raw_year,
                report_scope=raw_report_scope,
            )

        return intent

    def _fetch_operand(
        self,
        field: str,
        intent: CalculationIntent,
        period_label: str,
        stage: str = STAGE_FULL,
    ) -> tuple[Optional[OperandDetail], list[dict]]:
        field_label = FIELD_LABELS.get(field, field)
        search_query = f"{field_label} {period_label}".strip()

        metadata_filter = {
            "ticker": intent.ticker,
            "year": intent.year,
            "report_scope": intent.report_scope,
        }
        print(f"[Calculation] Đang tra cứu '{field_label}' | filter ticker={intent.ticker}, "
              f"year={intent.year}, report_scope={intent.report_scope}, stage={stage}")

        RRF_ids, content_lookup = self.search_pipeline.RRF_fuse(
            [search_query], metadata_filter=metadata_filter, stage=stage
        )
        if stage in (STAGE_FULL, STAGE_RERANKER):
            top_ids = self.search_pipeline.rerank(search_query, RRF_ids, content_lookup, top_k=5)
        else:
            # stage="bm25"/"dense"/"rrf": TẮT Reranker -- cắt thẳng top 5 từ
            # danh sách RRF_fuse() đã trả về (đã xếp hạng sẵn theo điểm gốc
            # hoặc theo RRF score, tuỳ stage), y hệt cách
            # HybridSearchPipeline._retrieve_for_entity() xử lý.
            top_ids = [
                _normalize_cid(cid) for cid in RRF_ids
                if content_lookup.get(_normalize_cid(cid), {}).get("content")
            ][:5]
            print(f"[Calculation][{stage.upper()}] Đã TẮT Reranker cho '{field_label}'. "
                  f"Cắt thẳng top {len(top_ids)} chunk ID: {top_ids}")
        print(f"[Calculation] '{field_label}': còn lại {len(RRF_ids)} chunk sau RRF/candidate (stage={stage}), giữ được {len(top_ids)} chunk cuối")

        contexts, seen_parents = [], set()
        for cid in top_ids:
            norm_cid = _normalize_cid(cid)
            info = content_lookup.get(norm_cid, {})
            parent_id = _normalize_cid(info.get("parent_id")) or norm_cid
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
            parent_doc = get_parent_chunk(parent_id)
            if parent_doc:
                text_content = parent_doc.get("text") or parent_doc.get("content", "")
                citation_info = {
                    "chunk_id": parent_id,               
                    "matched_chunk_id": norm_cid,
                    "doc_id": parent_doc.get("doc_id"),
                    "source_file": parent_doc.get("source_file"),
                    "page_start": parent_doc.get("page_start"),
                    "page_end": parent_doc.get("page_end"),
                    "section": parent_doc.get("section") or parent_doc.get("heading"),
                    "ticker": parent_doc.get("ticker"),
                    "year": parent_doc.get("year"),
                }
            else:
                text_content = info.get("content", "")
                citation_info = {
                    "chunk_id": norm_cid,
                    "matched_chunk_id": norm_cid,
                    "doc_id": None,
                    "source_file": "Báo cáo tài chính",
                }
            if text_content:
                contexts.append({"content": text_content, "citation": citation_info})

        if not contexts:
            print(f"[Calculation] '{field_label}': không tìm thấy ngữ cảnh nào phù hợp, bỏ qua field này.")
            return None, []

        numbered_blocks, citations = [], []
        for i, ctx in enumerate(contexts, start=1):
            citation = dict(ctx.get("citation", {}))
            citation["index"] = i
            citation["label"] = format_citation_label(citation, i)
            citations.append(citation)
            numbered_blocks.append(f"[{i}]\n{ctx['content']}")
        joined_context = "\n\n".join(numbered_blocks)

        extract_prompt = f"""Bạn là chuyên gia trích xuất số liệu tài chính. Dựa vào các đoạn ngữ
    cảnh được đánh số [1], [2]... dưới đây (trích từ báo cáo tài chính), hãy tìm ĐÚNG 1 con số
    cho chỉ tiêu: "{field_label}"{f" ({period_label})" if period_label else ""}.

    Yêu cầu:
    - Chỉ lấy số liệu CÓ TRONG ngữ cảnh, không suy diễn, không tự tính.
    - Đơn vị trả về LUÔN quy đổi ra đồng (VND) -- nếu ngữ cảnh ghi "triệu đồng"/"tỷ đồng",
    nhân tương ứng 1,000,000 / 1,000,000,000.
    - Nếu không tìm thấy số liệu phù hợp, trả "value": null.
    - CHỈ trả về ĐÚNG 1 con số duy nhất cho chỉ tiêu "{field_label}" nêu trên. Việc xác định câu
    hỏi cần tính những chỉ số tài chính (metric) nào đã được xử lý ở bước trước (extract_intent),
    KHÔNG lặp lại việc đó ở đây -- hàm này chỉ tra đúng 1 con số cho 1 field duy nhất.

    --- NGỮ CẢNH ---
    {joined_context}
    ---

    Chỉ trả về DUY NHẤT 1 JSON object, không thêm giải thích, đúng cấu trúc:
    {{"value": <số hoặc null>, "source_index": <n tương ứng [n] đã dùng, hoặc null>, "table": "<tên bảng/mục nếu có, hoặc null>"}}
    JSON Output:"""
        raw = self.extract_llm.invoke(extract_prompt).content.strip()
        data = self._safe_json(raw) or {}

        value = data.get("value")
        if value is None:
            print(f"[Calculation] '{field_label}': LLM trích xuất trả về null (không tìm thấy số liệu trong ngữ cảnh).")
            return None, citations
        try:
            value = float(value)
        except (TypeError, ValueError):
            print(f"[Calculation] '{field_label}': Giá trị trích xuất không hợp lệ ({value!r}), bỏ qua field này.")
            return None, citations

        source_index = data.get("source_index")
        if isinstance(source_index, int) and 1 <= source_index <= len(citations):
            chosen = citations[source_index - 1]
        else:
            chosen = citations[0] if citations else {}
        operand = OperandDetail(
            value=value,
            source=_clean_filename(chosen.get("source_file")),
            page=chosen.get("page_start"),
            table=data.get("table") or chosen.get("section"),
        )
        print(f"[Calculation] '{field_label}': giá trị trích được = {value} (nguồn: {operand.source}, trang {operand.page})")
        return operand, citations


    def calculate(
        self,
        query: str,
        session_id: Optional[str] = None,
        stage: str = STAGE_FULL,
    ) -> CalculationResponse:
        if stage not in VALID_STAGES:
            raise ValueError(f"stage không hợp lệ: {stage!r} (chọn 1 trong {VALID_STAGES})")

        print(f"[Calculation] Bắt đầu xử lý câu hỏi loại calculation: {query!r} (session={session_id!r}, stage={stage})")
        intent = self.extract_intent(query, session_id=session_id)
        intent_json = intent.model_dump()
        intent_json["stage"] = stage

        if not intent.metric_keys:
            print("[Calculation] Không khớp được metric_key nào trong registry.")
            return CalculationResponse(
                answer=("Mình chưa xác định được chính xác chỉ số tài chính bạn muốn tính. "
                        "Bạn có thể hỏi lại rõ hơn, ví dụ: 'Tính ROE của FPT năm 2024' nhé."),
                citations=[],
                intent=intent_json,
            )

        period_label = ""
        if intent.quarter and intent.year:
            period_label = f"quý {intent.quarter} năm {intent.year}"
        elif intent.year:
            period_label = f"năm {intent.year}"
        ticker_part = f"cho {intent.ticker} " if intent.ticker else ""

        # Cache theo field, DÙNG CHUNG cho mọi metric trong câu hỏi hiện tại
        # -- vd gross_margin và net_margin đều cần field 'revenue': nếu tính
        # cả 2 trong 1 câu hỏi ("Tính biên lợi nhuận gộp và biên lợi nhuận
        # ròng của X"), 'revenue' chỉ retrieve + gọi LLM trích số liệu ĐÚNG
        # 1 LẦN thay vì lặp lại cho từng metric.
        operand_cache: dict[str, tuple[Optional[OperandDetail], list[dict]]] = {}

        def fetch_operand_cached(field: str) -> tuple[Optional[OperandDetail], list[dict]]:
            if field not in operand_cache:
                operand_cache[field] = self._fetch_operand(field, intent, period_label, stage=stage)
            return operand_cache[field]

        section_answers: list[str] = []
        failed_notes: list[str] = []
        all_citations: list[dict] = []
        calculations: list = []
        metric_specs: list[dict] = []

        for metric_key in intent.metric_keys:
            spec = get_metric_spec(metric_key)
            if spec is None:
                # Không nên xảy ra (đã validate ở extract_intent()), nhưng
                # vẫn giữ nhánh phòng thủ để không crash cả câu hỏi.
                failed_notes.append(f"- Không tìm thấy công thức cho chỉ số '{metric_key}'.")
                continue

            metric_spec_json = spec.display_json(metric_key)
            metric_specs.append(metric_spec_json)
            print(f"[Calculation] Công thức sẽ dùng cho '{metric_key}': {metric_spec_json}")

            operands: dict[str, OperandDetail] = {}
            missing_fields: list[str] = []
            for field in spec.required_metrics:
                operand, used_citations = fetch_operand_cached(field)
                if operand is None:
                    missing_fields.append(FIELD_LABELS.get(field, field))
                    continue
                operands[field] = operand
                all_citations.extend(used_citations)

            if missing_fields:
                print(f"[Calculation] Thiếu operand: {missing_fields}, không đủ dữ liệu để tính {spec.name_vi}.")
                failed_notes.append(
                    f"- Không tìm đủ dữ liệu để tính **{spec.name_vi}** {ticker_part}{period_label}. "
                    f"Thiếu: {', '.join(missing_fields)}."
                )
                continue

            try:
                output = compute_metric_from_operands(metric_key, operands)
            except FormulaError as e:
                failed_notes.append(f"- Không thể tính **{spec.name_vi}**: {e}")
                continue

            print(f"[Calculation] Kết quả {spec.name_vi} = {output.result} {spec.unit} "
                  f"(công thức: {output.formula})")
            calculations.append(output)
            section_answers.append(format_calculation_answer(
                metric_label=spec.name_vi,
                period_label=period_label,
                ticker=intent.ticker,
                output=output,
                spec=spec,
                field_labels=FIELD_LABELS,
            ))

        # Dedup citation TOÀN CỤC trên tất cả các metric đã tính (theo
        # chunk_id thật -- xem ghi chú gốc, KHÔNG dùng (doc_id, page_start)).
        deduped_citations, seen_keys = [], set()
        for c in all_citations:
            key = c.get("chunk_id") or (c.get("doc_id"), c.get("page_start"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_citations.append(c)
        for i, c in enumerate(deduped_citations, start=1):
            c["index"] = i
            c["label"] = format_citation_label(c, i)

        if not section_answers:
            # KHÔNG có chỉ số nào tính được -- trả lại lý do cụ thể cho
            # TỪNG chỉ số đã thử, thay vì 1 câu chung chung.
            answer = "\n".join(failed_notes) or (
                f"Không thể tính được chỉ số bạn yêu cầu {ticker_part}{period_label}."
            )
            return CalculationResponse(
                answer=answer,
                citations=deduped_citations,
                metric_spec=metric_specs[0] if metric_specs else None,
                metric_specs=metric_specs,
                intent=intent_json,
            )

        if len(section_answers) == 1 and not failed_notes:
            # Câu hỏi chỉ có ĐÚNG 1 chỉ số (trường hợp phổ biến nhất) -- giữ
            # NGUYÊN định dạng câu trả lời như trước khi có multi-metric.
            answer = section_answers[0]
        else:
            answer = "\n\n---\n\n".join(section_answers)
            if failed_notes:
                answer += "\n\n---\n\n**Một số chỉ số không tính được:**\n" + "\n".join(failed_notes)

        return CalculationResponse(
            answer=answer,
            calculation=calculations[0] if calculations else None,
            calculations=calculations,
            citations=deduped_citations,
            metric_spec=metric_specs[0] if metric_specs else None,
            metric_specs=metric_specs,
            intent=intent_json,
        )