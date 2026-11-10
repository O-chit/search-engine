# SPEC: Hệ thống tìm kiếm sản phẩm cá nhân hóa (BM25 + Vector)

Tài liệu này là đặc tả đầy đủ để xây dựng hệ thống. Thực hiện theo đúng thứ tự các giai đoạn ở mục 14. Mỗi giai đoạn phải đạt tiêu chí hoàn thành trước khi sang giai đoạn sau.

---

## 1. Mục tiêu

Xây dựng một dịch vụ tìm kiếm sản phẩm thương mại điện tử:

- Nhận truy vấn bằng tiếng Anh hoặc tiếng Việt, trả về danh sách sản phẩm xếp theo độ liên quan.
- Truy xuất kết hợp BM25 (khớp từ khóa) và vector search (khớp ngữ nghĩa), gộp bằng Reciprocal Rank Fusion.
- Cá nhân hóa thứ tự kết quả theo lịch sử click của từng trình duyệt, không cần đăng nhập.
- Có một trang web tối giản: ô tìm kiếm, danh sách kết quả, ghi nhận click.

Không làm: đăng nhập, tài khoản, giỏ hàng, thanh toán, autocomplete, trang quản trị, crawler.

## 2. Ràng buộc

| Hạng mục | Giá trị |
| --- | --- |
| Máy chạy | 1 máy, 16 GB RAM, GPU RTX 3050 |
| Dữ liệu | Home Depot Product Search Relevance (file CSV), khoảng 124 nghìn sản phẩm, nội dung tiếng Anh |
| Ngôn ngữ truy vấn | Tiếng Anh và tiếng Việt |
| Độ trễ `/search` p95 | dưới 300 ms khi tắt cross-encoder |
| Định danh người dùng | `session_id` do trình duyệt tự sinh, lưu trong `localStorage` |

## 3. Công nghệ

| Thành phần | Lựa chọn |
| --- | --- |
| Ngôn ngữ | Python 3.11 |
| API | FastAPI + Uvicorn |
| Chỉ mục BM25 và vector | OpenSearch 2.x, một nút, chạy bằng Docker Compose |
| Lưu phiên, truy vấn, sự kiện | SQLite (file `data/app.db`), truy cập qua `sqlite3` chuẩn |
| Embedding | `BAAI/bge-m3` qua `sentence-transformers`, chạy trên GPU, fp16 |
| Cross-encoder (tùy chọn) | `BAAI/bge-reranker-v2-m3`, mặc định tắt |
| Learning-to-rank | LightGBM (`lambdarank`) |
| Giao diện | Một trang HTML + JavaScript thuần, FastAPI phục vụ dạng static |
| Kiểm thử | pytest |

Cấu hình OpenSearch trong `docker-compose.yml`:

- `discovery.type=single-node`
- `DISABLE_SECURITY_PLUGIN=true`
- `OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g`
- Mở cổng 9200, gắn volume để dữ liệu không mất khi khởi động lại.

Ghim phiên bản cụ thể của image OpenSearch và mọi thư viện Python trong `requirements.txt`. Kiểm tra cú pháp mapping `knn_vector` và truy vấn `knn` theo tài liệu của đúng phiên bản đã ghim.

## 4. Cấu trúc thư mục

```
search-engine/
├── docker-compose.yml
├── requirements.txt
├── .env.example
├── README.md
├── config.py                 # mọi tham số ở mục 13
├── data/
│   ├── raw/                  # các file CSV Home Depot đặt ở đây
│   ├── processed/            # products.parquet, embeddings.npy, uid_index.json
│   └── app.db                # SQLite
├── ingestion/
│   ├── load_homedepot.py     # đọc CSV, ghép bảng, làm sạch
│   └── normalize.py          # chuẩn hóa văn bản, dùng chung cho index và truy vấn
├── indexing/
│   ├── mapping.json
│   ├── embed.py              # tạo embeddings.npy
│   └── bulk_index.py
├── query/
│   └── parser.py             # chuẩn hóa, nhận diện ngôn ngữ, tạo vector
├── retrieval/
│   ├── bm25.py
│   ├── vector.py
│   └── fusion.py             # RRF có trọng số
├── ranking/
│   ├── features.py
│   ├── ltr.py
│   └── cross_encoder.py
├── personalization/
│   ├── profile.py            # cập nhật vector sở thích của phiên
│   └── scorer.py             # điểm cuối, quy tắc bảo vệ, vị trí khám phá
├── storage/
│   └── db.py                 # tạo bảng, hàm đọc ghi SQLite
├── api/
│   ├── main.py
│   ├── schemas.py
│   └── routes/               # search.py, events.py, session.py
├── web/
│   ├── index.html
│   └── app.js
├── eval/
│   ├── split.py              # chia truy vấn train/test
│   ├── run_eval.py
│   ├── simulate_personalization.py
│   └── results/
└── tests/
```

## 5. Dữ liệu nguồn

Đặt các file vào `data/raw/`. Trước khi viết code đọc, in ra tên cột và 3 dòng đầu của từng file để xác nhận lược đồ; nếu tên cột khác bảng dưới thì ánh xạ lại trong `load_homedepot.py`.

| File | Cột mong đợi | Dùng cho |
| --- | --- | --- |
| `train.csv` | `id`, `product_uid`, `product_title`, `search_term`, `relevance` | Tên sản phẩm; cặp truy vấn–sản phẩm có nhãn 1–3 để đánh giá và huấn luyện |
| `test.csv` | `id`, `product_uid`, `product_title`, `search_term` | Bổ sung tên sản phẩm chưa có trong `train.csv` |
| `product_descriptions.csv` | `product_uid`, `product_description` | Mô tả |
| `attributes.csv` | `product_uid`, `name`, `value` | Thương hiệu, gạch đầu dòng, thuộc tính |

Quy tắc đọc và ghép:

1. Đọc CSV với `encoding="ISO-8859-1"`; nếu lỗi thì thử `utf-8`.
2. Tập sản phẩm = hợp các `product_uid` trong `train.csv` và `test.csv`, mỗi `product_uid` một dòng, lấy `product_title` đầu tiên.
3. `brand` = `value` của dòng trong `attributes.csv` có `name == "MFG Brand Name"`.
4. `bullets` = nối các `value` có `name` bắt đầu bằng `Bullet`, theo thứ tự tên.
5. `attributes_text` = nối `"name: value"` của các thuộc tính còn lại, tối đa 30 thuộc tính mỗi sản phẩm.
6. Ghép `product_description` theo `product_uid`.
7. Làm sạch: bỏ thẻ HTML, chuẩn hóa Unicode NFC, gộp khoảng trắng thừa, tách các từ bị dính liền kiểu `chairsThe` thành `chairs The`. Giữ nguyên chữ hoa thường trong dữ liệu lưu; việc đưa về chữ thường do analyzer đảm nhiệm.
8. Ghi kết quả ra `data/processed/products.parquet` với các cột: `product_uid` (chuỗi), `title`, `brand`, `bullets`, `description`, `attributes_text`.

Dữ liệu này không có giá, danh mục và độ phổ biến; không tạo các trường đó.

## 6. Chỉ mục OpenSearch

Tên chỉ mục: `products`. `_id` của mỗi tài liệu là `product_uid`.

Settings:

- `index.knn: true`
- `number_of_shards: 1`, `number_of_replicas: 0`
- Hai similarity tùy chỉnh: `bm25_short` (`k1=1.2`, `b=0.3`) và `bm25_long` (`k1=1.2`, `b=0.75`).

Mapping:

| Trường | Kiểu | Analyzer | Similarity |
| --- | --- | --- | --- |
| `product_uid` | keyword | | |
| `title` | text | `english` | `bm25_short` |
| `brand` | text, kèm subfield `brand.raw` kiểu keyword | `standard` | `bm25_short` |
| `bullets` | text | `english` | `bm25_long` |
| `description` | text | `english` | `bm25_long` |
| `attributes_text` | text | `english` | `bm25_long` |
| `embedding` | `knn_vector`, `dimension: 1024` | | |

Cấu hình `embedding`: phương thức `hnsw`, `space_type: cosinesimil`, `engine: lucene`, `m: 16`, `ef_construction: 200`.

Loại trường `embedding` khỏi `_source` trả về khi truy vấn (`_source.excludes`) để phản hồi nhẹ.

## 7. Embedding

`indexing/embed.py`:

1. Văn bản của mỗi sản phẩm = `title + ". " + brand + ". " + bullets + ". " + 150 từ đầu của description`.
2. Nạp `BAAI/bge-m3` lên GPU, chuyển sang fp16 (`model.half()`), `max_seq_length = 256`.
3. Encode theo lô, `batch_size = 16`, `normalize_embeddings=True`. Gặp lỗi hết bộ nhớ GPU thì giảm một nửa batch và chạy tiếp.
4. Ghi `data/processed/embeddings.npy` (float32, kích thước `[số sản phẩm, 1024]`) và `data/processed/uid_index.json` (ánh xạ `product_uid` → chỉ số dòng).
5. Hỗ trợ chạy tiếp từ điểm dừng: lưu tạm sau mỗi 5000 sản phẩm.

`indexing/bulk_index.py` đọc `products.parquet` và `embeddings.npy`, ghi vào OpenSearch bằng bulk API, mỗi lô 500 tài liệu. Chạy lại phải ghi đè, không tạo bản trùng.

Tiến trình API nạp `embeddings.npy` bằng `np.load(..., mmap_mode="r")` cùng `uid_index.json` khi khởi động, để tra vector sản phẩm theo `product_uid` mà không gọi OpenSearch.

## 8. Xử lý truy vấn

`query/parser.py` nhận chuỗi thô và trả về đối tượng `ParsedQuery`:

```json
{
  "raw": "Máy khoan pin DeWalt 20V",
  "normalized": "máy khoan pin dewalt 20v",
  "lang": "vi",
  "has_model_code": false,
  "vector": [0.013, -0.042]
}
```

Các bước:

1. Chuẩn hóa Unicode NFC, chữ thường, bỏ khoảng trắng thừa, cắt còn tối đa 200 ký tự.
2. Truy vấn rỗng sau chuẩn hóa: trả lỗi 400.
3. `lang = "vi"` nếu chuỗi chứa ít nhất một ký tự có dấu tiếng Việt (`ăâđêôơư` và các nguyên âm mang dấu thanh); ngược lại `lang = "en"`.
4. `has_model_code = true` nếu có token dài từ 5 ký tự, gồm cả chữ và số (ví dụ `dcd771c2`).
5. Tạo vector truy vấn bằng `bge-m3` (GPU, fp16, chuẩn hóa độ dài 1). Cache kết quả bằng LRU 10 000 mục, khóa là chuỗi `normalized`.

Hàm chuẩn hóa nằm trong `ingestion/normalize.py` và được dùng chung cho cả lúc đánh chỉ mục lẫn lúc truy vấn.

Nạp mô hình embedding một lần khi API khởi động và gọi encode một lần với chuỗi giả để làm nóng.

## 9. Truy xuất hybrid

Chạy hai nhánh song song bằng `asyncio.gather`, mỗi nhánh lấy `RETRIEVAL_K = 200` kết quả, thời gian chờ tối đa 150 ms mỗi nhánh.

### 9.1. Nhánh BM25 (`retrieval/bm25.py`)

Truy vấn `multi_match` kiểu `best_fields` với `tie_breaker: 0.3` trên các trường và trọng số:

| Trường | Trọng số |
| --- | --- |
| `title` | 3.0 |
| `brand` | 2.0 |
| `bullets` | 1.5 |
| `attributes_text` | 1.0 |
| `description` | 1.0 |

- `lang == "en"`: `minimum_should_match: "75%"`.
- `lang == "vi"`: `minimum_should_match: "1"`. Dữ liệu là tiếng Anh nên nhánh này chỉ bắt được tên thương hiệu, mã model và thông số.

### 9.2. Nhánh vector (`retrieval/vector.py`)

Truy vấn `knn` trên trường `embedding` với `k = 200`. `bge-m3` là mô hình đa ngữ nên truy vấn tiếng Việt vẫn tìm được sản phẩm tiếng Anh ở nhánh này.

### 9.3. Gộp bằng RRF có trọng số (`retrieval/fusion.py`)

```
rrf(d) = w_bm25 / (RRF_K + rank_bm25(d)) + w_vec / (RRF_K + rank_vec(d))
```

- `RRF_K = 60`. Thứ hạng bắt đầu từ 1. Tài liệu vắng mặt ở một nhánh thì nhánh đó đóng góp 0.
- `lang == "en"`: `w_bm25 = 1.0`, `w_vec = 1.0`.
- `lang == "vi"`: `w_bm25 = 0.3`, `w_vec = 1.0`.
- Giữ lại `CANDIDATE_K = 100` tài liệu có `rrf` cao nhất. Với mỗi ứng viên lưu: `product_uid`, `bm25_score`, `bm25_rank`, `vec_score`, `vec_rank`, `rrf`.

### 9.4. Xử lý lỗi

- Một nhánh quá hạn hoặc lỗi: dùng kết quả của nhánh còn lại, đặt `degraded = true` trong phản hồi và ghi log.
- Cả hai nhánh lỗi: trả 503.
- Không có ứng viên nào: trả danh sách rỗng với mã 200.

## 10. Xếp hạng lại

Đầu ra của tầng này là `rel` trong khoảng 0–1 cho mỗi ứng viên. Chế độ chọn bằng `RANKER` trong config.

### 10.1. `RANKER = "rrf"` (mặc định ở các giai đoạn đầu)

`rel` = `rrf` chuẩn hóa min-max trong danh sách ứng viên. Danh sách chỉ có một phần tử thì `rel = 1`.

### 10.2. `RANKER = "ltr"`

Mô hình LightGBM `lambdarank` huấn luyện từ `train.csv` (`ranking/ltr.py`):

- Nhãn: `relevance` (1–3) quy về số nguyên 0–4 bằng `round((relevance - 1) * 2)`.
- Nhóm theo `search_term`. Chỉ dùng các truy vấn thuộc tập train do `eval/split.py` tạo.
- Đặc trưng (`ranking/features.py`), tính cho từng cặp truy vấn–sản phẩm:

| Nhóm | Đặc trưng |
| --- | --- |
| Khớp từ khóa | Điểm BM25 tổng; điểm BM25 riêng trên `title`; tỷ lệ token truy vấn xuất hiện trong `title`; tỷ lệ xuất hiện trong `bullets + description` |
| Khớp ngữ nghĩa | Cosine giữa vector truy vấn và vector sản phẩm |
| Thuộc tính | Tên `brand` có xuất hiện trong truy vấn không (0/1) |
| Truy vấn | Số token; `has_model_code` (0/1); `lang == "vi"` (0/1) |
| Sản phẩm | Số token của `title` |

- Điểm BM25 cho cặp có nhãn lấy bằng cách truy vấn OpenSearch có lọc theo `product_uid`.
- Lưu mô hình vào `data/processed/ltr.txt`. Lúc phục vụ, `rel` = điểm mô hình chuẩn hóa min-max trong danh sách ứng viên.
- Không đưa `bm25_rank`, `vec_rank`, `rrf` vào đặc trưng, vì các cặp có nhãn không đi qua bước truy xuất.

### 10.3. Cross-encoder (`ENABLE_CROSS_ENCODER`, mặc định `false`)

Khi bật: chấm lại `CROSS_ENCODER_TOP_N = 20` ứng viên đứng đầu bằng `bge-reranker-v2-m3` (fp16, GPU), đầu vào là cặp (truy vấn, `title + ". " + bullets` cắt 200 từ). Điểm qua sigmoid thay cho `rel` ở 20 vị trí này; các ứng viên còn lại giữ thứ tự cũ và nằm sau.

## 11. Cá nhân hóa không cần đăng nhập

### 11.1. Định danh

- `web/app.js` sinh một UUID v4 ở lần đầu mở trang, lưu vào `localStorage` với khóa `session_id`, gửi kèm mọi request trong header `X-Session-Id`.
- API tạo dòng mới trong bảng `sessions` khi gặp `session_id` chưa có. Request thiếu header vẫn được phục vụ, không cá nhân hóa.

### 11.2. Hồ sơ của phiên

| Thành phần | Cách tính |
| --- | --- |
| `profile_vec` | Trung bình trượt của embedding các sản phẩm đã click, chuẩn hóa độ dài 1 |
| `recent_uids` | Danh sách tối đa 5 `product_uid` vừa click gần nhất |
| `recent_vec` | Trung bình embedding của `recent_uids`, chuẩn hóa độ dài 1; tính lúc truy vấn |
| `event_count` | Số click hợp lệ đã ghi nhận |

Cập nhật `profile_vec` sau mỗi click hợp lệ, với `e` là embedding sản phẩm và `ETA = 0.2`:

```
nếu profile_vec chưa có:  profile_vec = e
ngược lại:                profile_vec = normalize((1 - ETA) * profile_vec + ETA * e)
```

Click hợp lệ: không trùng `product_uid` với click ngay trước đó của cùng phiên trong vòng 2 giây.

### 11.3. Điểm cuối (`personalization/scorer.py`)

```
lambda = min(1, event_count / WARMUP_EVENTS)
score(d) = rel(d) + lambda * (BETA_PROFILE * cos(profile_vec, d) + BETA_RECENT * cos(recent_vec, d))
```

- `WARMUP_EVENTS = 5`, `BETA_PROFILE = 0.15`, `BETA_RECENT = 0.25`.
- `event_count == 0` hoặc `personalize=false`: `score = rel`.
- Vector sản phẩm lấy từ `embeddings.npy` đã nạp trong bộ nhớ.
- Sắp xếp ứng viên theo `score` giảm dần.

### 11.4. Quy tắc bảo vệ

Bỏ qua cá nhân hóa (dùng `score = rel`) khi một trong các điều kiện sau đúng:

- `has_model_code == true`.
- Chuỗi `normalized` trùng hoàn toàn với `title` đã chuẩn hóa của ứng viên đứng đầu.

### 11.5. Vị trí khám phá

Khi cá nhân hóa được áp dụng và có ít nhất 10 kết quả: tìm ứng viên có `rel` cao nhất trong số các ứng viên nằm trong top 10 theo `rel` nhưng bị đẩy ra ngoài top 10 theo `score`. Nếu có, đặt nó vào vị trí thứ 8 và đánh dấu `explored = true`.

### 11.6. Xóa hồ sơ

`DELETE /session` xóa `profile_vec`, `recent_uids`, đặt `event_count = 0` cho phiên trong header.

## 12. Lưu trữ và API

### 12.1. Bảng SQLite (`storage/db.py`)

Bật `PRAGMA journal_mode=WAL`. Tạo bảng khi khởi động nếu chưa có.

| Bảng | Cột |
| --- | --- |
| `sessions` | `session_id` TEXT PK, `profile_vec` BLOB (float32, 1024 chiều), `recent_uids` TEXT (JSON), `event_count` INTEGER, `created_at` TEXT, `updated_at` TEXT |
| `queries` | `query_id` TEXT PK, `session_id` TEXT, `raw_text` TEXT, `normalized_text` TEXT, `lang` TEXT, `result_uids` TEXT (JSON, đúng thứ tự đã trả), `personalized` INTEGER, `degraded` INTEGER, `timing_ms` TEXT (JSON), `ts` TEXT |
| `events` | `event_id` INTEGER PK AUTOINCREMENT, `query_id` TEXT, `session_id` TEXT, `product_uid` TEXT, `type` TEXT, `position` INTEGER, `ts` TEXT |

Tạo index trên `events(session_id, ts)` và `queries(session_id, ts)`.

### 12.2. Endpoint

| Endpoint | Phương thức | Đầu vào | Trả về |
| --- | --- | --- | --- |
| `/search` | GET | `q` (bắt buộc), `page` (mặc định 1), `size` (mặc định 20, tối đa 50), `personalize` (mặc định `true`), `debug` (mặc định `false`); header `X-Session-Id` | Xem mẫu bên dưới |
| `/events` | POST | JSON: `query_id`, `product_uid`, `type` (`click`), `position`; header `X-Session-Id` | 202; ghi sự kiện và cập nhật hồ sơ bằng `BackgroundTasks` |
| `/session` | GET | header `X-Session-Id` | `event_count`, `recent_uids` |
| `/session` | DELETE | header `X-Session-Id` | 204 |
| `/health` | GET | | Trạng thái OpenSearch, mô hình đã nạp, số tài liệu trong chỉ mục |
| `/` | GET | | Trang `web/index.html` |

Mẫu phản hồi `/search`:

```json
{
  "query_id": "q_8f31c2a0",
  "normalized_query": "cordless drill",
  "lang": "en",
  "personalized": true,
  "degraded": false,
  "total": 100,
  "page": 1,
  "size": 20,
  "results": [
    {
      "product_uid": "100001",
      "title": "...",
      "brand": "...",
      "snippet": "160 ký tự đầu của bullets hoặc description",
      "score": 0.91,
      "position": 1,
      "explored": false,
      "debug": {"bm25_rank": 2, "vec_rank": 5, "rrf": 0.031, "rel": 0.84, "cos_profile": 0.42, "cos_recent": 0.51}
    }
  ],
  "timing_ms": {"parse": 12, "retrieval": 48, "rank": 5, "personalize": 2, "total": 70}
}
```

- Khối `debug` chỉ xuất hiện khi `debug=true`.
- Phân trang cắt từ danh sách 100 ứng viên đã xếp hạng; không truy xuất lại.
- Mỗi lần gọi `/search` với `page=1` ghi một dòng vào `queries`.
- Bật CORS cho `localhost`. Dùng Pydantic cho mọi schema vào ra (`api/schemas.py`).

### 12.3. Giao diện (`web/`)

- Một ô nhập, nút tìm, tìm khi nhấn Enter.
- Mỗi kết quả hiển thị `title`, `brand`, `snippet`; kết quả `explored` có nhãn nhỏ "Khám phá".
- Click vào kết quả gửi `POST /events` kèm `query_id`, `product_uid`, `position`, sau đó đánh dấu kết quả là đã xem.
- Có công tắc "Cá nhân hóa" (bật/tắt tham số `personalize`), công tắc "Debug" (hiển thị khối `debug` dưới mỗi kết quả) và nút "Xóa lịch sử" (gọi `DELETE /session`).
- Hiển thị `timing_ms.total` và số click đã ghi nhận của phiên.

## 13. Cấu hình (`config.py`, đọc từ `.env`)

| Tham số | Giá trị mặc định |
| --- | --- |
| `OPENSEARCH_URL` | `http://localhost:9200` |
| `INDEX_NAME` | `products` |
| `EMBED_MODEL` | `BAAI/bge-m3` |
| `EMBED_DIM` | 1024 |
| `EMBED_BATCH_SIZE` | 16 |
| `DEVICE` | `cuda` (tự chuyển `cpu` nếu không có GPU) |
| `RETRIEVAL_K` | 200 |
| `CANDIDATE_K` | 100 |
| `RRF_K` | 60 |
| `W_BM25_EN`, `W_VEC_EN` | 1.0, 1.0 |
| `W_BM25_VI`, `W_VEC_VI` | 0.3, 1.0 |
| `BRANCH_TIMEOUT_MS` | 150 |
| `KNN_EF_SEARCH` | 100 |
| `RANKER` | `rrf` |
| `ENABLE_CROSS_ENCODER` | `false` |
| `CROSS_ENCODER_MODEL` | `BAAI/bge-reranker-v2-m3` |
| `CROSS_ENCODER_TOP_N` | 20 |
| `ETA` | 0.2 |
| `WARMUP_EVENTS` | 5 |
| `BETA_PROFILE` | 0.15 |
| `BETA_RECENT` | 0.25 |
| `RECENT_MAX` | 5 |
| `EXPLORE_POSITION` | 8 |
| `SQLITE_PATH` | `data/app.db` |

Không viết cứng các giá trị này ở nơi khác trong code.

## 14. Các giai đoạn thực hiện

### Giai đoạn 1: Dữ liệu và BM25

- Viết `docker-compose.yml`, `requirements.txt`, `config.py`, `ingestion/`, `indexing/mapping.json`, `indexing/bulk_index.py` (chưa có embedding), `retrieval/bm25.py`, `api/` với `/search` chỉ dùng BM25 và `/health`.
- Hoàn thành khi: `products.parquet` có đủ sản phẩm; `/health` báo số tài liệu khớp số dòng parquet; `GET /search?q=cordless drill` trả về kết quả có chứa từ khóa trong `title`.

### Giai đoạn 2: Bộ đánh giá

- `eval/split.py`: chia các `search_term` duy nhất của `train.csv` thành 80% train, 20% test với seed 42; ghi ra `data/processed/split.json`.
- `eval/run_eval.py --config <tên>`: với mỗi truy vấn test, chạy pipeline (tắt cá nhân hóa) và tính:
  - `Recall@100`: tỷ lệ sản phẩm có `relevance >= 2.5` xuất hiện trong 100 ứng viên.
  - `NDCG@10` rút gọn: chỉ xét các sản phẩm có nhãn của truy vấn đó, sắp theo thứ tự hệ thống trả về; sản phẩm có nhãn nhưng không nằm trong 100 ứng viên xếp cuối. Gain = `2^label - 1` với `label = round((relevance - 1) * 2)`.
  - `MRR`: theo vị trí của sản phẩm đầu tiên có `relevance >= 2.5` trong danh sách hệ thống trả về.
  - Độ trễ p50 và p95.
- Ghi kết quả vào `eval/results/<tên>.json` và in bảng so sánh mọi cấu hình đã chạy.
- Hoàn thành khi: có file kết quả cho cấu hình `bm25`.

### Giai đoạn 3: Vector và hybrid

- Viết `indexing/embed.py`, bổ sung embedding vào `bulk_index.py`, đánh chỉ mục lại; viết `query/parser.py`, `retrieval/vector.py`, `retrieval/fusion.py`; `/search` chạy hybrid.
- Chạy đánh giá cho ba cấu hình: `bm25`, `vector`, `hybrid`.
- Hoàn thành khi: `Recall@100` của `hybrid` không thấp hơn cả `bm25` lẫn `vector`; truy vấn tiếng Việt như `máy khoan không dây` trả về sản phẩm máy khoan; p95 dưới 300 ms.

### Giai đoạn 4: Giao diện và ghi sự kiện

- Viết `storage/db.py`, `POST /events`, `GET/DELETE /session`, `web/index.html`, `web/app.js`.
- Hoàn thành khi: mỗi lần tìm tạo một dòng trong `queries`; mỗi click tạo một dòng trong `events` với đúng `query_id` và `position`.

### Giai đoạn 5: Cá nhân hóa

- Viết `personalization/profile.py`, `personalization/scorer.py`, nối vào `/search`.
- Viết `eval/simulate_personalization.py`: tạo hai phiên; phiên A click 6 sản phẩm lấy từ kết quả của `paint brush`, phiên B click 6 sản phẩm lấy từ kết quả của `cordless drill`; sau đó cả hai cùng tìm `tools`.
- Hoàn thành khi script xác nhận được cả bốn điều:
  1. Top 10 của hai phiên cho `tools` khác nhau.
  2. Cosine trung bình giữa top 10 và `profile_vec` của mỗi phiên cao hơn so với khi `personalize=false`.
  3. Với một truy vấn chứa mã model, thứ tự kết quả của hai phiên giống hệt nhau.
  4. Sau `DELETE /session`, kết quả của phiên trùng với kết quả khi `personalize=false`.

### Giai đoạn 6: Learning-to-rank và cross-encoder

- Viết `ranking/features.py`, `ranking/ltr.py`, `ranking/cross_encoder.py`.
- Chạy đánh giá cho `hybrid_ltr` và `hybrid_ltr_ce`.
- Hoàn thành khi: `NDCG@10` của `hybrid_ltr` cao hơn `hybrid`. Nếu không cao hơn, giữ `RANKER = "rrf"` làm mặc định và ghi kết quả vào README.

## 15. Kiểm thử (`tests/`)

Viết test cho từng phần ngay trong giai đoạn tạo ra nó:

- `normalize`: NFC, khoảng trắng, tách từ dính liền.
- `parser`: nhận diện `lang`, `has_model_code`, truy vấn rỗng.
- `fusion`: RRF với tài liệu chỉ có ở một nhánh; trọng số theo ngôn ngữ; kết quả ổn định khi điểm bằng nhau (phá hòa bằng `product_uid`).
- `profile`: click đầu tiên, cập nhật trung bình trượt, giới hạn `recent_uids`, bỏ click trùng trong 2 giây.
- `scorer`: `event_count = 0` cho `score == rel`; quy tắc bảo vệ; vị trí khám phá.
- API: `/search` thiếu `q` trả 422; thiếu header vẫn trả kết quả; `/events` với `query_id` không tồn tại trả 404.

Test của `fusion`, `profile`, `scorer`, `parser` (phần không cần mô hình) phải chạy được khi không có OpenSearch và không có GPU.

## 16. Quy tắc viết code

- Type hint cho mọi hàm công khai; docstring một dòng.
- Mọi lời gọi OpenSearch đi qua `retrieval/`; mọi truy cập SQLite đi qua `storage/db.py`.
- Ghi log dạng JSON một dòng cho mỗi request `/search`: `query_id`, `lang`, `degraded`, `timing_ms`.
- Không bắt exception rồi bỏ qua; lỗi ở một nhánh truy xuất phải được ghi log kèm nguyên nhân.
- Script dòng lệnh (`load_homedepot.py`, `embed.py`, `bulk_index.py`, `run_eval.py`) chạy bằng `python -m <module>` từ thư mục gốc và in tiến độ.
- `README.md` ghi đủ các lệnh theo thứ tự: khởi động OpenSearch, nạp dữ liệu, tạo embedding, đánh chỉ mục, chạy API, chạy đánh giá, chạy test.

## 17. Giới hạn đã biết

- Truy vấn tiếng Việt gõ không dấu được coi là tiếng Anh; kết quả phụ thuộc vào nhánh vector.
- Dữ liệu không có giá và danh mục nên không có bộ lọc.
- Nhãn Home Depot chỉ phủ vài sản phẩm cho mỗi truy vấn, nên `NDCG@10` rút gọn dùng để so sánh giữa các cấu hình, không phải số đo tuyệt đối.
- Cá nhân hóa mới được kiểm chứng bằng mô phỏng; cần người dùng thật để đo hiệu quả.
