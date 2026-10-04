# Multimodal PDF extraction

Local extraction stage for Gemini Embedding 2 + Pinecone RAG. Does not read `.env`, call APIs, or generate embeddings.

## Run on Windows

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m ingestion.extract_pdf
```

Outputs are saved under `storage/<filename>_<hash>/`: page screenshots, page text, individual raster image crops, tables as JSON/CSV and PNG, and `manifest.json`. Paths in the manifest are relative to its directory. Page numbers are one-based; bounding boxes use PDF page coordinates. Document IDs use the full source SHA-256.

Each image has nearby text candidates, not an invented caption. Later, use a verified PDF caption or a vision model to generate one and embed it together with the image. Keep original evidence paths in Pinecone metadata.

## Extraction limits

- All pages are rendered so vector drawings and complex layouts remain available as evidence. Automatic individual image extraction covers raster image occurrences; vector drawings and animals inside scanned pages require later segmentation/manual crops.
- Tiny raster decorations (under 12 PDF points or 16 source pixels in either dimension) are skipped; they remain visible in page screenshots.
- Tables are detected heuristically. JSON retains cell rows, merged-cell nulls, and header information. CSV contains the extracted rows; external headers are recorded separately in JSON. PNG includes the detected header.
- Default table detection uses drawn lines. For borderless tables, try `--table-strategy text` and review results for false positives.
- For scanned pages, install Tesseract and its language data, then pass `--ocr --language eng`. Without OCR, pages lacking native text are flagged in the manifest. OCR does not provide structured scanned-table recognition.
- Existing output directories are refused. For another run use `--output storage_review`; failed runs may leave partial output.
- Password-protected PDFs must be unlocked first.

## Next stage

### Conversation memory

`/api/chat` now loads completed previous turns from the same MySQL conversation before saving the current turn. Recent turns and older user messages relevant to the question are selected within a bounded context budget. Personal name questions use chat context without PDF retrieval or invented citations. Newer user corrections take priority in the answering prompt. No age-based expiry is applied, so a name from days earlier can be recalled in the same conversation. A new conversation has separate memory. This is bounded lexical memory, not exhaustive recall of all long conversations. History is sent to Groq along with the question; PDF questions still use Gemini/Pinecone retrieval. CLI callers can supply `--history` as a JSON list of role/content objects.

### Persistent chat history

`POST /api/conversations` creates a conversation. `/api/chat` accepts `conversation_id`, saves the question and a pending assistant message, then saves the answer and its citations/image references together. Failed answers are recorded with status `failed`. `GET /api/conversations/{id}` reloads messages and regenerates evidence URLs from saved source IDs. The browser stores the active conversation ID and restores its messages after refresh. New chat starts another conversation without deleting the previous one. Restart Uvicorn after code updates; run `python -m db.init_database` first if the tables do not exist.

The local demo has no user accounts or ownership controls. Persisted history is displayed, but follow-up questions are still independent searches; history is not sent to the model.

### MySQL chat schema

Configure `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, and `MYSQL_DATABASE` in `.env`, then run:

```powershell
python -m pip install -r requirements.txt
python -m db.init_database
```

SQLAlchemy uses `mysql+mysqlconnector` with `use_pure=True`. Initialization creates `rag_chat` by default, then `conversations`, `messages`, and `message_sources`. Foreign keys connect conversations to messages and messages to sources; deletes cascade. UUID primary keys, UTF-8 storage, query indexes, and UTC timestamps are included. Existing tables/data are preserved; `create_all` is not a schema migration tool. The engine and session dependency are in `db/database.py`, models in `db/models.py`. This step creates storage only; `/api/chat` persistence and conversation-history endpoints are not wired yet.

### FastAPI demo with floating chat

```powershell
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000. The floating chat supports general questions, image requests, and table searches. Answers include clickable page evidence and inline retrieved images. API documentation: http://127.0.0.1:8000/docs. Each question is independent; displayed conversation history is browser memory, not model conversation context.

The backend reuses the existing CLI pipeline with isolated temporary outputs. It runs one answer at a time, has a 240-second timeout, and serves only manifest-listed PNG evidence by record ID. It never exposes `.env`, checkpoints, or arbitrary filesystem paths. This is a local demo; authentication, durable jobs, and multi-user rate limiting are needed before public hosting. API keys remain server-side. The page uses Google Fonts with local font fallbacks.

### Evaluate the pipeline

Answers attach at most two images by default to leave headroom below the observed Groq free-tier input token limit. Larger text/table evidence may still require reducing `--top-k`.

```powershell
python -m evaluation.evaluate_rag --dry-run --allow-unreviewed
python -m evaluation.evaluate_rag --limit 1 --allow-unreviewed
python -m evaluation.evaluate_rag evaluation/evaluation_cases.json --output-dir evaluation_run_2
python -m evaluation.evaluate_rag --cached --allow-unreviewed
```

`evaluation/evaluation_cases.json` contains four draft starter cases, not verified ground truth. Check expected pages, image paths, and answers against the PDF, mark `reviewed=true`, and expand to 15-20 diverse questions. Unreviewed datasets are refused unless `--allow-unreviewed` is supplied; draft reports are explicitly labeled. Expected pages are PDF page numbers, not printed textbook numbers.

Evaluation saves retrieval and answer JSON per case plus `report.json`. It reuses saved retrieval for answer generation, avoiding duplicate query embeddings. Metrics include hit@k, reciprocal rank, expected-page/record recall, citation references, optional phrase checks, and expected-image selection. Phrase checks and valid labels do not establish factual correctness: fill in the human-review fields after inspecting answers and evidence. Unsupported-question checks verify empty sources/images only; review refusal wording manually. Errors are reported separately with explicit metric denominators. `--retrieval-only` skips Groq. `--cached` scores existing outputs without API calls. Use the same dataset ordering and settings when scoring cached output; dry runs validate the dataset only. Actual runs call Gemini, Pinecone, and Groq; files are retained per case if a request fails.

### Answer questions with Groq

```powershell
python -m rag.answer_question "What are amphibians?" --dry-run
python -m rag.answer_question "What are amphibians?"
python -m rag.answer_question "Give me a frog image" --type image --top-k 3 --output frog_answer.json
```

Uses the retrieval script, then sends evidence to Groq (`GROQ_API_KEY`). `GROQ_ANSWER_MODEL` or `--model` selects the vision model, default `qwen/qwen3.8-27b`. Up to three images are attached with text and structured table data. It validates source labels and returns answers, page citations, and selected image paths; a UI is still needed to display images. Generated claims and subject identification require review. Saved JSON includes only cited sources and selected images. `--dry-run` validates retrieval inputs without API calls. Existing output files are refused.

### Search your PDF

```powershell
python -m rag.retrieve_documents "Give me a frog image" --type image --dry-run
python -m rag.retrieve_documents "Give me a frog image" --type image --top-k 3
python -m rag.retrieve_documents "What are amphibians?" --output search_results.json
```

Retrieval reads model, dimension, index, and namespace from `embedding_checkpoint.json`. It scopes results to the prepared PDF and joins Pinecone IDs to original local records. `--type` restricts results to text, images, or tables. `--min-score` is an optional cutoff requiring calibration; similarity scores do not verify subject identity. Output includes original text/table data, source pages, and absolute local evidence paths. This script returns search results; it does not generate an answer or display images in a UI. `--output` saves complete JSON without overwriting an existing file. Use `--prepared`, `--evidence-root`, and `--checkpoint` to select other files.

### Embed and upload to Pinecone

```powershell
python -m ingestion.embed_documents --dry-run
python -m ingestion.embed_documents --limit 1
python -m ingestion.embed_documents
```

Reads `prepared_documents_captioned.jsonl` by default. Uses `GOOGLE_API_KEY` (or `GEMINI_API_KEY`) and `PINECONE_API_KEY` from `.env`. Gemini Embedding 2 produces one 1536-dimensional vector per record, including combined image/text inputs. An absent `pdf-multimodal` Pinecone index is created as cosine serverless on AWS us-east-1. Override with `--index` or `PINECONE_INDEX`; namespace defaults to `gemini-embedding-2-1536` and can be overridden with `--namespace` or `PINECONE_NAMESPACE`.

Actual runs call Gemini and write to Pinecone, potentially incurring costs. Checkpoints cache vectors before upload and record successful uploads, so retries do not repeat completed work. Source changes, target changes, or recreated indexes require a new checkpoint. Run only one writer per checkpoint. Local evidence remains on disk; Pinecone stores vectors and source metadata. Default request pacing is 15 seconds. Long quota waits stop with a resume message. Dry runs make no API calls. Use the same embedding model/dimensions for queries; text retrieval queries should use Gemini's `task: search result | query: ...` prefix.

### Generate image captions

Requests are spaced at least 15 seconds apart by default (`--interval` adjusts this). Quota retries wait at least 60 seconds and respect longer server retry hints. Saved captions are skipped on restart. Pacing does not resolve daily quotas or billing limits.

```powershell
python -m pip install -r requirements.txt
python -m ingestion.generate_captions --dry-run
python -m ingestion.generate_captions --limit 3
python -m ingestion.generate_captions
```

Set `GROQ_API_KEY` in `.env`. `GROQ_CAPTION_MODEL` or `--model` selects a vision generation model; the default is `qwen/qwen3.8-27b`, following Groq's vision documentation. Caption generation is separate from Gemini Embedding 2. Actual runs send image bytes and nearby text to Groq and may incur API charges. Dry runs only validate local evidence. Existing Gemini captions are retained and skipped on resume.

`captions.json` maps image record IDs to generated captions. Each successful caption is saved atomically; subsequent runs skip saved IDs. Review captions and decorative/multi-subject crops before embedding. The script does not segment images or modify prepared records. Avoid concurrent runs writing the same caption file. If prepared records were moved, pass `--evidence-root` pointing to their source manifest directory.

After reviewing captions, prepare a new output rather than overwriting the existing records:

```powershell
python -m ingestion.prepare_documents --captions storage/animal_kingdom_5eed592711d3/captions.json --output storage/animal_kingdom_5eed592711d3/prepared_documents_captioned.jsonl
```

API usage reference: https://console.groq.com/docs/vision

Prepare records locally (no API calls):

```powershell
python -m ingestion.prepare_documents
# Or choose a specific extraction:
python -m ingestion.prepare_documents storage/animal_kingdom_5eed592711d3/manifest.json
```

Creates `prepared_documents.jsonl` next to the manifest. Text is grouped into page-local chunks of 400 words with 50-word overlap (word counts, not token counts). Each image/table has one multimodal `embedding_input` containing its image path and accompanying text. All evidence paths resolve relative to the source manifest directory, even when `--output` points elsewhere. Files are validated before writing; existing output is refused.

Images without captions are marked `needs_review`; nearby text is unverified context. To supply reviewed or separately generated captions, pass `--captions captions.json`, where JSON maps image `element_id` values to caption strings. This script does not generate captions or split multi-animal crops. Table records retain structured rows and require review of detector results. Full-page screenshots remain linked as fallback evidence rather than separate embedding records.

Review extracted evidence, then embed text blocks and image/caption pairs separately with Gemini Embedding 2. Use table screenshots plus headers for retrieval, retaining structured rows for exact answers. Upload vectors to Pinecone with the manifest's source IDs and paths.
