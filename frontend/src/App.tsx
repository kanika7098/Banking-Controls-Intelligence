import { useState } from "react";
import {
  Activity,
  ArrowUpRight,
  CheckCircle2,
  CircleHelp,
  Database,
  FileSearch,
  FileText,
  LoaderCircle,
  MessageSquareText,
  Network,
  Search,
  ShieldCheck,
  Upload,
  XCircle,
} from "lucide-react";
import "./App.css";

const API = "http://127.0.0.1:8000";

type Citation = {
  document_id?: string;
  filename?: string;
  page_number?: number | string;
  page?: number | string;
  text?: string;
  content?: string;
  score?: number;
};

type ApiResult = {
  filename?: string;
  chunks_created?: number;
  vectors_indexed?: number;
  answer?: string;
  answer_text?: string;
  response?: string;
  grounded?: boolean;
  citations?: Citation[];
  retrieved_chunks?: number;
  results?: Citation[];
  chunks?: Citation[];
  message?: string;
  detail?: string;
  [key: string]: unknown;
};

export default function App() {
  const [query, setQuery] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [answer, setAnswer] = useState<ApiResult | null>(null);
  const [searchResults, setSearchResults] = useState<Citation[]>([]);
  const [uploadResult, setUploadResult] = useState<ApiResult | null>(null);
  const [loadingAsk, setLoadingAsk] = useState(false);
  const [loadingSearch, setLoadingSearch] = useState(false);
  const [loadingUpload, setLoadingUpload] = useState(false);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<"ask" | "search">("ask");

  async function askQuestion(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!query.trim()) return;
    setLoadingAsk(true);
    setError("");
    setAnswer(null);
    try {
      const response = await fetch(`${API}/api/v1/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ query: query.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
      setAnswer(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Could not connect to the API.");
    } finally {
      setLoadingAsk(false);
    }
  }

  async function runSearch(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!searchQuery.trim()) return;
    setLoadingSearch(true);
    setError("");
    setSearchResults([]);
    try {
      const response = await fetch(`${API}/api/v1/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "application/json" },
        body: JSON.stringify({ query: searchQuery.trim() }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Search failed (${response.status})`);
      setSearchResults(Array.isArray(data) ? data : data.results || data.chunks || []);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Search request failed.");
    } finally {
      setLoadingSearch(false);
    }
  }

  async function uploadDocument(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (!file) return;
    setLoadingUpload(true);
    setError("");
    setUploadResult(null);
    try {
      const body = new FormData();
      body.append("file", file);
      const response = await fetch(`${API}/api/v1/documents/upload`, {
        method: "POST",
        body,
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || `Upload failed (${response.status})`);
      setUploadResult(data);
      setFile(null);
      const input = document.getElementById("pdf-upload") as HTMLInputElement | null;
      if (input) input.value = "";
    } catch (e) {
      setError(e instanceof Error ? e.message : "Document upload failed.");
    } finally {
      setLoadingUpload(false);
    }
  }
  const citations: Citation[] = Array.isArray(answer?.citations) ? answer.citations : Array.isArray(answer?.results) ? answer.results : [];
  const answerText: string = String(answer?.answer ?? answer?.answer_text ?? answer?.response ?? "No answer text was returned.");
  const grounded = answer?.grounded;

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={23} /></div>
          <div><strong>ControlLens</strong><span>INTELLIGENCE PLATFORM</span></div>
        </div>

        <div className="workspace-label">WORKSPACE</div>
        <button className="nav-item selected" onClick={() => setActiveTab("ask")}>
          <MessageSquareText size={18} /> Research assistant
        </button>
        <button className="nav-item" onClick={() => setActiveTab("search")}>
          <FileSearch size={18} /> Semantic search
        </button>
        <a className="nav-item" href="#documents">
          <FileText size={18} /> Knowledge documents
        </a>

        <div className="sidebar-bottom">
          <div className="secure-note"><ShieldCheck size={17} /><div><strong>Evidence-first AI</strong><span>Answers linked to source material</span></div></div>
          <div className="sidebar-footer">BANKING CONTROLS INTELLIGENCE<br />LOCAL DEVELOPMENT ENVIRONMENT</div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">Workspace <span>/</span> <strong>{activeTab === "ask" ? "Research assistant" : "Semantic search"}</strong></div>
          <div className="environment"><span className="status-dot" /> Local environment</div>
        </header>

        <section className="page-heading">
          <div>
            <div className="eyebrow"><span className="eyebrow-line" /> BANKING RISK & COMPLIANCE</div>
            <h1>Controls intelligence,<br className="desktop-break" /> grounded in evidence.</h1>
            <p>Explore control documents, retrieve relevant evidence, and ask questions with source-linked answers.</p>
          </div>
          <div className="heading-icon"><Network size={31} strokeWidth={1.4} /><span>RAG ENGINE</span></div>
        </section>

        <section className="metrics-grid">
          <div className="metric-card"><div className="metric-icon purple"><Database size={19} /></div><span className="metric-label">KNOWLEDGE BASE</span><strong>FAISS</strong><small>Vector similarity index</small></div>
          <div className="metric-card"><div className="metric-icon blue"><Activity size={19} /></div><span className="metric-label">EMBEDDINGS</span><strong>384 dimensions</strong><small>Sentence Transformers</small></div>
          <div className="metric-card"><div className="metric-icon green"><ShieldCheck size={19} /></div><span className="metric-label">ANSWER METHOD</span><strong>Evidence-first RAG</strong><small>Local Ollama model</small></div>
        </section>

        <div className="work-grid">
          <section className="panel research-panel">
            <div className="panel-heading">
              <div className="panel-icon"><MessageSquareText size={19} /></div>
              <div><h2>Research assistant</h2><p>Ask questions grounded in your indexed documents.</p></div>
            </div>
            <div className="tabs">
              <button className={activeTab === "ask" ? "tab active" : "tab"} onClick={() => { setActiveTab("ask"); setError(""); }}>Ask a question</button>
              <button className={activeTab === "search" ? "tab active" : "tab"} onClick={() => { setActiveTab("search"); setError(""); }}>Search evidence</button>
            </div>

            {activeTab === "ask" ? (
              <form onSubmit={askQuestion}>
                <label className="field-label" htmlFor="question">YOUR QUESTION</label>
                <textarea id="question" value={query} onChange={(e) => setQuery(e.target.value)} placeholder="e.g. What cybersecurity controls should banks implement?" rows={4} />
                <div className="form-foot"><span><ShieldCheck size={14} /> Answers are checked against retrieved evidence</span><button className="primary-button" type="submit" disabled={loadingAsk || !query.trim()}>{loadingAsk ? <LoaderCircle className="spin" size={16} /> : <MessageSquareText size={16} />} Ask question <ArrowUpRight size={15} /></button></div>
              </form>
            ) : (
              <form onSubmit={runSearch}>
                <label className="field-label" htmlFor="search-query">SEARCH YOUR DOCUMENTS</label>
                <textarea id="search-query" value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} placeholder="Describe the control, policy, or evidence you want to find..." rows={4} />
                <div className="form-foot"><span><Search size={14} /> Semantic retrieval from indexed chunks</span><button className="primary-button" type="submit" disabled={loadingSearch || !searchQuery.trim()}>{loadingSearch ? <LoaderCircle className="spin" size={16} /> : <Search size={16} />} Search evidence <ArrowUpRight size={15} /></button></div>
              </form>
            )}

            {error && <div className="error-box"><XCircle size={18} /><div><strong>Request unsuccessful</strong><p>{error}</p><small>Check that the backend is running at {API}.</small></div></div>}

            {answer && activeTab === "ask" && <div className="result-block">
              <div className="result-heading"><h3>Generated answer</h3><span className={grounded === false ? "result-badge warning" : "result-badge"}>{grounded === false ? "Insufficient evidence" : grounded === true ? "Evidence grounded" : "API response"}</span></div>
              <p className="answer-text">{answerText}</p>
              <div className="result-meta"><span><FileText size={14} /> {citations.length} citation(s)</span><span>{answer.retrieved_chunks ?? 0} chunks retrieved</span></div>
              {citations.length > 0 && <div className="citation-list"><h4>Source citations</h4>{citations.map((c: Citation, i: number) => <div className="citation" key={`${c.document_id || c.filename || "source"}-${c.page_number || c.page || i}-${i}`}><div className="citation-number">{i + 1}</div><div><strong>{c.filename || c.document_id || "Indexed document"}</strong><span>Page {c.page_number ?? c.page ?? "N/A"}</span>{(c.text || c.content) && <p>{c.text || c.content}</p>}</div></div>)}</div>}
            </div>}

            {activeTab === "search" && searchResults.length > 0 && <div className="result-block"><div className="result-heading"><h3>Retrieved evidence</h3><span className="result-badge">{searchResults.length} result(s)</span></div>{searchResults.map((c, i) => <div className="citation search-result" key={i}><div className="citation-number">{i + 1}</div><div><strong>{c.filename || c.document_id || "Indexed document"}</strong><span>Page {c.page_number ?? c.page ?? "N/A"}</span><p>{c.text || c.content || "No text field was returned. Inspect the raw API response to confirm the schema."}</p></div></div>)}</div>}

            {!answer && searchResults.length === 0 && !error && <div className="empty-state"><div className="empty-icon"><CircleHelp size={22} /></div><strong>Your research starts here</strong><p>Submit a question to generate an answer or search for evidence from indexed documents.</p></div>}
          </section>

          <aside className="right-column">
            <section className="panel upload-panel" id="documents">
              <div className="panel-heading">
                <div className="panel-icon upload-icon"><Upload size={19} /></div>
                <div><h2>Knowledge documents</h2><p>Add a PDF to your research index.</p></div>
              </div>
              <form onSubmit={uploadDocument}>
                <label className="upload-zone" htmlFor="pdf-upload"><Upload size={23} /><strong>{file ? file.name : "Choose a PDF document"}</strong><span>{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB selected` : "PDF format · Maximum 10 MB"}</span><input id="pdf-upload" type="file" accept=".pdf,application/pdf" onChange={(e) => setFile(e.target.files?.[0] || null)} /></label>
                <button className="primary-button full-button" type="submit" disabled={!file || loadingUpload}>{loadingUpload ? <LoaderCircle className="spin" size={16} /> : <Upload size={16} />}{loadingUpload ? "Processing document..." : "Upload and index"}</button>
              </form>
              {uploadResult && <div className="success-box"><CheckCircle2 size={17} /><div><strong>Document indexed</strong><p>{uploadResult.filename || "PDF uploaded"} · {String(uploadResult.chunks_created ?? "—")} chunks created · {String(uploadResult.vectors_indexed ?? "—")} vectors indexed</p></div></div>}
              <div className="upload-note"><ShieldCheck size={15} /><span>Documents are processed by your local backend. Only upload files you are authorized to use.</span></div>
            </section>

            <section className="panel pipeline-panel">
              <div className="panel-heading"><div className="panel-icon"><Network size={19} /></div><div><h2>Processing pipeline</h2><p>From document to grounded answer.</p></div></div>
              <div className="pipeline">
                <div className="pipeline-step"><span className="pipeline-number">01</span><div><strong>Document ingestion</strong><p>PDF extraction and OCR fallback</p></div><CheckCircle2 size={16} /></div>
                <div className="pipeline-connector" />
                <div className="pipeline-step"><span className="pipeline-number">02</span><div><strong>Chunking & embeddings</strong><p>Overlapping text chunks · 384D</p></div><CheckCircle2 size={16} /></div>
                <div className="pipeline-connector" />
                <div className="pipeline-step"><span className="pipeline-number">03</span><div><strong>Vector retrieval</strong><p>FAISS similarity search</p></div><CheckCircle2 size={16} /></div>
                <div className="pipeline-connector" />
                <div className="pipeline-step"><span className="pipeline-number">04</span><div><strong>Local LLM generation</strong><p>Ollama · source-grounded response</p></div><CheckCircle2 size={16} /></div>
              </div>
            </section>
          </aside>
        </div>

        <footer className="page-footer"><span>CONTROLLENS <span className="footer-dot">•</span> BANKING CONTROLS INTELLIGENCE</span><span>Local prototype <span className="footer-dot">•</span> Verify source documents before operational use</span></footer>
      </main>
    </div>
  );
}
