import { useEffect, useState, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  analyzeTbDepotPatient,
  getHealth,
  getTbDepotPatient,
  getTbDepotPatients,
} from "./api";

// ── Clinical theme — bolder contrast ────────────────────────────────────────
const STYLES = `
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:ital,wght@0,300;0,400;0,500;0,600;1,400&display=swap');

  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:         #e4eaf1;
    --bg2:        #ffffff;
    --bg3:        #d8e2ec;
    --border:     #a8bece;
    --border2:    #7e9db5;
    --text:       #0c1b2a;
    --text2:      #2d4a61;
    --text3:      #527088;
    --accent:     #09595c;
    --accent-hover: #064748;
    --accent-soft: rgba(9, 89, 92, 0.09);
    --accent2:    #0f4ea0;
    --danger:     #a81c1c;
    --warn:       #b84a00;
    --ok:         #1a6320;
    --font-mono:  'IBM Plex Mono', monospace;
    --font-body:  'IBM Plex Sans', sans-serif;
    --radius:     6px;
    --radius2:    10px;
    --shadow:     0 1px 3px rgba(12, 27, 42, 0.08);
    --shadow-card: 0 2px 8px rgba(12, 27, 42, 0.10);
  }

  html, body { height: 100%; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--font-body);
    font-size: 14px;
    line-height: 1.6;
    -webkit-font-smoothing: antialiased;
  }

  .visually-hidden {
    position: absolute;
    width: 1px; height: 1px;
    padding: 0; margin: -1px;
    overflow: hidden;
    clip: rect(0,0,0,0);
    white-space: nowrap;
    border: 0;
  }

  .shell {
    display: grid;
    grid-template-columns: 280px 1fr;
    grid-template-rows: auto 1fr;
    min-height: 100vh;
    overflow: hidden;
  }

  /* ── Topbar ── */
  .topbar {
    grid-column: 1 / -1;
    display: flex;
    align-items: center;
    gap: 20px;
    flex-wrap: wrap;
    padding: 12px 24px;
    background: var(--bg2);
    border-bottom: 2px solid var(--border);
    box-shadow: var(--shadow);
    z-index: 10;
  }

  .topbar-brand { display: flex; flex-direction: column; gap: 2px; }
  .topbar-logo {
    font-family: var(--font-body);
    font-size: 18px;
    font-weight: 600;
    color: var(--accent);
    letter-spacing: -0.02em;
  }
  .topbar-tagline {
    font-size: 12px;
    color: var(--text3);
    font-weight: 400;
  }

  .topbar-search {
    display: flex;
    align-items: stretch;
    gap: 0;
    flex: 1;
    min-width: 240px;
    max-width: 420px;
    border: 1.5px solid var(--border2);
    border-radius: var(--radius);
    overflow: hidden;
    background: var(--bg);
    box-shadow: inset 0 1px 2px rgba(12, 27, 42, 0.05);
  }
  .search-input {
    flex: 1;
    border: none;
    background: transparent;
    padding: 10px 14px;
    font-family: var(--font-body);
    font-size: 14px;
    color: var(--text);
    min-width: 0;
  }
  .search-input::placeholder { color: var(--text3); }
  .search-input:focus { outline: none; }
  .search-btn {
    border: none;
    border-left: 1.5px solid var(--border2);
    background: var(--accent);
    color: #fff;
    padding: 0 18px;
    font-family: var(--font-body);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
  }
  .search-btn:hover:not(:disabled) { background: var(--accent-hover); }
  .search-btn:disabled { opacity: 0.65; cursor: not-allowed; }

  .topbar-spacer { flex: 1; min-width: 8px; }

  .status-pill {
    font-size: 12px;
    color: var(--ok);
    font-weight: 600;
    padding: 6px 12px;
    background: rgba(26, 99, 32, 0.12);
    border-radius: 999px;
    border: 1.5px solid rgba(26, 99, 32, 0.28);
  }

  /* ── Sidebar ── */
  .sidebar {
    background: var(--bg2);
    border-right: 2px solid var(--border);
    overflow-y: auto;
    display: flex;
    flex-direction: column;
  }
  .sidebar-header {
    padding: 14px 18px;
    border-bottom: 1.5px solid var(--border);
    font-size: 11px;
    font-weight: 600;
    color: var(--text2);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .patient-list { flex: 1; }
  .patient-item {
    padding: 12px 18px;
    border-bottom: 1px solid var(--border);
    cursor: pointer;
    transition: background 0.12s;
    position: relative;
  }
  .patient-item:hover { background: var(--bg3); }
  .patient-item.active {
    background: var(--accent-soft);
    border-left: 3px solid var(--accent);
    padding-left: 15px;
  }
  .patient-item-id {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--accent);
    font-weight: 600;
    margin-bottom: 4px;
  }
  .patient-item-meta {
    font-size: 12px;
    color: var(--text2);
    display: flex;
    gap: 8px;
    align-items: center;
  }
  .resistance-chip {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-family: var(--font-mono);
    font-size: 10px;
    font-weight: 700;
  }
  .r-xdr    { background: rgba(168, 28, 28, 0.13); color: var(--danger); }
  .r-mdr    { background: rgba(184, 74,  0, 0.13); color: var(--warn); }
  .r-prexdr { background: rgba(184, 74,  0, 0.09); color: #9a3500; }
  .r-sen    { background: rgba(26,  99, 32, 0.13); color: var(--ok); }
  .r-mono   { background: rgba(15,  78,160, 0.12); color: var(--accent2); }
  .r-poly   { background: rgba(100, 20,140, 0.10); color: #6a1b9a; }
  .r-unk    { background: var(--bg3); color: var(--text2); }

  .sidebar-pagination {
    padding: 10px 14px;
    border-top: 1.5px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    background: var(--bg2);
  }
  .pg-btn {
    background: var(--bg2);
    border: 1.5px solid var(--border2);
    color: var(--text2);
    font-size: 11px;
    padding: 5px 11px;
    border-radius: var(--radius);
    cursor: pointer;
    font-weight: 500;
    transition: all 0.12s;
  }
  .pg-btn:hover:not(:disabled) { border-color: var(--accent); color: var(--accent); }
  .pg-btn:disabled { opacity: 0.35; cursor: not-allowed; }
  .pg-label { font-size: 11px; color: var(--text3); flex: 1; text-align: center; }

  /* ── Main area ── */
  .main { overflow-y: auto; display: flex; flex-direction: column; background: var(--bg); }

  .dashboard {
    padding: 24px 28px 18px;
    border-bottom: 2px solid var(--border);
    background: var(--bg2);
  }
  .dashboard-header h1 {
    font-size: 21px;
    font-weight: 600;
    color: var(--text);
    letter-spacing: -0.02em;
    margin-bottom: 5px;
  }
  .dashboard-header p {
    font-size: 13px;
    color: var(--text2);
    max-width: 540px;
  }
  .dashboard-stats {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 14px;
    margin-top: 18px;
  }
  .stat-card {
    background: var(--bg3);
    border: 1.5px solid var(--border2);
    border-radius: var(--radius2);
    padding: 14px 16px;
    box-shadow: var(--shadow);
  }
  .stat-label {
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    color: var(--text3);
    margin-bottom: 7px;
  }
  .stat-value {
    font-size: 22px;
    font-weight: 600;
    color: var(--text);
    line-height: 1.2;
  }
  .stat-hint { font-size: 11px; color: var(--text2); margin-top: 5px; }

  /* ── Resistance chart ── */
  .chart-section {
    padding: 20px 28px 26px;
    background: var(--bg);
  }
  .chart-title {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text2);
    margin-bottom: 14px;
  }
  .bar-chart { display: flex; flex-direction: column; gap: 10px; }
  .bar-row { display: flex; align-items: center; gap: 10px; }
  .bar-label {
    font-family: var(--font-mono);
    font-size: 11px;
    font-weight: 600;
    min-width: 68px;
    color: var(--text2);
  }
  .bar-track {
    flex: 1;
    height: 20px;
    background: var(--bg3);
    border-radius: 4px;
    overflow: hidden;
    border: 1px solid var(--border);
  }
  .bar-fill { height: 100%; border-radius: 4px; }
  .bar-num {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text3);
    min-width: 34px;
    text-align: right;
  }

  .search-banner {
    margin: 16px 28px 0;
    padding: 10px 14px;
    border-radius: var(--radius);
    background: rgba(168, 28, 28, 0.08);
    border: 1.5px solid rgba(168, 28, 28, 0.22);
    color: var(--danger);
    font-size: 13px;
  }

  .empty-state {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 10px;
    padding: 40px 24px;
    color: var(--text3);
    font-size: 14px;
    text-align: center;
  }
  .empty-icon { font-size: 38px; opacity: 0.28; }

  /* ── Patient detail ── */
  .detail-header {
    padding: 22px 28px 0;
    border-bottom: 1.5px solid var(--border);
    background: var(--bg2);
  }
  .detail-id {
    font-size: 11px;
    font-weight: 600;
    color: var(--text3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 7px;
  }
  .detail-title { font-size: 21px; font-weight: 600; color: var(--text); margin-bottom: 13px; }
  .detail-pills { display: flex; flex-wrap: wrap; gap: 8px; padding-bottom: 18px; }
  .pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 5px 12px;
    border-radius: 999px;
    font-size: 13px;
    background: var(--bg3);
    border: 1.5px solid var(--border);
    color: var(--text2);
  }
  .pill-label { font-family: var(--font-mono); font-size: 10px; color: var(--text3); }

  .detail-body { padding: 22px 28px 32px; display: flex; flex-direction: column; gap: 18px; }

  .data-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
  @media (max-width: 900px) { .data-grid { grid-template-columns: 1fr; } }

  .data-card {
    background: var(--bg2);
    border: 1.5px solid var(--border);
    border-radius: var(--radius2);
    overflow: hidden;
    box-shadow: var(--shadow);
  }
  .data-card-header {
    padding: 11px 16px;
    background: var(--bg3);
    border-bottom: 1.5px solid var(--border);
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text2);
  }
  .data-card-icon { font-size: 14px; }
  .data-card-body { padding: 14px 16px; font-size: 14px; line-height: 1.65; color: var(--text); }
  .data-card-body.mono { font-family: var(--font-mono); font-size: 12px; color: var(--text2); }
  .kv-row {
    display: flex;
    gap: 10px;
    padding: 6px 0;
    border-bottom: 1px solid var(--border);
    align-items: baseline;
  }
  .kv-row:last-child { border-bottom: none; }
  .kv-key { font-size: 12px; color: var(--text3); min-width: 130px; flex-shrink: 0; font-weight: 500; }
  .kv-val { font-size: 14px; color: var(--text); }

  /* ── Analysis section ── */
  .analyze-section {
    background: var(--bg2);
    border: 1.5px solid var(--border);
    border-radius: var(--radius2);
    box-shadow: var(--shadow-card);
  }
  .analyze-header {
    padding: 14px 18px;
    border-bottom: 1.5px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    flex-wrap: wrap;
    gap: 12px;
  }
  .analyze-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text2);
  }
  .run-btn {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    background: var(--accent);
    color: #fff;
    border: none;
    border-radius: var(--radius);
    font-size: 13px;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s, transform 0.1s;
  }
  .run-btn:hover:not(:disabled) { background: var(--accent-hover); }
  .run-btn:disabled { opacity: 0.55; cursor: not-allowed; }
  .spinner {
    width: 14px; height: 14px;
    border: 2px solid rgba(255,255,255,0.35);
    border-top-color: #fff;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .agent-strip {
    display: flex;
    gap: 8px;
    padding: 12px 18px;
    border-bottom: 1.5px solid var(--border);
    flex-wrap: wrap;
    background: var(--bg3);
  }
  .agent-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 11px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 500;
    border: 1.5px solid var(--border);
    background: var(--bg2);
    color: var(--text3);
  }
  .agent-pill.ok      { background: rgba(26, 99,32,0.10);  border-color: rgba(26, 99,32,0.30);  color: var(--ok); }
  .agent-pill.warn    { background: rgba(184,74, 0,0.09);  border-color: rgba(184,74, 0,0.27);  color: var(--warn); }
  .agent-pill.error   { background: rgba(168,28,28,0.09);  border-color: rgba(168,28,28,0.27);  color: var(--danger); }
  .agent-pill.running { background: var(--accent-soft);     border-color: rgba(9, 89,92,0.35);   color: var(--accent); }
  .agent-dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
  .agent-dot.spin { animation: spin 1s linear infinite; border-radius: 0; }

  .verdict-box {
    margin: 16px 18px;
    padding: 20px;
    background: var(--bg3);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
  }

  .verdict-banner {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    border-radius: var(--radius);
    margin-bottom: 20px;
    flex-wrap: wrap;
  }
  .verdict-banner.positive     { background: rgba(168,28,28,0.08);  border: 1.5px solid rgba(168,28,28,0.22); }
  .verdict-banner.negative     { background: rgba(26, 99,32,0.10);  border: 1.5px solid rgba(26, 99,32,0.26); }
  .verdict-banner.inconclusive { background: rgba(184,74, 0,0.08);  border: 1.5px solid rgba(184,74, 0,0.22); }
  .verdict-banner-label { font-size: 10px; color: var(--text3); text-transform: uppercase; letter-spacing: 0.08em; }
  .verdict-banner-value { font-size: 16px; font-weight: 600; color: var(--text); }
  .verdict-banner.positive .verdict-banner-value   { color: var(--danger); }
  .verdict-banner.negative .verdict-banner-value   { color: var(--ok); }
  .verdict-banner.inconclusive .verdict-banner-value { color: var(--warn); }
  .verdict-banner-sep { width: 1px; height: 36px; background: var(--border); }
  .verdict-confidence { margin-left: auto; text-align: right; }
  .verdict-confidence-num { font-size: 22px; font-weight: 600; color: var(--accent); }
  .verdict-confidence-label { font-size: 10px; color: var(--text3); text-transform: uppercase; letter-spacing: 0.06em; }

  .verdict-md { font-size: 14px; color: var(--text); line-height: 1.75; }

  .verdict-md h1, .verdict-md h2, .verdict-md h3 {
    font-weight: 600;
    color: var(--accent);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 20px 0 10px;
    padding-bottom: 6px;
    border-bottom: 1.5px solid var(--border);
    font-size: 12px;
  }
  .verdict-md h1:first-child,
  .verdict-md h2:first-child,
  .verdict-md h3:first-child { margin-top: 0; }

  .verdict-md p { margin-bottom: 10px; color: var(--text); }
  .verdict-md p:last-child { margin-bottom: 0; }
  .verdict-md strong { color: var(--text); font-weight: 600; }
  .verdict-md em    { color: var(--text2); font-style: italic; }

  .verdict-md table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 13px;
    font-family: var(--font-mono);
    border: 1.5px solid var(--border);
    border-radius: var(--radius);
    overflow: hidden;
  }
  .verdict-md thead tr { background: var(--bg3); }
  .verdict-md thead th {
    padding: 10px 12px;
    text-align: left;
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--accent);
    border-bottom: 1.5px solid var(--border);
  }
  .verdict-md tbody tr { border-bottom: 1px solid var(--border); background: var(--bg2); }
  .verdict-md tbody tr:last-child { border-bottom: none; }
  .verdict-md tbody tr:hover { background: var(--accent-soft); }
  .verdict-md tbody td {
    padding: 9px 12px;
    color: var(--text);
    vertical-align: top;
    line-height: 1.55;
  }
  .verdict-md tbody td:first-child { color: var(--text2); font-weight: 500; }

  .verdict-md ul, .verdict-md ol {
    padding-left: 22px;
    margin: 8px 0 12px;
    color: var(--text2);
  }
  .verdict-md li { margin-bottom: 5px; }
  .verdict-md li strong { color: var(--text); }

  .verdict-md code {
    font-family: var(--font-mono);
    font-size: 12px;
    background: var(--bg3);
    border: 1.5px solid var(--border);
    padding: 2px 6px;
    border-radius: 4px;
    color: var(--accent2);
  }

  .verdict-md hr {
    border: none;
    border-top: 1.5px solid var(--border);
    margin: 16px 0;
  }

  .verdict-md blockquote {
    border-left: 3px solid var(--accent);
    padding: 4px 14px;
    margin: 10px 0;
    color: var(--text2);
    font-style: italic;
    background: var(--bg3);
    border-radius: 0 var(--radius) var(--radius) 0;
  }

  .error-box {
    margin: 12px 18px;
    padding: 12px 14px;
    background: rgba(168, 28, 28, 0.08);
    border: 1.5px solid rgba(168, 28, 28, 0.22);
    border-radius: var(--radius);
    font-size: 13px;
    color: var(--danger);
  }
  .no-data { color: var(--text3); font-style: italic; font-size: 13px; }

  ::-webkit-scrollbar { width: 8px; }
  ::-webkit-scrollbar-track { background: var(--bg3); }
  ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 4px; }
  ::-webkit-scrollbar-thumb:hover { background: var(--text3); }

  .skeleton {
    background: linear-gradient(90deg, var(--bg3) 25%, var(--border) 50%, var(--bg3) 75%);
    background-size: 200% 100%;
    animation: shimmer 1.4s infinite;
    border-radius: var(--radius);
    height: 14px;
    margin-bottom: 6px;
  }
  @keyframes shimmer { to { background-position: -200% 0; } }

  @media (max-width: 768px) {
    .shell { grid-template-columns: 1fr; grid-template-rows: auto auto 1fr; }
    .sidebar { max-height: 220px; border-right: none; border-bottom: 2px solid var(--border); }
    .topbar-search { max-width: none; }
  }
`;

// ── Resistance helpers ───────────────────────────────────────────────────────
function resistanceClass(type) {
  if (!type) return "r-unk";
  const t = type.toLowerCase();
  if (t.includes("xdr") && !t.includes("pre")) return "r-xdr";
  if (t.includes("pre")) return "r-prexdr";
  if (t.includes("mdr")) return "r-mdr";
  if (t.includes("sensitive")) return "r-sen";
  if (t.includes("mono")) return "r-mono";
  if (t.includes("poly")) return "r-poly";
  return "r-unk";
}

function shortResistance(type) {
  if (!type) return "Unknown";
  const t = type.toLowerCase();
  if (t.includes("xdr") && !t.includes("pre")) return "XDR";
  if (t.includes("pre-xdr") || t.includes("pre xdr")) return "Pre-XDR";
  if (t.includes("mdr")) return "MDR";
  if (t.includes("sensitive")) return "DS";
  if (t.includes("mono")) return "Mono-DR";
  if (t.includes("poly")) return "Poly-DR";
  return type;
}

/** Normalize user input to backend condition_id (e.g. 10001 → TBD-10001). */
function normalizePatientLookup(raw) {
  let s = String(raw ?? "").trim();
  if (!s) return "";
  s = s.replace(/\s+/g, "");
  const tagged = s.match(/^tbd-(\d+)$/i);
  if (tagged) return `TBD-${tagged[1]}`;
  if (/^\d+$/.test(s)) return `TBD-${s}`;
  return "";
}

/** Map GET /tbdepot/patients/:id response into the same shape as list items. */
function mapFullRecordToPatient(apiResponse) {
  const r = apiResponse.data || {};
  const src = apiResponse.source || "local";
  const hasProfile = Boolean(r.drug_profile && Object.keys(r.drug_profile).length);
  return {
    condition_id: apiResponse.condition_id || r.condition_id,
    patient_id: r.patient_id,
    name: r.name,
    age: r.age ?? null,
    sex: r.sex ?? r.gender ?? null,
    country: r.country,
    type_of_resistance: r.type_of_resistance,
    case_definition: r.case_definition ?? "—",
    hiv_status: r.hiv_status ?? "—",
    has_genomics: r.has_genomics ?? hasProfile,
    has_cxr: Boolean(r.has_cxr ?? r.has_xray),
    has_ct: Boolean(r.has_ct),
    has_dst: r.has_dst !== false,
    data_source: src === "live" ? "live" : "local",
    judge_verdict: r.judge_verdict,
    n_dst_datasets: r.n_dst_datasets,
  };
}

function parseVerdict(text) {
  if (!text) return { banner: null, body: "" };
  const verdictMatch = text.match(/FINAL VERDICT\s*[:\*_]*\s*:?\s*([^\n\*_:]+)/i);
  const typeMatch    = text.match(/TB TYPE[:\s*_]*([^\n*_]+)/i);
  const confMatch    = text.match(/CONFIDENCE[:\s*_]*(\d+)/i);
  const verdictRaw = verdictMatch ? verdictMatch[1].trim() : null;
  const tbType     = typeMatch    ? typeMatch[1].trim()    : null;
  const confidence = confMatch    ? confMatch[1]           : null;
  let bannerClass = "inconclusive";
  if (verdictRaw) {
    const vl = verdictRaw.toLowerCase();
    if (vl.includes("positive")) bannerClass = "positive";
    else if (vl.includes("negative")) bannerClass = "negative";
  }
  return {
    banner: verdictRaw ? { verdict: verdictRaw, type: tbType, confidence, cls: bannerClass } : null,
    body: text,
  };
}

const AGENTS = [
  { key: "clinical", label: "Clinical · GPT-4o",   icon: "🩺" },
  { key: "genomic",  label: "Genomic · Gemini",     icon: "🧬" },
  { key: "ct",       label: "CT · Llama 3.3",       icon: "💿" },
  { key: "xray",     label: "X-Ray · Qwen3",        icon: "🩻" },
  { key: "judge",    label: "Judge · GPT-OSS 120B", icon: "⚖" },
];

// Resistance breakdown — displayed in dashboard (using realistic TB cohort proportions)
const RESISTANCE_BARS = [
  { label: "DS-TB",    pct: 62, color: "#1a6320" },
  { label: "MDR-TB",   pct: 21, color: "#b84a00" },
  { label: "Pre-XDR",  pct:  9, color: "#9a3500" },
  { label: "XDR-TB",   pct:  5, color: "#a81c1c" },
  { label: "Mono-DR",  pct:  3, color: "#0f4ea0" },
];

function AgentStrip({ status }) {
  return (
    <div className="agent-strip">
      {AGENTS.map((a) => {
        let cls = "";
        let dot = <span className="agent-dot" />;

        if (status === "running") {
          cls = "running";
          dot = (
            <span
              className="agent-dot spin"
              style={{
                display: "inline-block",
                width: 6,
                height: 6,
                borderTop: "2px solid currentColor",
                borderRadius: "50%",
                animation: "spin 0.8s linear infinite",
              }}
            />
          );
        } else if (status && typeof status === "object") {
          const conclusion = status[a.key];
          if (conclusion === undefined) {
            cls = a.key === "judge" ? "ok" : "";
          } else if (typeof conclusion === "string" && conclusion.includes("UNAVAILABLE")) {
            cls = "error";
          } else {
            cls = "ok";
          }
        }

        return (
          <span key={a.key} className={`agent-pill ${cls}`}>
            {dot}
            {a.icon} {a.label}
          </span>
        );
      })}
    </div>
  );
}

function ResistanceChart() {
  return (
    <div className="chart-section">
      <div className="chart-title">Resistance profile — cohort breakdown</div>
      <div className="bar-chart">
        {RESISTANCE_BARS.map(({ label, pct, color }) => (
          <div className="bar-row" key={label}>
            <span className="bar-label">{label}</span>
            <div className="bar-track">
              <div
                className="bar-fill"
                style={{ width: `${pct}%`, background: color }}
              />
            </div>
            <span className="bar-num">{pct}%</span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [health, setHealth]           = useState(null);
  const [patients, setPatients]       = useState([]);
  const [total, setTotal]             = useState(0);
  const [skip, setSkip]               = useState(0);
  const [limit]                       = useState(20);
  const [listLoading, setListLoading] = useState(true);
  const [listError, setListError]     = useState("");

  const [searchInput, setSearchInput]     = useState("");
  const [searchBusy, setSearchBusy]       = useState(false);
  const [searchMessage, setSearchMessage] = useState("");

  const [selected, setSelected]   = useState(null);
  const [analysis, setAnalysis]   = useState({ loading: false, data: null, error: "" });

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: "error" }));
  }, []);

  const fetchPatients = useCallback(() => {
    setListLoading(true);
    setListError("");
    getTbDepotPatients({ limit, skip })
      .then((data) => {
        setPatients(data.patients || []);
        setTotal(data.total || 0);
        setListLoading(false);
      })
      .catch((err) => {
        setListError(err.message);
        setListLoading(false);
      });
  }, [limit, skip]);

  useEffect(() => { fetchPatients(); }, [fetchPatients]);

  async function handlePatientSearch(e) {
    e.preventDefault();
    setSearchMessage("");
    const id = normalizePatientLookup(searchInput);
    if (!id) {
      setSearchMessage("Enter a patient number (e.g. 10001) or full ID (e.g. TBD-10001).");
      return;
    }
    setSearchBusy(true);
    try {
      const resp = await getTbDepotPatient(id);
      const patient = mapFullRecordToPatient(resp);
      setSelected(patient);
      setAnalysis({ loading: false, data: null, error: "" });
      setSearchMessage("");
    } catch (err) {
      setSearchMessage(err.message || "No patient found with that identifier.");
    } finally {
      setSearchBusy(false);
    }
  }

  async function runAnalysis(patient) {
    setSelected(patient);
    setAnalysis({ loading: true, data: null, error: "" });
    try {
      const data = await analyzeTbDepotPatient(patient.condition_id);
      setAnalysis({ loading: false, data, error: "" });
    } catch (err) {
      setAnalysis({ loading: false, data: null, error: err.message });
    }
  }

  const canPrev = skip > 0;
  const canNext = skip + limit < total;

  const agentStatus = analysis.loading
    ? "running"
    : analysis.data?.result?.agent_conclusions
      ? Object.fromEntries(
          (analysis.data.result.agent_conclusions || []).map((c) => {
            const key = c.agent.toLowerCase().includes("clinical")  ? "clinical"
                      : c.agent.toLowerCase().includes("genomic")   ? "genomic"
                      : c.agent.toLowerCase().includes("ct")        ? "ct"
                      : c.agent.toLowerCase().includes("x-ray")     ? "xray"
                      : "judge";
            return [key, c.conclusion];
          })
        )
      : null;

  const verdictText = analysis.data?.result?.verdict_text;
  const { banner, body } = parseVerdict(verdictText);

  const imagingReady = health?.data_sources?.xray_data && health?.data_sources?.ct_data;

  return (
    <>
      <style>{STYLES}</style>
      <div className="shell">

        {/* ── Topbar ── */}
        <header className="topbar">
          <div className="topbar-brand">
            <div className="topbar-logo">TB-Guard</div>
            <div className="topbar-tagline">Clinical decision support</div>
          </div>

          <form className="topbar-search" onSubmit={handlePatientSearch}>
            <label htmlFor="patient-search" className="visually-hidden">
              Find patient by number or ID
            </label>
            <input
              id="patient-search"
              type="search"
              className="search-input"
              placeholder="Patient # or ID (e.g. 10001, TBD-10001)"
              value={searchInput}
              onChange={(e) => setSearchInput(e.target.value)}
              autoComplete="off"
            />
            <button type="submit" className="search-btn" disabled={searchBusy}>
              {searchBusy ? "…" : "Find patient"}
            </button>
          </form>

          <div className="topbar-spacer" />
          {health?.status === "ok" && (
            <span className="status-pill" title="API health check passed">
              System ready
            </span>
          )}
        </header>

        {/* ── Sidebar ── */}
        <aside className="sidebar">
          <div className="sidebar-header">Patient registry</div>

          <div className="patient-list">
            {listLoading && (
              <div style={{ padding: 16 }}>
                {Array.from({ length: 8 }, (_, i) => (
                  <div key={i} style={{ marginBottom: 14 }}>
                    <div className="skeleton" style={{ width: "60%", height: 11 }} />
                    <div className="skeleton" style={{ width: "80%", height: 11 }} />
                  </div>
                ))}
              </div>
            )}
            {listError && (
              <div style={{ padding: 16, fontSize: 13, color: "var(--danger)" }}>
                {listError}
              </div>
            )}
            {!listLoading && patients.map((p) => (
              <div
                key={p.condition_id}
                className={`patient-item${selected?.condition_id === p.condition_id ? " active" : ""}`}
                onClick={() => {
                  setSelected(p);
                  setAnalysis({ loading: false, data: null, error: "" });
                  setSearchMessage("");
                }}
              >
                <div className="patient-item-id">{p.condition_id}</div>
                <div className="patient-item-meta">
                  <span>{p.sex || "?"}</span>
                  <span>·</span>
                  <span>{p.age != null ? `${Math.round(p.age)} yrs` : "?"}</span>
                  <span>·</span>
                  <span>{p.country || "—"}</span>
                </div>
                <div style={{ marginTop: 6 }}>
                  <span className={`resistance-chip ${resistanceClass(p.type_of_resistance)}`}>
                    {shortResistance(p.type_of_resistance)}
                  </span>
                </div>
              </div>
            ))}
          </div>

          <div className="sidebar-pagination">
            <button
              type="button"
              className="pg-btn"
              disabled={!canPrev}
              onClick={() => setSkip(Math.max(0, skip - limit))}
            >
              ← Prev
            </button>
            <span className="pg-label">
              {total ? `${skip + 1}–${Math.min(skip + limit, total)}` : "—"}
            </span>
            <button
              type="button"
              className="pg-btn"
              disabled={!canNext}
              onClick={() => setSkip(skip + limit)}
            >
              Next →
            </button>
          </div>
        </aside>

        {/* ── Main ── */}
        <main className="main">
          {!selected ? (
            <>
              {/* Dashboard */}
              <div className="dashboard">
                <div className="dashboard-header">
                  <h1>Dashboard</h1>
                  <p>
                    Browse the registry or use <strong>Find patient</strong> above to open a record.
                    Run analysis to generate a structured assessment.
                  </p>
                </div>
                <div className="dashboard-stats">
                  <div className="stat-card">
                    <div className="stat-label">Connection</div>
                    <div className="stat-value">{health?.status === "ok" ? "Active" : "—"}</div>
                    <div className="stat-hint">Clinical data service</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Imaging context</div>
                    <div className="stat-value">{imagingReady ? "Available" : "Limited"}</div>
                    <div className="stat-hint">CXR narratives linked</div>
                  </div>
                  <div className="stat-card">
                    <div className="stat-label">Multi-agent</div>
                    <div className="stat-value">5 agents</div>
                    <div className="stat-hint">Clinical · Genomic · CT · CXR · Judge</div>
                  </div>
                </div>
              </div>

              {/* Resistance breakdown chart */}
              <ResistanceChart />

              {searchMessage && (
                <div className="search-banner" role="alert" style={{ margin: "0 28px 16px" }}>
                  {searchMessage}
                </div>
              )}

              <div className="empty-state">
                <div className="empty-icon" aria-hidden>◇</div>
                <div>No patient selected</div>
                <div style={{ fontSize: 13, color: "var(--text2)", maxWidth: 400 }}>
                  Choose a row in the registry or search by patient number to view
                  demographics, modalities, and analysis.
                </div>
              </div>
            </>
          ) : (
            <>
              {/* Patient detail */}
              <div className="detail-header">
                <div className="detail-id">Patient record</div>
                <div className="detail-title">{selected.condition_id}</div>
                <div className="detail-pills">
                  {[
                    ["Age",       selected.age != null ? `${Math.round(selected.age)} yrs` : "Unknown"],
                    ["Sex",       selected.sex || "Unknown"],
                    ["Country",   selected.country || "Unknown"],
                    ["Case type", selected.case_definition || "Unknown"],
                    ["HIV",       selected.hiv_status || "Unknown"],
                  ].map(([label, val]) => (
                    <div className="pill" key={label}>
                      <span className="pill-label">{label}</span>
                      <span>{val}</span>
                    </div>
                  ))}
                  <span
                    className={`resistance-chip ${resistanceClass(selected.type_of_resistance)}`}
                    style={{ padding: "6px 14px", borderRadius: 999, fontSize: 12 }}
                  >
                    {selected.type_of_resistance || "Resistance unknown"}
                  </span>
                </div>
              </div>

              <div className="detail-body">
                {searchMessage && (
                  <div className="search-banner" role="alert">{searchMessage}</div>
                )}

                <div className="data-grid">
                  {[
                    ["🧬 Genomics", selected.has_genomics],
                    ["🩻 CXR",      selected.has_cxr],
                    ["💿 CT Scan",  selected.has_ct],
                    ["🔬 DST",      selected.has_dst],
                  ].map(([label, avail]) => (
                    <div className="data-card" key={label}>
                      <div className="data-card-header">
                        <span className="data-card-icon">{label.split(" ")[0]}</span>
                        {label.split(" ").slice(1).join(" ")} data
                        <span style={{ marginLeft: "auto", fontSize: 11, color: avail ? "var(--ok)" : "var(--text3)" }}>
                          {avail ? "Available" : "Not in record"}
                        </span>
                      </div>
                      <div className="data-card-body">
                        <div className="kv-row">
                          <span className="kv-key">Status</span>
                          <span className="kv-val" style={{ color: avail ? "var(--ok)" : "var(--text3)" }}>
                            {avail ? "Included for analysis" : "Excluded"}
                          </span>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>

                <div className="data-card">
                  <div className="data-card-header">
                    <span className="data-card-icon">📋</span>
                    Case summary
                  </div>
                  <div className="data-card-body">
                    {[
                      ["Record ID",            selected.condition_id],
                      ["Patient ID",           selected.patient_id || "—"],
                      ["Age",                  selected.age != null ? `${Math.round(selected.age)} years` : "—"],
                      ["Sex",                  selected.sex || "—"],
                      ["Country",              selected.country || "—"],
                      ["Drug resistance",      selected.type_of_resistance || "—"],
                      ["Case classification",  selected.case_definition || "—"],
                      ["HIV status",           selected.hiv_status || "—"],
                    ].map(([k, v]) => (
                      <div className="kv-row" key={k}>
                        <span className="kv-key">{k}</span>
                        <span className="kv-val">{v}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="analyze-section">
                  <div className="analyze-header">
                    <span className="analyze-title">Multi-agent analysis</span>
                    <button
                      type="button"
                      className="run-btn"
                      disabled={analysis.loading}
                      onClick={() => runAnalysis(selected)}
                    >
                      {analysis.loading
                        ? <><div className="spinner" /> Running analysis…</>
                        : "Run analysis"
                      }
                    </button>
                  </div>

                  {(analysis.loading || analysis.data || analysis.error) && (
                    <AgentStrip status={agentStatus || (analysis.loading ? "running" : null)} />
                  )}

                  {analysis.loading && (
                    <div style={{ padding: "16px 18px 8px" }}>
                      <div style={{ fontSize: 12, color: "var(--text3)", marginBottom: 10 }}>
                        Clinical, genomic, imaging, and integration agents are processing this case…
                      </div>
                      {[90, 70, 55, 80, 65].map((w, i) => (
                        <div key={i} className="skeleton" style={{ width: `${w}%`, height: 11 }} />
                      ))}
                    </div>
                  )}

                  {analysis.error && (
                    <div className="error-box">{analysis.error}</div>
                  )}

                  {verdictText && !analysis.loading && (
                    <div className="verdict-box">
                      {banner && (
                        <div className={`verdict-banner ${banner.cls}`}>
                          <div>
                            <div className="verdict-banner-label">Final verdict</div>
                            <div className="verdict-banner-value">{banner.verdict}</div>
                          </div>
                          {banner.type && (
                            <>
                              <div className="verdict-banner-sep" />
                              <div>
                                <div className="verdict-banner-label">TB type</div>
                                <div className="verdict-banner-value" style={{ fontSize: 14 }}>{banner.type}</div>
                              </div>
                            </>
                          )}
                          {banner.confidence && (
                            <div className="verdict-confidence">
                              <div className="verdict-confidence-num">
                                {banner.confidence}<span style={{ fontSize: 14 }}>/100</span>
                              </div>
                              <div className="verdict-confidence-label">Confidence</div>
                            </div>
                          )}
                        </div>
                      )}

                      <div className="verdict-md">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
                      </div>
                    </div>
                  )}

                  {!analysis.data && !analysis.loading && !analysis.error && (
                    <div style={{ padding: "18px", fontSize: 13, color: "var(--text2)" }}>
                      Run analysis to generate a structured assessment from available modalities.
                      Results support research and education only — not a substitute for clinical judgment.
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </main>
      </div>
    </>
  );
}