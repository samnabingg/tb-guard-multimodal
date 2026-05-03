const rawApiBaseUrl =
  import.meta.env.VITE_API_BASE_URL?.trim() || "http://localhost:8000";
export const API_BASE_URL = rawApiBaseUrl.replace(/\/+$/, "");

async function request(path, options = {}) {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
    ...options,
  });

  if (!response.ok) {
    const text = await response.text();
    let message = text || "Unknown error";
    try {
      const parsed = JSON.parse(text);
      message =
        typeof parsed?.detail === "string"
          ? parsed.detail
          : JSON.stringify(parsed?.detail ?? parsed);
    } catch {
      // keep plain text
    }
    throw new Error(`API ${response.status}: ${message}`);
  }

  return response.json();
}

// ── Health ────────────────────────────────────────────────────────────────────

export function getHealth() {
  return request("/health");
}

// ── Legacy local-data patients ────────────────────────────────────────────────

export function getPatients({ limit = 10, skip = 0 } = {}) {
  return request(`/patients?limit=${limit}&skip=${skip}`);
}

// ── TB Portals DEPOT patients (real or mock) ──────────────────────────────────

export function getTbDepotPatients({ limit = 10, skip = 0 } = {}) {
  return request(`/tbdepot/patients?limit=${limit}&skip=${skip}`);
}

export function getTbDepotPatient(conditionId) {
  return request(`/tbdepot/patients/${encodeURIComponent(conditionId)}`);
}

/**
 * Fetch full record + run council for a condition_id.
 * Returns { condition_id, source, patient_case, result }
 */
export function analyzeTbDepotPatient(conditionId) {
  return request(`/tbdepot/analyze/${encodeURIComponent(conditionId)}`, {
    method: "POST",
  });
}

// ── Legacy analyze routes (kept for manual form) ──────────────────────────────

export function analyzeCase(payload) {
  return request("/analyze-case", { method: "POST", body: JSON.stringify(payload) });
}

export function analyzeTbDepotCase(payload) {
  return request("/analyze-tbdepot-case", { method: "POST", body: JSON.stringify(payload) });
}