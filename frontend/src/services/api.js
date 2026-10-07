const API_BASE = (import.meta.env.VITE_API_BASE_URL || (import.meta.env.DEV ? 'http://localhost:8000' : '')).replace(/\/+$/, '')
async function request(path, options) {
  if (!API_BASE && import.meta.env.PROD) throw new Error('VITE_API_BASE_URL must be set to the deployed backend URL.')
  const response = await fetch(`${API_BASE}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Request failed (${response.status})`) }
  return response.json()
}
export const api = {
  dashboard: () => request('/api/dashboard'), forecast: () => request('/api/forecast'),
  resources: () => request('/api/resources'), alerts: () => request('/api/alerts'),
  recommendations: () => request('/api/recommendations'),
  simulate: (payload) => request('/api/simulation', { method: 'POST', body: JSON.stringify(payload) }),
  surge: () => request('/api/emergency-surge', { method: 'POST' }),
}
