
import axios from 'axios'

import type {
  AskRequest,
  QueryResponse,
} from '../types/verification'

const API_BASE_URL =
  import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:8000'

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 180_000,
  headers: {
    'Content-Type': 'application/json',
    Accept: 'application/json',
  },
})

export async function askQuestion(
  payload: AskRequest,
): Promise<QueryResponse> {
  const response = await apiClient.post<QueryResponse>(
    '/ask',
    payload,
  )

  return response.data
}

export async function checkApiHealth(): Promise<boolean> {
  try {
    const response = await apiClient.get('/api/v1/health', {
      timeout: 10_000,
    })

    return response.status === 200
  } catch {
    return false
  }
}

export async function fetchMetrics(): Promise<Record<string, unknown>> {
  const response = await apiClient.get('/api/v1/metrics')
  return response.data
}

export async function fetchBenchmarks(): Promise<Record<string, unknown>> {
  const response = await apiClient.get('/api/v1/benchmarks')
  return response.data
}

export async function fetchHistory(limit = 20): Promise<Record<string, unknown>> {
  const response = await apiClient.get(`/api/v1/history?limit=${limit}`)
  return response.data
}

export async function ingestDocument(
  content: string,
  title?: string,
  format = 'txt'
): Promise<Record<string, unknown>> {
  const response = await apiClient.post('/api/v1/ingest', {
    content,
    document_title: title,
    format,
  })
  return response.data
}


export function getApiErrorMessage(error: unknown): string {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail

    if (typeof detail === 'string') {
      return detail
    }

    if (!error.response) {
      return 'Cannot connect to the ParmanaAI backend. Check that the API server is running.'
    }

    return error.message
  }

  if (error instanceof Error) {
    return error.message
  }

  return 'An unexpected error occurred.'
}
