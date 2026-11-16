
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter } from 'react-router-dom'
import {
  QueryClient,
  QueryClientProvider,
} from '@tanstack/react-query'

import './index.css'
import './App.css'

import App from './App'
import { AnalysisProvider } from './context/AnalysisContext'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
      staleTime: 30_000,
    },
    mutations: {
      retry: false,
    },
  },
})

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <AnalysisProvider>
          <App />
        </AnalysisProvider>
      </BrowserRouter>
    </QueryClientProvider>
  </StrictMode>,
)
