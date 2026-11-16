
import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from 'react'

import type { QueryResponse } from '../types/verification'

interface AnalysisContextValue {
  response: QueryResponse | null
  question: string | null
  setAnalysis: (question: string, response: QueryResponse) => void
  clearAnalysis: () => void
}

const AnalysisContext = createContext<AnalysisContextValue | undefined>(
  undefined,
)

export function AnalysisProvider({
  children,
}: {
  children: ReactNode
}) {
  const [response, setResponse] = useState<QueryResponse | null>(null)
  const [question, setQuestion] = useState<string | null>(null)

  const setAnalysis = (
    newQuestion: string,
    newResponse: QueryResponse,
  ) => {
    setQuestion(newQuestion)
    setResponse(newResponse)
  }

  const clearAnalysis = () => {
    setQuestion(null)
    setResponse(null)
  }

  return (
    <AnalysisContext.Provider
      value={{
        response,
        question,
        setAnalysis,
        clearAnalysis,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  )
}

export function useAnalysisContext() {
  const context = useContext(AnalysisContext)

  if (!context) {
    throw new Error(
      'useAnalysisContext must be used inside AnalysisProvider.',
    )
  }

  return context
}
