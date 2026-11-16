
import { useMutation } from '@tanstack/react-query'

import { askQuestion } from '../services/api'
import type { AskRequest } from '../types/verification'

export function useAskQuestion() {
  return useMutation({
    mutationFn: (payload: AskRequest) => askQuestion(payload),

    retry: false,
  })
}
