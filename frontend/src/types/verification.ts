
export type VerificationVerdict =
  | 'SUPPORTED'
  | 'CONTRADICTED'
  | 'INSUFFICIENT_EVIDENCE'
  | 'NO_EVIDENCE_PROVIDED'
  | 'NO_NUMERICAL_EVIDENCE'
  | 'NUMERICALLY_CONSISTENT'
  | 'NUMERICAL_DISCREPANCY'
  | 'CONTEXT_MISMATCH'
  | 'INCONCLUSIVE'
  | 'CONFLICTING_SIGNALS'
  | 'PARTIALLY_SUPPORTED'
  | string

export interface AskRequest {
  query: string
  external_context?: string
}

export interface NliScores {
  entailment: number
  contradiction: number
  neutral: number
}


export interface EvidenceRecord {
  evidence_id: string
  text: string
  document_id?: string | null
  chunk_id?: string | null
  source_type?: string
  topic?: string | null
  match_type?: string
  retrieval_score?: number | null
  origin?: string
  provenance?: Record<string, unknown>
  verified?: boolean
  verification_status?: string
}

export interface EvidenceAssessment {
  evidence_id?: string | null
  document_id?: string | null
  chunk_id?: string | null
  source_type?: string
  nli_verdict?: VerificationVerdict
  nli_scores?: NliScores
  contradiction_guard_passed?: boolean
  is_primary_evidence?: boolean
}


export interface NliSignal {
  verdict?: VerificationVerdict
  scores?: NliScores
}

export interface NumericalSignal {
  verdict?: VerificationVerdict
  reason?: string
  source_assessment_count?: number
  unique_document_count?: number
}

export interface VerificationSignals {
  nli?: NliSignal
  numerical?: NumericalSignal
}

export interface NumericalAssessment {
  verdict: VerificationVerdict
  reason?: string

  source_assessment_count?: number
  unique_document_count?: number
  consistent_source_count?: number
  discrepant_source_count?: number
  context_mismatch_count?: number
  inconclusive_source_count?: number
  no_numerical_evidence_count?: number

  source_assessments?: unknown[]

  clinical_truth_established?: boolean
  medical_contradiction_established?: boolean

  [key: string]: unknown
}

export interface ClaimRecord {
  claim?: string
  claim_text?: string

  verdict?: VerificationVerdict
  nli_verdict?: VerificationVerdict
  combined_verdict?: VerificationVerdict

  numerical_assessment?: NumericalAssessment
  verification_signals?: VerificationSignals
  verification_components?: string[]

  confidence?: number
  method?: string

  supporting_source?: string
  source_type?: string
  citation?: string
  citation_match?: string

  supporting_sources?: number
  total_sources?: number

  claim_type?: string
  explanation?: string
  nli_scores?: NliScores
  verification_method_used?: string

  claim_id?: string
  extraction_type?: string
  is_inferential?: boolean
  is_definitive?: boolean
  contains_causal_language?: boolean
  contains_numerical_information?: boolean

  numerical_values?: unknown[]
  numerical_mentions?: unknown[]
  original_sentence?: string
  atomic?: boolean

  [key: string]: unknown

  primary_evidence_id?: string | null
  evidence_assessments?: EvidenceAssessment[]
}

export interface QueryResponse {
  answer: string
  claims: ClaimRecord[]
  evidence?: EvidenceRecord[]
  hallucination_score: number
  relevance_score: number
}

export interface ApiErrorResponse {
  detail?: string
  message?: string
}
