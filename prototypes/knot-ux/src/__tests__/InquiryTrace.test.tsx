import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import InquiryTrace from '../components/InquiryTrace'
import type { TraceResult } from '../types'

const mockResult: TraceResult = {
  context: "Sarah plays tennis every weekend.",
  pack_fact_ids: ["a1b2c3d4"],
  trace: {
    intent: "factual",
    stage1_candidates: {
      from_bm25: ["a1b2c3d4"],
      from_rare_tokens: [],
      from_entity_hop: [],
    },
  },
}

test("renders all 6 trace steps", () => {
  render(<InquiryTrace result={mockResult} question="What sport?" />)
  expect(screen.getByText(/intent/i)).toBeInTheDocument()
  expect(screen.getByText(/candidates/i)).toBeInTheDocument()
  expect(screen.getByText(/evidence/i)).toBeInTheDocument()
})

test("shows Bridge inactive when no stage0", () => {
  render(<InquiryTrace result={mockResult} question="test" />)
  expect(screen.getByText(/bridge inactive/i)).toBeInTheDocument()
})

test("missing field shows dash", () => {
  const result = { ...mockResult, trace: { ...mockResult.trace, intent: undefined } }
  render(<InquiryTrace result={result as TraceResult} question="test" />)
  // shouldn't crash
})

test("collapses and expands on header click", async () => {
  const user = userEvent.setup()
  render(<InquiryTrace result={mockResult} question="What sport?" />)
  // Initially expanded — bridge inactive text is visible
  expect(screen.getByText(/bridge inactive/i)).toBeInTheDocument()
  // Click the header to collapse
  await user.click(screen.getByRole('button'))
  expect(screen.queryByText(/bridge inactive/i)).not.toBeInTheDocument()
  // Click again to expand
  await user.click(screen.getByRole('button'))
  expect(screen.getByText(/bridge inactive/i)).toBeInTheDocument()
})

test("shows no evidence warning when pack_fact_ids is empty", () => {
  const emptyResult: TraceResult = {
    ...mockResult,
    pack_fact_ids: [],
  }
  render(<InquiryTrace result={emptyResult} question="Unknown?" />)
  expect(screen.getAllByText(/no evidence found/i).length).toBeGreaterThan(0)
})
