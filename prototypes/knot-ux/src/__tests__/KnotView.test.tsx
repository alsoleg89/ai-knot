import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import KnotView from '../components/KnotView'
import mockKnot from '../data/mock-knot.json'
import type { KnotData } from '../types'

test("renders correct strand count", () => {
  render(<KnotView data={mockKnot as KnotData} />)
  const data = mockKnot as KnotData
  data.strands.forEach(strand => {
    expect(screen.getByText(strand.label)).toBeInTheDocument()
  })
})

test("clicking bead shows detail panel", async () => {
  const user = userEvent.setup()
  render(<KnotView data={mockKnot as KnotData} />)
  const firstBead = mockKnot.strands[0]?.beads[0]
  if (firstBead) {
    const beadEl = screen.getByText(new RegExp(firstBead.content.slice(0, 20)))
    await user.click(beadEl)
    // detail panel should appear
    expect(screen.getByText(firstBead.content)).toBeInTheDocument()
  }
})

test("clicking same bead twice closes detail panel", async () => {
  const user = userEvent.setup()
  render(<KnotView data={mockKnot as KnotData} />)
  const firstBead = mockKnot.strands[0]?.beads[0]
  if (firstBead) {
    const snippet = firstBead.content.slice(0, 20)
    // Open
    await user.click(screen.getByText(new RegExp(snippet)))
    expect(screen.getByText('Bead detail')).toBeInTheDocument()
    // Close via close button
    await user.click(screen.getByRole('button', { name: /close detail panel/i }))
    expect(screen.queryByText('Bead detail')).not.toBeInTheDocument()
  }
})

test("onBeadClick callback is invoked", async () => {
  const user = userEvent.setup()
  const handleClick = vi.fn()
  render(<KnotView data={mockKnot as KnotData} onBeadClick={handleClick} />)
  const firstBead = mockKnot.strands[0]?.beads[0]
  if (firstBead) {
    await user.click(screen.getByText(new RegExp(firstBead.content.slice(0, 20))))
    expect(handleClick).toHaveBeenCalledTimes(1)
    expect(handleClick).toHaveBeenCalledWith(expect.objectContaining({ id: firstBead.id }))
  }
})
