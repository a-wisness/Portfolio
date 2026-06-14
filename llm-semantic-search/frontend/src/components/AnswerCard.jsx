// Renders the synthesized answer, styling inline [n] citation markers.

function renderWithCitations(text) {
  // Split on bracketed numbers like [1] or [12], keeping the delimiters.
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    if (/^\[\d+\]$/.test(part)) {
      return (
        <sup key={i} className="citation">
          {part}
        </sup>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export default function AnswerCard({ answer }) {
  if (!answer) return null;
  return (
    <div className="answer">
      <div className="answer__label">Answer</div>
      <div className="answer__body">{renderWithCitations(answer)}</div>
    </div>
  );
}
