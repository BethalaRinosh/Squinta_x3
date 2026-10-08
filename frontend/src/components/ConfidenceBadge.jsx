export default function ConfidenceBadge({ confidence, className = '' }) {
  const pct = Math.round(confidence * 100);

  let colorClasses;
  let label;
  if (confidence > 0.8) {
    colorClasses = 'bg-green-100 text-green-800';
    label = 'High';
  } else if (confidence >= 0.5) {
    colorClasses = 'bg-amber-100 text-amber-800';
    label = 'Medium';
  } else {
    colorClasses = 'bg-red-100 text-red-800';
    label = 'Low';
  }

  return (
    <span
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs font-medium ${colorClasses} ${className}`}
      title={`Confidence: ${pct}%`}
    >
      <span
        className={`inline-block w-1.5 h-1.5 rounded-full ${
          confidence > 0.8
            ? 'bg-green-500'
            : confidence >= 0.5
              ? 'bg-amber-500'
              : 'bg-red-500'
        }`}
      />
      {pct}%
    </span>
  );
}
