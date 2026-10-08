import { useMemo } from 'react';

export default function StructuredOcrView({ results = [], visualElements = [], selectedResultId, onSelectResult }) {
  const lines = useMemo(() => {
    const items = results.filter((r) => r && typeof r.text === 'string' && r.text.trim()).map((r) => ({
      ...r, x: Number(r.bbox_x) || 0, y: Number(r.bbox_y) || 0, w: Number(r.bbox_w) || 0, h: Number(r.bbox_h) || 0,
      cy: (Number(r.bbox_y) || 0) + (Number(r.bbox_h) || 0) / 2,
    })).sort((a, b) => a.y - b.y || a.x - b.x);
    if (!items.length) return [];
    const heights = items.map((r) => r.h).filter((h) => h > 0).sort((a, b) => a - b);
    const medianH = heights[Math.floor(heights.length / 2)] || 24;
    const tolerance = Math.max(8, medianH * 0.55);
    const grouped = [];
    for (const item of items) {
      let best = null; let distance = Infinity;
      for (const line of grouped) {
        const d = Math.abs(item.cy - line.cy);
        if (d <= tolerance && d < distance) { best = line; distance = d; }
      }
      if (!best) grouped.push({ cy: item.cy, minY: item.y, maxY: item.y + item.h, items: [item] });
      else { best.items.push(item); best.cy = best.items.reduce((sum, r) => sum + r.cy, 0) / best.items.length; best.minY = Math.min(best.minY, item.y); best.maxY = Math.max(best.maxY, item.y + item.h); }
    }
    grouped.sort((a, b) => a.minY - b.minY);
    const minX = Math.min(...items.map((r) => r.x));
    const widths = items.map((r) => r.w / Math.max(1, r.text.length)).filter((v) => Number.isFinite(v) && v > 1).sort((a, b) => a - b);
    const charWidth = widths.length ? widths[Math.floor(widths.length / 2)] : Math.max(8, medianH * 0.45);
    return grouped.map((line, index) => {
      const sorted = line.items.sort((a, b) => a.x - b.x);
      const indent = Math.max(0, Math.min(36, Math.round((sorted[0].x - minX) / charWidth)));
      let right = null;
      const parts = sorted.map((item) => {
        const gap = right == null ? 0 : item.x - right;
        const spaces = gap > charWidth * 0.45 ? Math.min(12, Math.max(1, Math.round(gap / charWidth))) : 0;
        right = Math.max(right || 0, item.x + item.w);
        return { result: item, spaces };
      });
      const previous = grouped[index - 1];
      return {
        id: 'line-' + index + '-' + sorted[0].id,
        indent, parts,
        blankBefore: previous ? line.minY - previous.maxY > medianH * 0.9 : false,
        emphasis: Math.max(...sorted.map((r) => r.h), medianH) > medianH * 1.35,
      };
    });
  }, [results]);


  const tables = useMemo(() => {
    const tableElements = visualElements.filter(
      (e) => String(e.element_type || '').toLowerCase() === 'table'
    );

    return tableElements.map((table, tableIndex) => {
      let geometry = {};
      try { geometry = table.geometry ? JSON.parse(table.geometry) : {}; } catch { geometry = {}; }

      const scale = (value, axis) => {
        const n = Number(value);
        if (!Number.isFinite(n)) return 0;
        return geometry.coordinate_space === 'pixel'
          ? n
          : (n / 1000) * axis;
      };

      const pageW = 1000;
      const pageH = 1000;
      const rawPoints = Array.isArray(geometry.points) ? geometry.points : [];
      const points = rawPoints
        .filter((p) => Array.isArray(p) && p.length >= 2)
        .map((p) => [scale(p[0], pageW), scale(p[1], pageH)]);

      const bx = Number(table.bbox_x) || 0;
      const by = Number(table.bbox_y) || 0;
      const bw = Number(table.bbox_w) || 1;
      const bh = Number(table.bbox_h) || 1;

      const x1 = geometry.coordinate_space === 'pixel' ? bx : (bx / 1000) * pageW;
      const y1 = geometry.coordinate_space === 'pixel' ? by : (by / 1000) * pageH;
      const x2 = x1 + (geometry.coordinate_space === 'pixel' ? bw : (bw / 1000) * pageW);
      const y2 = y1 + (geometry.coordinate_space === 'pixel' ? bh : (bh / 1000) * pageH);

      const gridLines = Array.isArray(geometry.grid_lines)
        ? geometry.grid_lines
            .filter((line) => Array.isArray(line) && line.length >= 2)
            .map((line) => line
              .filter((p) => Array.isArray(p) && p.length >= 2)
              .map((p) => [scale(p[0], pageW), scale(p[1], pageH)]))
            .filter((line) => line.length >= 2)
        : [];

      const verticals = [];
      const horizontals = [];
      gridLines.forEach((line) => {
        const a = line[0];
        const b = line[line.length - 1];
        if (Math.abs(a[0] - b[0]) >= Math.abs(a[1] - b[1])) horizontals.push(line);
        else verticals.push(line);
      });

      const xs = [x1, x2, ...verticals.flatMap((line) => line.map((p) => p[0]))]
        .sort((a, b) => a - b);
      const ys = [y1, y2, ...horizontals.flatMap((line) => line.map((p) => p[1]))]
        .sort((a, b) => a - b);

      const unique = (values) => values.filter((v, i) => i === 0 || Math.abs(v - values[i - 1]) > 8);
      const columns = unique(xs);
      const rows = unique(ys);

      const inside = results.filter((r) => {
        const cx = (Number(r.bbox_x) || 0) + (Number(r.bbox_w) || 0) / 2;
        const cy = (Number(r.bbox_y) || 0) + (Number(r.bbox_h) || 0) / 2;
        return cx >= bx && cx <= bx + bw && cy >= by && cy <= by + bh;
      });

      return { id: table.id || tableIndex, columns, rows, inside, points };
    });
  }, [visualElements, results]);

  const arrowCount = visualElements.filter((e) => String(e.element_type || '').toLowerCase() === 'arrow').length;
  if (!lines.length) return <div className="p-6 text-center text-sm text-gray-400">No text to format yet.</div>;

  return (
    <div className="p-3">
      <div className="mb-2 flex items-center justify-between text-[11px] text-gray-400">
        <span>Reconstructed from handwriting position</span>
        {arrowCount > 0 && <span>{arrowCount} arrow{arrowCount !== 1 ? 's' : ''} detected</span>}
      </div>
      <div className="rounded-lg border border-gray-200 bg-white px-4 py-4 shadow-sm overflow-x-auto">
        <div className="min-w-max font-mono text-[13px] leading-7 text-gray-800">
          {lines.map((line) => (
            <div key={line.id} className={line.blankBefore ? 'mt-4' : ''} style={{ paddingLeft: line.indent + 'ch' }}>
              {line.parts.map(({ result, spaces }) => (
                <span key={result.id}>
                  {' '.repeat(spaces)}
                  <button type="button" onClick={() => onSelectResult?.(result.id)}
                    className={'rounded px-0.5 text-left transition-colors hover:bg-primary-50 ' + (result.id === selectedResultId ? 'bg-primary-100 text-primary-800 ' : '') + (line.emphasis ? 'font-semibold' : '')}
                    title={'Confidence: ' + Math.round((result.confidence || 0) * 100) + '%'}>
                    {result.text.trim()}
                  </button>
                </span>
              ))}
            </div>
          ))}
        </div>
      </div>
      {tables.length > 0 && (
        <div className="mt-4 space-y-4">
          {tables.map((table) => (
            <div key={table.id}>
              <div className="mb-2 text-[11px] font-medium text-gray-500">
                Detected table
              </div>
              <div className="overflow-x-auto rounded-lg border border-gray-300 bg-white">
                <table className="w-full border-collapse text-sm">
                  <tbody>
                    {table.rows.slice(0, -1).map((_, rowIndex) => (
                      <tr key={rowIndex}>
                        {table.columns.slice(0, -1).map((_, colIndex) => {
                          const left = table.columns[colIndex];
                          const right = table.columns[colIndex + 1];
                          const top = table.rows[rowIndex];
                          const bottom = table.rows[rowIndex + 1];
                          const cell = table.inside
                            .filter((r) => {
                              const cx = (Number(r.bbox_x) || 0) + (Number(r.bbox_w) || 0) / 2;
                              const cy = (Number(r.bbox_y) || 0) + (Number(r.bbox_h) || 0) / 2;
                              return cx >= left && cx <= right && cy >= top && cy <= bottom;
                            })
                            .sort((a, b) => (a.bbox_x || 0) - (b.bbox_x || 0))
                            .map((r) => r.text.trim())
                            .filter(Boolean)
                            .join(' ');
                          return (
                            <td key={colIndex} className="border border-gray-400 px-3 py-2 align-top whitespace-pre-wrap">
                              {cell}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ))}
        </div>
      )}

      <p className="mt-2 px-1 text-[11px] leading-4 text-gray-400">
        Line breaks, spacing, indentation, and table outlines are reconstructed from the original text positions and detected page geometry.
      </p>
    </div>
  );
}