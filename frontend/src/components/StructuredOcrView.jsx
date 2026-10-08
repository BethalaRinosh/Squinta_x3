import { useEffect, useMemo, useState } from 'react';

const COLORS = {
  arrow: '#2563eb',
  table: '#374151',
  box: '#2563eb',
  circle: '#2563eb',
  connector: '#7c3aed',
  underline: '#d97706',
  bracket: '#b45309',
  diagram: '#2563eb',
};

function parseGeometry(element) {
  if (!element?.geometry) return {};
  try {
    return JSON.parse(element.geometry);
  } catch {
    return {};
  }
}

function pixelPoint(point, width, height, geometry) {
  if (!Array.isArray(point) || point.length < 2) return null;
  const x = Number(point[0]);
  const y = Number(point[1]);
  if (!Number.isFinite(x) || !Number.isFinite(y)) return null;

  if (geometry?.coordinate_space === 'pixel') return [x, y];

  return [(x / 1000) * width, (y / 1000) * height];
}

function elementPoints(element, width, height) {
  const geometry = parseGeometry(element);
  return (Array.isArray(geometry.points) ? geometry.points : [])
    .map((point) => pixelPoint(point, width, height, geometry))
    .filter(Boolean);
}

function bboxFor(element) {
  return {
    x: Number(element?.bbox_x) || 0,
    y: Number(element?.bbox_y) || 0,
    w: Number(element?.bbox_w) || 1,
    h: Number(element?.bbox_h) || 1,
  };
}

export default function StructuredOcrView({
  results = [],
  visualElements = [],
  imageSrc,
  selectedResultId,
  onSelectResult,
  onSpeak,
  speakingResultId,
}) {
  const [pageSize, setPageSize] = useState({ width: 1000, height: 1400 });

  useEffect(() => {
    if (!imageSrc) return undefined;

    const image = new Image();
    image.onload = () => {
      if (image.naturalWidth && image.naturalHeight) {
        setPageSize({
          width: image.naturalWidth,
          height: image.naturalHeight,
        });
      }
    };
    image.src = imageSrc;

    return () => {
      image.onload = null;
    };
  }, [imageSrc]);

  const textItems = useMemo(
    () => results
      .filter((result) => result && typeof result.text === 'string' && result.text.trim())
      .map((result) => ({ ...result, ...bboxFor(result) })),
    [results],
  );

  const visuals = useMemo(
    () => visualElements
      .map((element) => ({
        ...element,
        geometry: parseGeometry(element),
        points: elementPoints(element, pageSize.width, pageSize.height),
        bbox: bboxFor(element),
      }))
      .filter((element) => element.bbox.w > 0 && element.bbox.h > 0),
    [visualElements, pageSize],
  );

  const arrowCount = visuals.filter((e) => String(e.element_type).toLowerCase() === 'arrow').length;
  const tableCount = visuals.filter((e) => String(e.element_type).toLowerCase() === 'table').length;

  if (!textItems.length && !visuals.length) {
    return <div className="p-6 text-center text-sm text-gray-400">No formatted content yet.</div>;
  }

  const sx = 100 / pageSize.width;
  const sy = 100 / pageSize.height;

  return (
    <div className="p-3">
      <div className="mb-2 flex items-center justify-between text-[11px] text-gray-400">
        <span>2D reconstruction from original page geometry</span>
        <span>
          {arrowCount > 0 ? arrowCount + ' arrow' + (arrowCount !== 1 ? 's' : '') : ''}
          {arrowCount > 0 && tableCount > 0 ? ' · ' : ''}
          {tableCount > 0 ? tableCount + ' table' + (tableCount !== 1 ? 's' : '') : ''}
        </span>
      </div>

      <div
        className="relative w-full overflow-hidden rounded-lg border border-gray-300 bg-white shadow-sm"
        style={{ aspectRatio: pageSize.width + ' / ' + pageSize.height }}
      >
        <svg
          className="absolute inset-0 h-full w-full pointer-events-none"
          viewBox={'0 0 ' + pageSize.width + ' ' + pageSize.height}
          preserveAspectRatio="none"
          aria-hidden="true"
        >
          <defs>
            <marker
              id="formatted-arrow-head"
              markerWidth="14"
              markerHeight="11"
              refX="12"
              refY="5.5"
              orient="auto"
              markerUnits="userSpaceOnUse"
            >
              <path d="M0,0 L14,5.5 L0,11 Z" fill={COLORS.arrow} />
            </marker>
          </defs>

          {visuals.map((element) => {
            const type = String(element.element_type || 'diagram').toLowerCase();
            const color = COLORS[type] || COLORS.diagram;
            const { x, y, w, h } = element.bbox;
            const points = element.points;

            const fallback = [
              [x, y],
              [x + w, y],
              [x + w, y + h],
              [x, y + h],
            ];

            const linePoints = points.length >= 2
              ? points
              : [[x, y + h / 2], [x + w, y + h / 2]];

            const polygonPoints = points.length >= 4 ? points : fallback;
            const strokeWidth = Math.max(2, Math.min(5, Math.max(pageSize.width, pageSize.height) / 650));

            if (type === 'arrow') {
              return (
                <polyline
                  key={'arrow-' + element.id}
                  points={linePoints.map(([px, py]) => px + ',' + py).join(' ')}
                  fill="none"
                  stroke={color}
                  strokeWidth={strokeWidth}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  markerEnd="url(#formatted-arrow-head)"
                />
              );
            }

            if (type === 'table') {
              const gridLines = Array.isArray(element.geometry?.grid_lines)
                ? element.geometry.grid_lines
                  .map((line) => line
                    .map((point) => pixelPoint(point, pageSize.width, pageSize.height, element.geometry))
                    .filter(Boolean))
                  .filter((line) => line.length >= 2)
                : [];

              return (
                <g key={'table-' + element.id}>
                  <polygon
                    points={polygonPoints.map(([px, py]) => px + ',' + py).join(' ')}
                    fill="none"
                    stroke={color}
                    strokeWidth={strokeWidth}
                    strokeLinejoin="round"
                  />
                  {gridLines.map((line, index) => (
                    <polyline
                      key={'table-grid-' + element.id + '-' + index}
                      points={line.map(([px, py]) => px + ',' + py).join(' ')}
                      fill="none"
                      stroke={color}
                      strokeWidth={Math.max(1.5, strokeWidth * 0.7)}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    />
                  ))}
                </g>
              );
            }

            if (type === 'circle') {
              const center = Array.isArray(element.geometry?.center)
                ? pixelPoint(element.geometry.center, pageSize.width, pageSize.height, element.geometry)
                : [x + w / 2, y + h / 2];
              const radius = Array.isArray(element.geometry?.radius)
                ? [
                    Math.abs(Number(element.geometry.radius[0])) * (
                      element.geometry?.coordinate_space === 'pixel' ? 1 : pageSize.width / 1000
                    ),
                    Math.abs(Number(element.geometry.radius[1])) * (
                      element.geometry?.coordinate_space === 'pixel' ? 1 : pageSize.height / 1000
                    ),
                  ]
                : [w / 2, h / 2];

              return (
                <ellipse
                  key={'circle-' + element.id}
                  cx={center[0]}
                  cy={center[1]}
                  rx={Math.max(1, radius[0])}
                  ry={Math.max(1, radius[1])}
                  fill="none"
                  stroke={color}
                  strokeWidth={strokeWidth}
                />
              );
            }

            if (type === 'box') {
              return (
                <polygon
                  key={'box-' + element.id}
                  points={polygonPoints.map(([px, py]) => px + ',' + py).join(' ')}
                  fill="none"
                  stroke={color}
                  strokeWidth={strokeWidth}
                  strokeLinejoin="round"
                />
              );
            }

            if (type === 'underline' || type === 'connector' || type === 'bracket') {
              return (
                <polyline
                  key={type + '-' + element.id}
                  points={linePoints.map(([px, py]) => px + ',' + py).join(' ')}
                  fill="none"
                  stroke={color}
                  strokeWidth={strokeWidth}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              );
            }

            return (
              <polygon
                key={type + '-' + element.id}
                points={polygonPoints.map(([px, py]) => px + ',' + py).join(' ')}
                fill="none"
                stroke={color}
                strokeWidth={strokeWidth}
                strokeLinejoin="round"
              />
            );
          })}
        </svg>

        {textItems.map((result) => {
          const isSelected = result.id === selectedResultId;
          const fontSize = Math.max(10, Math.min(22, result.h * 0.72));

          return (
            <div
              key={result.id}
              className="absolute group"
              style={{
                left: (result.x * sx) + '%',
                top: (result.y * sy) + '%',
                width: Math.max(1, result.w * sx) + '%',
                minHeight: Math.max(1, result.h * sy) + '%',
              }}
            >
              <button
                type="button"
                onClick={() => onSelectResult?.(result.id)}
                className={
                  'block overflow-visible whitespace-nowrap select-text text-left font-mono leading-none ' +
                  'rounded px-0.5 transition-colors hover:bg-primary-50 ' +
                  (isSelected ? 'bg-primary-100 text-primary-800' : 'text-gray-800')
                }
                style={{
                  fontSize: fontSize + 'px',
                  textDecoration: result.struck_through ? 'line-through' : 'none',
                  textDecorationThickness: result.struck_through ? '2px' : undefined,
                }}
                title={
                  'Confidence: ' + Math.round((result.confidence || 0) * 100) + '%' +
                  (result.struck_through ? ' · Detected as struck through' : '')
                }
              >
                {result.text.trim()}
              </button>
              {onSpeak && result.text && (
                <button
                  type="button"
                  onClick={(event) => {
                    event.stopPropagation();
                    onSpeak(result);
                  }}
                  className={"absolute -right-6 top-1/2 -translate-y-1/2 p-0.5 rounded bg-white border border-gray-200 text-gray-400 opacity-0 group-hover:opacity-100 hover:text-primary-600 transition-opacity " + (speakingResultId === result.id ? "opacity-100 text-primary-600" : "")}
                  title={speakingResultId === result.id ? "Stop reading" : "Read this line aloud"}
                  aria-label={speakingResultId === result.id ? "Stop reading" : "Read this line aloud"}
                >
                  <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    {speakingResultId === result.id ? (
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 6h12v12H6z" />
                    ) : (
                      <>
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5L6 9H3v6h3l5 4V5z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.5 8.5a5 5 0 010 7M18.5 6a8 8 0 010 12" />
                      </>
                    )}
                  </svg>
                </button>
              )}
            </div>
          );
        })}
      </div>

      <p className="mt-2 px-1 text-[11px] leading-4 text-gray-400">
        Text, arrows, table outlines, and detected structure are placed using the original page coordinates.
      </p>
    </div>
  );
}
