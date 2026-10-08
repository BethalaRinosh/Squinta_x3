import { useState, useRef, useEffect, useCallback } from 'react';

export default function PageViewer({
  imageSrc,
  ocrResults = [],
  visualElements = [],
  selectedResultId,
  onSelectResult,
  crop = null,
  drawMode = false,
  onDrawBbox,
}) {
  const containerRef = useRef(null);
  const imgRef = useRef(null);
  const [imgDimensions, setImgDimensions] = useState({ width: 0, height: 0, naturalWidth: 0, naturalHeight: 0 });
  const [loaded, setLoaded] = useState(false);
  const [drawing, setDrawing] = useState(null);

  // Reset loaded state when image source changes (e.g., after rotation)
  useEffect(() => {
    setLoaded(false);
  }, [imageSrc]);

  const updateDimensions = useCallback(() => {
    if (!imgRef.current) return;
    const img = imgRef.current;
    setImgDimensions({
      width: img.clientWidth,
      height: img.clientHeight,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    });
  }, []);

  useEffect(() => {
    if (!loaded) return;
    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, [loaded, updateDimensions]);

  const handleImageLoad = () => {
    setLoaded(true);
    updateDimensions();
  };

  const scaleX = imgDimensions.naturalWidth > 0 ? imgDimensions.width / imgDimensions.naturalWidth : 1;
  const scaleY = imgDimensions.naturalHeight > 0 ? imgDimensions.height / imgDimensions.naturalHeight : 1;

  // ── Draw mode mouse handlers ──
  const toNatural = useCallback((clientX, clientY) => {
    if (!imgRef.current) return { x: 0, y: 0 };
    const rect = imgRef.current.getBoundingClientRect();
    const x = Math.round((clientX - rect.left) / scaleX);
    const y = Math.round((clientY - rect.top) / scaleY);
    return {
      x: Math.max(0, Math.min(x, imgDimensions.naturalWidth)),
      y: Math.max(0, Math.min(y, imgDimensions.naturalHeight)),
    };
  }, [scaleX, scaleY, imgDimensions.naturalWidth, imgDimensions.naturalHeight]);

  const handleMouseDown = useCallback((e) => {
    if (!drawMode) return;
    e.preventDefault();
    const pos = toNatural(e.clientX, e.clientY);
    setDrawing({ startX: pos.x, startY: pos.y, currentX: pos.x, currentY: pos.y });
  }, [drawMode, toNatural]);

  const handleMouseMove = useCallback((e) => {
    if (!drawing) return;
    const pos = toNatural(e.clientX, e.clientY);
    setDrawing(prev => ({ ...prev, currentX: pos.x, currentY: pos.y }));
  }, [drawing, toNatural]);

  const handleMouseUp = useCallback(() => {
    if (!drawing) return;
    const x = Math.min(drawing.startX, drawing.currentX);
    const y = Math.min(drawing.startY, drawing.currentY);
    const w = Math.abs(drawing.currentX - drawing.startX);
    const h = Math.abs(drawing.currentY - drawing.startY);
    if (w > 5 && h > 5) {
      onDrawBbox?.({ x, y, w, h });
    }
    setDrawing(null);
  }, [drawing, onDrawBbox]);

  // ── Touch handlers (mobile draw support) ──
  const handleTouchStart = useCallback((e) => {
    if (!drawMode) return;
    e.preventDefault();
    const touch = e.touches[0];
    const pos = toNatural(touch.clientX, touch.clientY);
    setDrawing({ startX: pos.x, startY: pos.y, currentX: pos.x, currentY: pos.y });
  }, [drawMode, toNatural]);

  const handleTouchMove = useCallback((e) => {
    if (!drawing) return;
    e.preventDefault();
    const touch = e.touches[0];
    const pos = toNatural(touch.clientX, touch.clientY);
    setDrawing(prev => ({ ...prev, currentX: pos.x, currentY: pos.y }));
  }, [drawing, toNatural]);

  const handleTouchEnd = useCallback(() => {
    handleMouseUp();
  }, [handleMouseUp]);

  // Draw rect in display coords
  const drawRect = drawing ? {
    left: Math.min(drawing.startX, drawing.currentX) * scaleX,
    top: Math.min(drawing.startY, drawing.currentY) * scaleY,
    width: Math.abs(drawing.currentX - drawing.startX) * scaleX,
    height: Math.abs(drawing.currentY - drawing.startY) * scaleY,
  } : null;

  // Crop overlay in display coords
  const cropDisplay = crop ? {
    left: crop.crop_x * scaleX,
    top: crop.crop_y * scaleY,
    width: crop.crop_w * scaleX,
    height: crop.crop_h * scaleY,
  } : null;

  return (
    <div
      ref={containerRef}
      className={`relative inline-block w-full overflow-hidden ${drawMode ? 'cursor-crosshair' : ''}`}
      onMouseDown={handleMouseDown}
      onMouseMove={handleMouseMove}
      onMouseUp={handleMouseUp}
      onMouseLeave={handleMouseUp}
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
      onTouchEnd={handleTouchEnd}
    >
      {!loaded && (
        <div className="skeleton w-full" style={{ paddingBottom: '141.4%' }} />
      )}
      <div className="relative">
        <img
          ref={imgRef}
          src={imageSrc}
          alt="Document page"
          loading="lazy"
          onLoad={handleImageLoad}
          className={`w-full h-auto block transition-opacity duration-300 ${loaded ? 'opacity-100' : 'opacity-0'}`}
          draggable={false}
        />

        {/* Persistent crop overlay */}
        {loaded && cropDisplay && !drawMode && (
          <>
            <div className="absolute inset-x-0 top-0 bg-black/30" style={{ height: `${cropDisplay.top}px` }} />
            <div className="absolute inset-x-0 bg-black/30" style={{ top: `${cropDisplay.top + cropDisplay.height}px`, bottom: 0 }} />
            <div className="absolute bg-black/30" style={{ top: `${cropDisplay.top}px`, left: 0, width: `${cropDisplay.left}px`, height: `${cropDisplay.height}px` }} />
            <div className="absolute bg-black/30" style={{ top: `${cropDisplay.top}px`, left: `${cropDisplay.left + cropDisplay.width}px`, right: 0, height: `${cropDisplay.height}px` }} />
            <div
              className="absolute border-2 border-dashed border-green-500"
              style={{
                left: `${cropDisplay.left}px`,
                top: `${cropDisplay.top}px`,
                width: `${cropDisplay.width}px`,
                height: `${cropDisplay.height}px`,
              }}
            />
          </>
        )}

        {/* OCR result bounding boxes */}
        {loaded && !drawMode &&
          ocrResults.map((result) => {
            const x = result.bbox_x;
            const y = result.bbox_y;
            const w = result.bbox_w;
            const h = result.bbox_h;

            if (x == null || y == null || w == null || h == null) return null;

            const isSelected = result.id === selectedResultId;

            return (
              <div
                key={result.id}
                onClick={() => onSelectResult?.(result.id)}
                className={`absolute border-2 cursor-pointer transition-all duration-200 ${
                  isSelected
                    ? 'border-primary-500 bg-primary-500/20 shadow-lg'
                    : 'border-transparent hover:border-primary-300 hover:bg-primary-300/10'
                }`}
                style={{
                  left: `${x * scaleX}px`,
                  top: `${y * scaleY}px`,
                  width: `${w * scaleX}px`,
                  height: `${h * scaleY}px`,
                }}
                title={result.text}
              />
            );
          })}

        {/* Detected visual structures. Render the actual geometry returned by
            OCR instead of approximating every element with an axis-aligned CSS box. */}
        {loaded && !drawMode && visualElements.length > 0 && (
          <svg
            className="absolute inset-0 w-full h-full pointer-events-none overflow-visible"
            viewBox={`0 0 ${imgDimensions.naturalWidth || 1} ${imgDimensions.naturalHeight || 1}`}
            preserveAspectRatio="none"
            aria-hidden="true"
          >
            <defs>
              <marker
                id="visual-arrow-head"
                markerWidth="10"
                markerHeight="8"
                refX="8"
                refY="4"
                orient="auto"
                markerUnits="userSpaceOnUse"
              >
                <path d="M0,0 L10,4 L0,8 Z" fill="currentColor" />
              </marker>
            </defs>

            {visualElements.map((element) => {
              const x = Number(element.bbox_x);
              const y = Number(element.bbox_y);
              const w = Number(element.bbox_w);
              const h = Number(element.bbox_h);
              if (![x, y, w, h].every(Number.isFinite) || w <= 0 || h <= 0) return null;

              const type = String(element.element_type || 'diagram').toLowerCase();
              let geometry = null;
              try {
                geometry = element.geometry ? JSON.parse(element.geometry) : null;
              } catch {
                geometry = null;
              }

              const rawPoints = Array.isArray(geometry?.points) ? geometry.points : [];
              const points = rawPoints
                .filter((p) => Array.isArray(p) && p.length >= 2)
                .map((p) => [Number(p[0]), Number(p[1])])
                .filter(([px, py]) => Number.isFinite(px) && Number.isFinite(py));

              const fallback = [
                [x, y],
                [x + w, y],
                [x + w, y + h],
                [x, y + h],
              ];

              const lineFallback = [
                [x, y + h / 2],
                [x + w, y + h / 2],
              ];

              const pts = points.length >= 2 ? points : fallback;
              const linePts = points.length >= 2 ? points : lineFallback;
              const pointString = pts.map(([px, py]) => `${px},${py}`).join(' ');
              const lineString = linePts.map(([px, py]) => `${px},${py}`).join(' ');

              const stroke =
                type === 'arrow' ? '#2563eb' :
                type === 'connector' ? '#7c3aed' :
                type === 'underline' ? '#d97706' :
                type === 'bracket' ? '#b45309' :
                type === 'table' ? '#4f46e5' :
                '#2563eb';

              const common = {
                stroke,
                strokeWidth: Math.max(2, Math.min(5, Math.max(imgDimensions.naturalWidth, imgDimensions.naturalHeight) / 900)),
                fill: 'none',
                vectorEffect: 'non-scaling-stroke',
                strokeLinecap: 'round',
                strokeLinejoin: 'round',
              };

              if (type === 'circle') {
                const center = Array.isArray(geometry?.center) && geometry.center.length >= 2
                  ? [Number(geometry.center[0]), Number(geometry.center[1])]
                  : [x + w / 2, y + h / 2];
                const radius = Array.isArray(geometry?.radius) && geometry.radius.length >= 2
                  ? [Math.max(1, Number(geometry.radius[0])), Math.max(1, Number(geometry.radius[1]))]
                  : [w / 2, h / 2];

                return (
                  <ellipse
                    {...common}
                    cx={center[0]}
                    cy={center[1]}
                    rx={radius[0]}
                    ry={radius[1]}
                  >
                    <title>{element.label || 'circle'}</title>
                  </ellipse>
                );
              }

              if (type === 'arrow') {
                return (
                  <polyline
                    {...common}
                    points={lineString}
                    markerEnd="url(#visual-arrow-head)"
                  >
                    <title>{element.label || 'arrow'}</title>
                  </polyline>
                );
              }

              if (type === 'connector' || type === 'underline' || type === 'bracket') {
                return (
                  <polyline {...common} points={lineString}>
                    <title>{element.label || type}</title>
                  </polyline>
                );
              }

              if (type === 'table') {
                const quad = points.length >= 4 ? points.slice(0, 4) : fallback;
                const rows = Math.max(1, Number(geometry?.rows) || 1);
                const columns = Math.max(1, Number(geometry?.columns) || 1);
                const [tl, tr, br, bl] = quad;

                const lerp = (a, b, t) => [
                  a[0] + (b[0] - a[0]) * t,
                  a[1] + (b[1] - a[1]) * t,
                ];
                const gridLines = [];

                for (let row = 1; row < rows; row += 1) {
                  const t = row / rows;
                  const left = lerp(tl, bl, t);
                  const right = lerp(tr, br, t);
                  gridLines.push(
                    <line
                      key={`row-${row}`}
                      {...common}
                      strokeWidth={Math.max(1, common.strokeWidth * 0.65)}
                      x1={left[0]}
                      y1={left[1]}
                      x2={right[0]}
                      y2={right[1]}
                    />,
                  );
                }

                for (let column = 1; column < columns; column += 1) {
                  const t = column / columns;
                  const top = lerp(tl, tr, t);
                  const bottom = lerp(bl, br, t);
                  gridLines.push(
                    <line
                      key={`column-${column}`}
                      {...common}
                      strokeWidth={Math.max(1, common.strokeWidth * 0.65)}
                      x1={top[0]}
                      y1={top[1]}
                      x2={bottom[0]}
                      y2={bottom[1]}
                    />,
                  );
                }

                return (
                  <g key={`visual-${element.id}`}>
                    <polygon {...common} points={quad.map(([px, py]) => `${px},${py}`).join(' ')}>
                      <title>{element.label || 'table'}</title>
                    </polygon>
                    {gridLines}
                  </g>
                );
              }

              if (type === 'box') {
                return (
                  <polygon {...common} points={pointString}>
                    <title>{element.label || 'box'}</title>
                  </polygon>
                );
              }

              return (
                <polygon {...common} points={pointString}>
                  <title>{element.label || type}</title>
                </polygon>
              );
            })}
          </svg>
        )}

        {/* Drawing rectangle (dashed blue) */}
        {loaded && drawRect && (
          <div
            className="absolute border-2 border-dashed border-blue-500 bg-blue-500/10"
            style={{
              left: `${drawRect.left}px`,
              top: `${drawRect.top}px`,
              width: `${drawRect.width}px`,
              height: `${drawRect.height}px`,
            }}
          />
        )}
      </div>
    </div>
  );
}
