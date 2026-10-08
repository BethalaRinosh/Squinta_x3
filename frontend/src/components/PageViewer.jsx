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

        {/* Detected visual structures: arrows, brackets, tables, boxes, circles, etc. */}
        {loaded && !drawMode && visualElements.map((element) => {
          const x = element.bbox_x;
          const y = element.bbox_y;
          const w = element.bbox_w;
          const h = element.bbox_h;
          if (x == null || y == null || w == null || h == null) return null;

          const type = (element.element_type || 'diagram').toLowerCase();
          const label = element.label || type;

          if (type === 'arrow') {
            return (
              <div
                key={`visual-${element.id}`}
                className="absolute pointer-events-none"
                style={{
                  left: `${x * scaleX}px`,
                  top: `${(y + h / 2) * scaleY}px`,
                  width: `${w * scaleX}px`,
                  height: '2px',
                }}
                title={label}
              >
                <div className="relative w-full h-full bg-blue-500">
                  <div className="absolute right-0 -top-1.5 w-0 h-0 border-t-2 border-b-2 border-l-4 border-t-transparent border-b-transparent border-l-blue-500" />
                </div>
              </div>
            );
          }

          const borderClass =
            type === 'circle' ? 'rounded-full border-blue-500' :
            type === 'table' ? 'border-dashed border-indigo-500' :
            type === 'bracket' ? 'border-dashed border-amber-500' :
            type === 'underline' ? 'border-b-2 border-amber-500' :
            'border-2 border-blue-400';

          return (
            <div
              key={`visual-${element.id}`}
              className={`absolute pointer-events-none ${borderClass}`}
              style={{
                left: `${x * scaleX}px`,
                top: `${y * scaleY}px`,
                width: `${w * scaleX}px`,
                height: `${h * scaleY}px`,
              }}
              title={label}
            >
              {type !== 'underline' && (
                <span className="absolute -top-5 left-0 px-1.5 py-0.5 rounded bg-white/90 border border-gray-200 text-[10px] font-medium text-gray-600 whitespace-nowrap">
                  {label}
                </span>
              )}
            </div>
          );
        })}

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
