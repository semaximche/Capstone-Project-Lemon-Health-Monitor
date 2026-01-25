import { useEffect, useRef, useState } from "react";
import type { ResultsDisplay } from "~/types/analysis";

export default function AnalysisDisplay({ data }: { data: ResultsDisplay }) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [scale, setScale] = useState({ x: 1, y: 1 });

  const updateScale = () => {
    if (!imgRef.current) return;

    const img = imgRef.current;

    setScale({
      x: img.clientWidth / img.naturalWidth,
      y: img.clientHeight / img.naturalHeight,
    });
  };

  useEffect(() => {
    if (!imgRef.current) return;

    const observer = new ResizeObserver(updateScale);
    observer.observe(imgRef.current);

    return () => observer.disconnect();
  }, []);

  return (
    <div className="w-full flex justify-center p-4">
      <div className="relative inline-block">
        <img
          ref={imgRef}
          onLoad={updateScale}
          src={
            data.image?.startsWith("data:")
              ? data.image
              : `data:image/jpeg;base64,${data.image}`
          }
          alt="Analysis result"
          className="max-w-full h-auto rounded-xl border-2 border-cyan-500/30 shadow-2xl"
        />

        {data.classifications && data.classifications.map((item, idx) => {
          const [x1, y1, x2, y2] = item.box;

          return (
            <div
              key={idx}
              className="absolute border-2 border-cyan-400 rounded-lg pointer-events-none neon-glow"
              style={{
                left: x1 * scale.x,
                top: y1 * scale.y,
                width: (x2 - x1) * scale.x,
                height: (y2 - y1) * scale.y,
              }}
            >
              <div className="absolute -top-7 left-0 bg-gradient-to-r from-cyan-500 to-teal-500 text-white text-xs font-display font-semibold px-3 py-1 rounded-lg shadow-lg whitespace-nowrap">
                {item.efficientnet_class_name}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}