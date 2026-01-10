import { useEffect, useRef, useState } from "react";

export default function AnalysisDisplay({ data }: { data: ResultsDisplay}) {
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [scale, setScale] = useState({ x: 1, y: 1 });

  const handleImageLoad = () => {
    if (!imgRef.current) return;

    const { naturalWidth, naturalHeight, clientWidth, clientHeight } =
      imgRef.current;

    setScale({
      x: clientWidth / naturalWidth,
      y: clientHeight / naturalHeight,
    });
  };

  useEffect(() => {
    if (!imgRef.current) return;

    const img = imgRef.current;

    const updateScale = () => {
      setScale({
        x: img.clientWidth / img.naturalWidth,
        y: img.clientHeight / img.naturalHeight,
      });
    };

    updateScale();

    const observer = new ResizeObserver(updateScale);
    observer.observe(img);

    return () => observer.disconnect();
  }, []);

  return (
    <div className="w-full flex justify-center p-4">
      <div className="relative inline-block">
        <img
          src={
            data.image.startsWith('data:')
              ? data.image
              : `data:image/jpeg;base64,${data.image}`
          }
          alt="Analysis result"
          className="max-w-full h-auto rounded-lg border border-emerald-700 shadow-lg"
        />

        {data.classifications.map((item, idx) => {
          const [x1, y1, x2, y2] = item.box;
          console.log(item);

          return (
            <div
              key={idx}
              className="absolute border-2 border-lime-400 rounded-md"
              style={{
                left: x1 * scale.x,
                top: y1 * scale.y,
                width: (x2 - x1) * scale.x,
                height: (y2 - y1) * scale.y,
              }}
            >
              <div className="absolute -top-6 left-0 bg-lime-400 text-black text-xs font-semibold px-2 py-0.5 rounded">
                {item.efficientnet_class_name}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};