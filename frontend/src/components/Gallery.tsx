import { useEffect, useState } from "react";
import axios from "axios";
import { Download } from "lucide-react";
import { motion } from "framer-motion";

interface GalleryImage {
  status?: string;
  error?: string;
  image_path: string;
  annotations?: Array<{ label: string; score: number }>;
}

const Gallery = () => {
  const [images, setImages] = useState<GalleryImage[]>([]);

  useEffect(() => {
    const fetchImages = async () => {
      try {
        const res = await axios.get("http://localhost:8000/results");
        setImages(Object.values(res.data));
      } catch (e) {
        console.error("Failed to fetch images", e);
      }
    };
    fetchImages();
  }, []);

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {images.map((img, idx) => {
        if (img.status === "failed") {
          return (
            <div
              key={idx}
              className="glass rounded-xl p-6 border border-red-500/50 flex flex-col justify-center items-center text-center gap-2"
            >
              <div className="text-red-500 font-bold">Processing Failed</div>
              <div className="text-sm text-gray-400">{img.error}</div>
              <div className="text-xs text-gray-600 mt-2 break-all">
                {img.image_path}
              </div>
            </div>
          );
        }

        return (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: idx * 0.1 }}
            key={idx}
            className="glass rounded-xl overflow-hidden group relative"
          >
            <div className="aspect-video bg-gray-900 relative">
              <img
                src={`http://localhost:8000/${img.image_path.replace(
                  /\\/g,
                  "/"
                )}`}
                alt={img.image_path}
                className="w-full h-full object-cover"
              />

              <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity bg-black/50 p-4">
                <div className="flex flex-wrap gap-2">
                  {img.annotations &&
                    img.annotations.map((ann, i) => (
                      <span
                        key={i}
                        className="px-2 py-1 bg-primary/80 rounded text-xs text-white"
                      >
                        {ann.label} ({Math.round(ann.score * 100)}%)
                      </span>
                    ))}
                </div>
              </div>
            </div>

            <div className="p-4 flex justify-between items-center">
              <h3 className="font-medium truncate flex-1">
                {img.image_path.split("\\").pop()}
              </h3>
              <button className="p-2 hover:bg-white/10 rounded-lg transition-colors">
                <Download size={18} />
              </button>
            </div>
          </motion.div>
        );
      })}

      {images.length === 0 && (
        <div className="col-span-full text-center py-12 text-gray-500">
          No images processed yet. Upload some data to get started.
        </div>
      )}
    </div>
  );
};

export default Gallery;
