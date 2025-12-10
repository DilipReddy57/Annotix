"use client";

import { motion, useAnimation, AnimatePresence } from "framer-motion";
import { useEffect, useRef, useState, memo, useCallback } from "react";
// import { Button } from "@/components/ui/button" // Removing circular dep if unused, or keep if needed
import { cn } from "@/lib/utils";
import { Activity } from "lucide-react";

// --- SAMPLE ASSET DATA (Adapted from Products to Annotation Assets) ---
const sampleAssets = [
  // KEY ASSETS - Displayed prominently
  {
    id: 1,
    name: "Autonomous Vehicle Cam 04",
    price: "98% Conf", // Mapped from Price
    score: 98,
    image:
      "https://images.unsplash.com/photo-1494438639946-1ebd1d20bf85?w=300&q=80", // Car/Road
  },
  {
    id: 2,
    name: "Drone Survey - Site B",
    price: "96% Conf",
    score: 96,
    image:
      "https://images.unsplash.com/photo-1473968512647-3e447244af8f?w=300&q=80", // Drone
  },
  {
    id: 3,
    name: "Pedestrian Tracking",
    price: "99% Conf",
    score: 99,
    image:
      "https://images.unsplash.com/photo-1557804506-669a67965ba0?w=300&q=80", // Crowd
  },
  {
    id: 4,
    name: "Industrial Arm Scan",
    price: "94% Conf",
    score: 94,
    image:
      "https://images.unsplash.com/photo-1581091226825-a6a2a5aee158?w=300&q=80", // Robot
  },
  {
    id: 5,
    name: "Satellite Grid V2",
    price: "100% Conf",
    score: 100,
    image:
      "https://images.unsplash.com/photo-1451187580459-43490279c0fa?w=300&q=80", // Satellite
  },

  // BACKGROUND ASSETS
  {
    id: 6,
    name: "LiDAR Point Cloud",
    price: "88%",
    score: 88,
    image:
      "https://images.unsplash.com/photo-1550751827-4bd374c3f58b?w=300&q=80",
  },
  {
    id: 7,
    name: "Thermal Scan",
    price: "92%",
    score: 92,
    image:
      "https://images.unsplash.com/photo-1518544806352-a22860520026?w=300&q=80",
  },
  {
    id: 8,
    name: "Traffic Flow",
    price: "85%",
    score: 85,
    image:
      "https://images.unsplash.com/photo-1566378246598-5b11a0d486cc?w=300&q=80",
  },
  {
    id: 9,
    name: "Medical X-Ray",
    price: "97%",
    score: 97,
    image:
      "https://images.unsplash.com/photo-1530497610245-94d3c16cda48?w=300&q=80",
  },
  {
    id: 10,
    name: "Microscope Slide",
    price: "89%",
    score: 89,
    image:
      "https://images.unsplash.com/photo-1576086213369-97a306d36557?w=300&q=80",
  },
  {
    id: 11,
    name: "Facial Recog",
    price: "95%",
    score: 95,
    image:
      "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?w=300&q=80",
  },
  {
    id: 12,
    name: "OCR Doc Scan",
    price: "99%",
    score: 99,
    image:
      "https://images.unsplash.com/photo-1562240020-ce31ccb0fa7d?w=300&q=80",
  },
  {
    id: 13,
    name: "Wildlife Cam",
    price: "82%",
    score: 82,
    image:
      "https://images.unsplash.com/photo-1504208434309-cb69f4fe52b0?w=300&q=80",
  },
  {
    id: 14,
    name: "Retail Shelf",
    price: "91%",
    score: 91,
    image:
      "https://images.unsplash.com/photo-1556740738-b6a63e27c4df?w=300&q=80",
  },
  {
    id: 15,
    name: "Agri-Field",
    price: "87%",
    score: 87,
    image:
      "https://images.unsplash.com/photo-1625246333195-f8196812c850?w=300&q=80",
  },
];

const keyAssets = sampleAssets.slice(0, 5);
const backgroundAssets = sampleAssets.slice(5);

interface AssetMetadata {
  name: string;
  price: string; // "Confidence"
  score: number; // IoU Score
}

// Helper function to generate random entry/exit points
function getRandomEdgePoint(
  containerSize: { width: number; height: number },
  edge: "top" | "bottom" | "left" | "right"
) {
  const margin = 100;
  switch (edge) {
    case "top":
      return { x: Math.random() * containerSize.width, y: -margin };
    case "bottom":
      return {
        x: Math.random() * containerSize.width,
        y: containerSize.height + margin,
      };
    case "left":
      return { x: -margin, y: Math.random() * containerSize.height };
    case "right":
      return {
        x: containerSize.width + margin,
        y: Math.random() * containerSize.height,
      };
  }
}

// Helper function to create curved path between two points
function createCurvedPath(
  start: { x: number; y: number },
  end: { x: number; y: number },
  containerSize: { width: number; height: number }
) {
  const curveVariation = 30 + Math.random() * 60;
  const midX = (start.x + end.x) / 2 + (Math.random() - 0.5) * curveVariation;
  const midY = (start.y + end.y) / 2 + (Math.random() - 0.5) * curveVariation;
  return [start, { x: midX, y: midY }, end];
}

interface AnimatedAssetProps {
  product: (typeof sampleAssets)[0];
  isKeyProduct?: boolean;
  containerSize: { width: number; height: number };
  onReachCenter?: (metadata: AssetMetadata) => void;
  onComplete?: () => void;
}

function AnimatedAsset({
  product: asset,
  isKeyProduct = false,
  containerSize,
  onReachCenter,
  onComplete,
}: AnimatedAssetProps) {
  const controls = useAnimation();

  useEffect(() => {
    const animateAsset = async () => {
      if (isKeyProduct) {
        // Featured asset animation
        const edges: Array<"top" | "bottom" | "left" | "right"> = [
          "top",
          "bottom",
          "left",
          "right",
        ];
        const entryEdge = edges[Math.floor(Math.random() * edges.length)];

        const startPoint = getRandomEdgePoint(containerSize, entryEdge);
        const centerPoint = {
          x: containerSize.width / 2 - 40,
          y: containerSize.height / 2 - 40,
        };

        await controls.set({
          x: startPoint.x,
          y: startPoint.y,
          scale: 0.7,
          filter: "blur(4px)",
          opacity: 0.8,
        });

        await controls.start({
          x: centerPoint.x,
          y: centerPoint.y,
          scale: 1.8,
          filter: "blur(0px)",
          opacity: 1,
          transition: {
            duration: 3 + Math.random() * 2,
            ease: "easeInOut",
            type: "tween",
          },
        });

        onReachCenter?.({
          name: asset.name,
          price: asset.price,
          score: asset.score,
        });

        await new Promise((resolve) => setTimeout(resolve, 3000));

        const exitEdges: Array<"top" | "bottom" | "left" | "right"> = [
          "top",
          "bottom",
          "left",
          "right",
        ];
        const randomExitEdge =
          exitEdges[Math.floor(Math.random() * exitEdges.length)];
        const randomExitPoint = getRandomEdgePoint(
          containerSize,
          randomExitEdge
        );

        await controls.start({
          x: randomExitPoint.x,
          y: randomExitPoint.y,
          scale: 0.7,
          filter: "blur(4px)",
          opacity: 0.5,
          transition: {
            duration: 2.5 + Math.random() * 1,
            ease: "easeInOut",
            type: "tween",
          },
        });
      } else {
        // Background loop
        const animateLoop = async () => {
          while (true) {
            const edges: Array<"top" | "bottom" | "left" | "right"> = [
              "top",
              "bottom",
              "left",
              "right",
            ];
            const entryEdge = edges[Math.floor(Math.random() * edges.length)];
            const exitEdge = edges[Math.floor(Math.random() * edges.length)];

            const startPoint = getRandomEdgePoint(containerSize, entryEdge);
            const endPoint = getRandomEdgePoint(containerSize, exitEdge);
            const path = createCurvedPath(startPoint, endPoint, containerSize);

            await controls.set({
              x: startPoint.x,
              y: startPoint.y,
              scale: 0.5 + Math.random() * 0.4,
              filter: "blur(2px)",
              opacity: 0.6 + Math.random() * 0.4,
            });

            for (let i = 1; i < path.length; i++) {
              await controls.start({
                x: path[i].x,
                y: path[i].y,
                transition: {
                  duration: 2 + Math.random() * 2,
                  ease: "linear",
                  type: "tween",
                },
              });
            }
            await new Promise((resolve) => setTimeout(resolve, 100));
          }
        };
        animateLoop();
      }

      if (isKeyProduct) onComplete?.();
    };

    animateAsset();
  }, [isKeyProduct, containerSize, controls]);

  return (
    <motion.div
      className="absolute w-16 h-16 md:w-20 md:h-20"
      animate={controls}
      initial={{ scale: 0.5, filter: "blur(2px)", opacity: 0 }}
      style={{ willChange: "transform, opacity, filter" }}
    >
      <div className="relative w-full h-full rounded-lg overflow-hidden border border-border/30 shadow-lg">
        <img
          src={asset.image}
          alt={asset.name}
          className="object-cover w-full h-full"
        />
        <div className="absolute inset-0 bg-black/20" />
      </div>
    </motion.div>
  );
}

function CircularProgress({
  value,
  size = 32,
  className,
}: {
  value: number;
  size?: number;
  className?: string;
}) {
  const percentage = Math.min(Math.max(value, 0), 100);
  return (
    <div
      className={cn("relative flex items-center justify-center", className)}
      style={{ width: size, height: size }}
    >
      <div
        className="absolute inset-0 rounded-full border-[2.5px] border-gray-200 dark:border-gray-700"
        style={{ borderColor: "hsl(var(--muted))" }}
      />
      <div
        className="absolute inset-0 rounded-full"
        style={{
          background: `conic-gradient(from 0deg, hsl(142 76% 36%) 0deg ${
            percentage * 3.6
          }deg, transparent ${percentage * 3.6}deg 360deg)`,
          mask: `radial-gradient(circle at center, transparent ${
            size / 2 - 3
          }px, black ${size / 2 - 2}px)`,
          WebkitMask: `radial-gradient(circle at center, transparent ${
            size / 2 - 3
          }px, black ${size / 2 - 2}px)`,
        }}
      />
      <span className="relative text-xs font-bold text-green-600 dark:text-green-400 z-10">
        {value}
      </span>
    </div>
  );
}

const MetadataDisplay = memo(function MetadataDisplay({
  metadata,
}: {
  metadata: AssetMetadata;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.8 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.8 }}
      transition={{ duration: 0.3, ease: "easeInOut" }}
      className="absolute inset-0 flex items-center justify-center pointer-events-none z-20"
    >
      <div className="relative w-20 h-20 md:w-24 md:h-24">
        {/* Confidence Bubble */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5, x: 15 }}
          animate={{ opacity: 1, scale: 1, x: 0 }}
          transition={{ delay: 0.1, duration: 0.3 }}
          className="absolute left-0 top-1/2 transform -translate-x-full -translate-y-1/2 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border border-gray-200/30 dark:border-gray-700/30 rounded-lg p-2.5 shadow-lg"
        >
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
              <Activity className="w-3 h-3 text-white" />
            </div>
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                Conf.
              </div>
              <div className="text-sm font-bold text-gray-900 dark:text-white">
                {metadata.price}
              </div>
            </div>
          </div>
        </motion.div>

        {/* IoU Bubble */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5, x: -15 }}
          animate={{ opacity: 1, scale: 1, x: 0 }}
          transition={{ delay: 0.2, duration: 0.3 }}
          className="absolute right-0 top-1/2 transform translate-x-full -translate-y-1/2 bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border border-gray-200/30 dark:border-gray-700/30 rounded-lg p-2.5 shadow-lg"
        >
          <div className="flex items-center gap-2">
            <CircularProgress value={metadata.score} size={32} />
            <div>
              <div className="text-xs text-gray-500 dark:text-gray-400 font-medium">
                IoU
              </div>
              <div className="text-sm font-bold text-green-600 dark:text-green-400">
                {metadata.score}/100
              </div>
            </div>
          </div>
        </motion.div>

        {/* Title Bubble */}
        <motion.div
          initial={{ opacity: 0, scale: 0.5, y: 15 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          className="absolute top-0 left-1/2 transform -translate-x-1/2 -translate-y-full bg-white/95 dark:bg-gray-900/95 backdrop-blur-sm border border-gray-200/30 dark:border-gray-700/30 rounded-lg p-3 shadow-lg min-w-[200px]"
        >
          <div className="text-sm font-semibold text-gray-900 dark:text-white text-center">
            {metadata.name}
          </div>
        </motion.div>
      </div>
    </motion.div>
  );
});

export function SpotlightHero() {
  const containerRef = useRef<HTMLDivElement>(null);
  const [containerSize, setContainerSize] = useState({
    width: 800,
    height: 600,
  });
  const [currentMetadata, setCurrentMetadata] = useState<AssetMetadata | null>(
    null
  );
  const [keyAssetIndex, setKeyAssetIndex] = useState(0);
  const [isKeyAssetAnimating, setIsKeyAssetAnimating] = useState(true);
  const [backgroundAssetInstances] = useState(() =>
    Array.from({ length: 15 }, (_, i) => ({
      id: `bg-${i}`,
      product: backgroundAssets[i % backgroundAssets.length],
    }))
  );

  useEffect(() => {
    const updateSize = () => {
      if (containerRef.current) {
        const rect = containerRef.current.getBoundingClientRect();
        setContainerSize({ width: rect.width, height: rect.height });
      }
    };
    updateSize();
    window.addEventListener("resize", updateSize);
    return () => window.removeEventListener("resize", updateSize);
  }, []);

  const handleKeyComplete = useCallback(() => {
    setIsKeyAssetAnimating(false);
    setTimeout(() => {
      setKeyAssetIndex((prev) => (prev + 1) % keyAssets.length);
      setIsKeyAssetAnimating(true);
    }, 100);
  }, []);

  return (
    <div className="w-full h-full relative" ref={containerRef}>
      {/* Background Assets */}
      {backgroundAssetInstances.map((item) => (
        <AnimatedAsset
          key={item.id}
          product={item.product}
          containerSize={containerSize}
        />
      ))}

      {/* Featured Asset */}
      {isKeyAssetAnimating && (
        <AnimatedAsset
          product={keyAssets[keyAssetIndex]}
          isKeyProduct={true}
          containerSize={containerSize}
          onReachCenter={setCurrentMetadata}
          onComplete={handleKeyComplete}
        />
      )}

      <AnimatePresence mode="wait">
        {currentMetadata && <MetadataDisplay metadata={currentMetadata} />}
      </AnimatePresence>
    </div>
  );
}
