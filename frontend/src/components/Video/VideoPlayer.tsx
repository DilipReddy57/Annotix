import React, { useRef, useState, useEffect } from "react";
import {
  Play,
  Pause,
  Square,
  SkipBack,
  SkipForward,
  Maximize2,
} from "lucide-react";

interface Annotation {
  frame: number;
  bbox: number[];
  label: string;
  score: number;
  timestamp?: number;
}

interface VideoPlayerProps {
  src: string;
  annotations?: Annotation[];
  fps?: number; // Default 30
}

const VideoPlayer: React.FC<VideoPlayerProps> = ({
  src,
  annotations = [],
  fps = 30,
}) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // Sync Canvas with Video
  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!video || !canvas) return;

      if (!video.paused && !video.ended) {
        setCurrentTime(video.currentTime);
      }

      // Resize canvas to match video
      if (video.videoWidth) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext("2d");
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Find annotations for current frame
        const currentFrame = Math.floor(video.currentTime * fps);

        // Simple optimization: filtering array every frame is slow for huge lists,
        // but fine for MVP (<10k items).
        const activeAnns = annotations.filter(
          (a) => Math.abs(a.frame - currentFrame) < 2 // Show for 2 frames window
        );

        activeAnns.forEach((ann) => {
          const [x, y, w, h] = ann.bbox;

          // Draw Box
          ctx.strokeStyle = "#8b5cf6"; // Violet
          ctx.lineWidth = 3;
          ctx.strokeRect(x, y, w, h);

          // Draw Label
          ctx.fillStyle = "rgba(139, 92, 246, 0.8)";
          ctx.fillRect(x, y - 20, ctx.measureText(ann.label).width + 10, 20);
          ctx.fillStyle = "white";
          ctx.font = "bold 14px Inter";
          ctx.fillText(ann.label, x + 5, y - 5);
        });
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, [annotations, fps]);

  const togglePlay = () => {
    if (videoRef.current) {
      if (isPlaying) {
        videoRef.current.pause();
      } else {
        videoRef.current.play();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const onLoadedMetadata = () => {
    if (videoRef.current) setDuration(videoRef.current.duration);
  };

  const handleSeek = (e: React.ChangeEvent<HTMLInputElement>) => {
    const time = parseFloat(e.target.value);
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  return (
    <div className="flex flex-col gap-4">
      <div className="relative rounded-2xl overflow-hidden bg-black aspect-video group shadow-2xl shadow-primary/10 border border-white/5">
        <video
          ref={videoRef}
          src={src}
          className="w-full h-full object-contain"
          onLoadedMetadata={onLoadedMetadata}
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          loop
        />
        <canvas
          ref={canvasRef}
          className="absolute inset-0 w-full h-full pointer-events-none"
        />

        {/* Controls Overlay */}
        <div className="absolute bottom-0 left-0 right-0 p-6 bg-gradient-to-t from-black/90 via-black/50 to-transparent opacity-0 group-hover:opacity-100 transition-all duration-300">
          <div className="flex flex-col gap-2">
            {/* Progress Bar */}
            <input
              type="range"
              min="0"
              max={duration || 100}
              value={currentTime}
              onChange={handleSeek}
              className="w-full h-1 bg-white/20 rounded-lg appearance-none cursor-pointer accent-primary"
            />

            <div className="flex items-center justify-between">
              <div className="flex items-center gap-4">
                <button
                  onClick={togglePlay}
                  className="p-2 rounded-full bg-white/10 hover:bg-white/20 text-white transition-colors backdrop-blur-md"
                >
                  {isPlaying ? <Pause size={20} /> : <Play size={20} />}
                </button>
                <span className="text-xs font-mono text-white/80">
                  {currentTime.toFixed(1)}s / {duration.toFixed(1)}s
                </span>
              </div>
              <button className="text-white/70 hover:text-white">
                <Maximize2 size={20} />
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VideoPlayer;
