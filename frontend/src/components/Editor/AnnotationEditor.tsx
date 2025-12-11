import { useState, useEffect, useRef } from "react";
import {
  Layers,
  Eye,
  EyeOff,
  Trash2,
  Save,
  Wand2,
  MousePointer2,
  ZoomIn,
  Undo2,
  Redo2,
  Loader2,
  ChevronLeft,
  ChevronRight,
} from "lucide-react";
import { api } from "../../api/client";

interface Annotation {
  id: string;
  label: string;
  bbox: number[];
  score: number;
  visible?: boolean;
}

interface EditorProps {
  imageUrl: string;
  initialAnnotations: Annotation[];
  onSave: (annotations: Annotation[]) => void;
  projectId: string;
  imageId: string;
  // Navigation props
  currentIndex?: number;
  totalImages?: number;
  onPrevious?: () => void;
  onNext?: () => void;
}

const AnnotationEditor: React.FC<EditorProps> = ({
  imageUrl,
  initialAnnotations,
  onSave,
  projectId,
  imageId,
  currentIndex = 0,
  totalImages = 1,
  onPrevious,
  onNext,
}) => {
  const [annotations, setAnnotations] =
    useState<Annotation[]>(initialAnnotations);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [tool, setTool] = useState<"select" | "segment">("select");
  const [prompt, setPrompt] = useState("");
  const [isProcessing, setIsProcessing] = useState(false);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    setAnnotations(initialAnnotations.map((a) => ({ ...a, visible: true })));
  }, [initialAnnotations]);

  // Color palette for different object instances
  const instanceColors = [
    "#818cf8", // indigo
    "#f472b6", // pink
    "#34d399", // emerald
    "#fbbf24", // amber
    "#60a5fa", // blue
    "#a78bfa", // violet
    "#fb7185", // rose
    "#2dd4bf", // teal
    "#fb923c", // orange
    "#4ade80", // green
    "#c084fc", // purple
    "#38bdf8", // sky
  ];

  const getInstanceColor = (index: number) =>
    instanceColors[index % instanceColors.length];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const img = new Image();
    img.src = imageUrl;
    img.onload = () => {
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0);

      annotations.forEach((ann, index) => {
        if (!ann.visible) return;
        const [x, y, w, h] = ann.bbox;
        const color = getInstanceColor(index);
        const isSelected = selectedId === ann.id;

        // Draw bounding box with instance color
        ctx.strokeStyle = isSelected ? color : `${color}99`; // 99 = 60% opacity
        ctx.lineWidth = isSelected ? 3 : 2;
        ctx.setLineDash(isSelected ? [] : [4, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.setLineDash([]);

        // Draw label for all objects (not just selected)
        const text = `${ann.label} ${(ann.score * 100).toFixed(0)}%`;
        const textWidth = ctx.measureText(text).width;

        // Label background
        ctx.fillStyle = isSelected ? color : `${color}CC`; // CC = 80% opacity
        ctx.fillRect(x, y - 24, textWidth + 10, 24);

        // Label text
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 14px Inter";
        ctx.fillText(text, x + 5, y - 7);

        // Fill overlay for selected
        if (isSelected) {
          ctx.fillStyle = `${color}1A`; // 1A = 10% opacity
          ctx.fillRect(x, y, w, h);
        }
      });
    };
  }, [imageUrl, annotations, selectedId]);

  const handleMagicSegment = async () => {
    if (!prompt.trim()) return;
    setIsProcessing(true);
    try {
      const results = await api.segmentInteractive(projectId, imageId, prompt);
      const newAnns = results.map((r: any) => ({
        id: crypto.randomUUID(),
        label: r.label,
        bbox: r.bbox,
        score: r.score,
        visible: true,
      }));
      setAnnotations([...annotations, ...newAnns]);
      setPrompt("");
      setTool("select");
    } catch (e) {
      console.error("Segmentation failed", e);
      alert("Failed to segment. Check backend logs.");
    } finally {
      setIsProcessing(false);
    }
  };

  const toggleVisibility = (id: string) => {
    setAnnotations(
      annotations.map((a) => (a.id === id ? { ...a, visible: !a.visible } : a))
    );
  };

  const deleteAnnotation = (id: string) => {
    setAnnotations(annotations.filter((a) => a.id !== id));
  };

  return (
    <div className="flex h-screen bg-[#09090b] text-foreground overflow-hidden">
      {/* Floating Toolbar */}
      <div className="absolute left-6 top-1/2 -translate-y-1/2 flex flex-col gap-2 p-2 bg-card/80 backdrop-blur-xl border border-white/10 rounded-2xl shadow-2xl z-50">
        <ToolButton
          icon={<MousePointer2 size={20} />}
          active={tool === "select"}
          onClick={() => setTool("select")}
          tooltip="Select (V)"
        />
        <ToolButton
          icon={<Wand2 size={20} />}
          active={tool === "segment"}
          onClick={() => setTool("segment")}
          tooltip="Magic Wand (W)"
        />
        <div className="h-px bg-white/10 my-1" />
        <ToolButton
          icon={<ZoomIn size={20} />}
          onClick={() => {}}
          tooltip="Zoom In"
        />
      </div>

      {/* Magic Prompt Input (Visible when 'segment' tool is active) */}
      {tool === "segment" && (
        <div className="absolute top-8 left-1/2 -translate-x-1/2 z-50 flex gap-2 animate-in slide-in-from-top-4 duration-300">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleMagicSegment()}
            placeholder="Describe object to segment..."
            className="w-80 bg-black/50 backdrop-blur-xl border border-primary/50 text-white px-4 py-2 rounded-xl focus:outline-none focus:ring-2 focus:ring-primary shadow-2xl shadow-primary/20"
            autoFocus
          />
          <button
            onClick={handleMagicSegment}
            disabled={isProcessing}
            className="bg-primary hover:bg-primary/90 text-white px-4 py-2 rounded-xl font-medium transition-all shadow-lg flex items-center gap-2"
          >
            {isProcessing ? (
              <Loader2 className="animate-spin" size={18} />
            ) : (
              <Wand2 size={18} />
            )}
            Segment
          </button>
        </div>
      )}

      {/* Main Canvas Area */}
      <div className="flex-1 flex items-center justify-center bg-[url('https://grainy-gradients.vercel.app/noise.svg')] bg-opacity-5 relative">
        <div className="relative shadow-2xl shadow-black/50 border border-white/5 rounded-sm overflow-hidden">
          <canvas
            ref={canvasRef}
            className="max-w-[80vw] max-h-[80vh] object-contain"
          />
        </div>

        {/* Bottom Bar with Navigation */}
        <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 px-6 py-3 bg-card/80 backdrop-blur-xl border border-white/10 rounded-full shadow-2xl">
          {/* Previous Image */}
          <button
            onClick={onPrevious}
            disabled={!onPrevious || currentIndex <= 0}
            className="text-muted-foreground hover:text-foreground transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Previous Image"
          >
            <ChevronLeft size={20} />
          </button>

          {/* Image Counter */}
          <span className="text-xs font-mono text-muted-foreground min-w-[60px] text-center">
            {currentIndex + 1} / {totalImages}
          </span>

          {/* Next Image */}
          <button
            onClick={onNext}
            disabled={!onNext || currentIndex >= totalImages - 1}
            className="text-muted-foreground hover:text-foreground transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
            title="Next Image"
          >
            <ChevronRight size={20} />
          </button>

          <div className="w-px h-4 bg-white/10" />

          <button className="text-muted-foreground hover:text-foreground transition-colors">
            <Undo2 size={18} />
          </button>
          <button className="text-muted-foreground hover:text-foreground transition-colors">
            <Redo2 size={18} />
          </button>
          <div className="w-px h-4 bg-white/10" />
          <span className="text-xs font-mono text-muted-foreground">100%</span>
        </div>
      </div>

      {/* Right Inspector Panel */}
      <div className="w-80 border-l border-white/10 bg-card/50 backdrop-blur-md flex flex-col">
        <div className="p-4 border-b border-white/10 flex justify-between items-center bg-card/50">
          <span className="font-semibold text-sm tracking-wide">Inspector</span>
          <button
            onClick={() => onSave(annotations)}
            className="flex items-center gap-2 bg-primary hover:bg-primary/90 text-white px-3 py-1.5 rounded-lg text-xs font-medium transition-all shadow-lg shadow-primary/20"
          >
            <Save size={14} /> Save Changes
          </button>
        </div>

        <div className="flex-1 overflow-y-auto">
          {/* Layers Section */}
          <div className="p-4">
            <div className="flex items-center justify-between mb-3 text-xs font-medium text-muted-foreground uppercase tracking-wider">
              <div className="flex items-center gap-2">
                <Layers size={14} /> Layers
              </div>
              <span>{annotations.length}</span>
            </div>

            <div className="space-y-1">
              {annotations.map((ann) => (
                <div
                  key={ann.id}
                  className={`group p-3 rounded-lg border border-transparent flex justify-between items-center cursor-pointer transition-all duration-200
                    ${
                      selectedId === ann.id
                        ? "bg-primary/10 border-primary/20 shadow-sm"
                        : "hover:bg-white/5 hover:border-white/5"
                    }`}
                  onClick={() => setSelectedId(ann.id)}
                >
                  <div className="flex items-center gap-3">
                    <div
                      className={`w-2 h-2 rounded-full ${
                        selectedId === ann.id
                          ? "bg-primary"
                          : "bg-muted-foreground"
                      }`}
                    />
                    <div>
                      <div
                        className={`text-sm font-medium ${
                          selectedId === ann.id
                            ? "text-primary-foreground"
                            : "text-foreground"
                        }`}
                      >
                        {ann.label}
                      </div>
                      <div className="text-[10px] text-muted-foreground font-mono">
                        {(ann.score * 100).toFixed(1)}% Conf
                      </div>
                    </div>
                  </div>
                  <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggleVisibility(ann.id);
                      }}
                      className="p-1.5 hover:bg-white/10 rounded-md text-muted-foreground hover:text-foreground"
                    >
                      {ann.visible ? <Eye size={14} /> : <EyeOff size={14} />}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteAnnotation(ann.id);
                      }}
                      className="p-1.5 hover:bg-red-500/20 rounded-md text-muted-foreground hover:text-red-400"
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

const ToolButton = ({ icon, active, onClick, tooltip }: any) => (
  <button
    onClick={onClick}
    className={`p-3 rounded-xl transition-all duration-200 relative group ${
      active
        ? "bg-primary text-white shadow-lg shadow-primary/25"
        : "text-muted-foreground hover:text-foreground hover:bg-white/10"
    }`}
    title={tooltip}
  >
    {icon}
    {active && (
      <div className="absolute inset-0 rounded-xl ring-1 ring-inset ring-white/20" />
    )}
  </button>
);

export default AnnotationEditor;
