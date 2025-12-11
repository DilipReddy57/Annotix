import axios from "axios";

// Use environment variable for API URL (empty string uses same origin with Netlify redirects)
export const API_BASE_URL = import.meta.env.VITE_API_URL || "";

export const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const api = {
  // Segmentation
  segmentInteractive: async (
    projectId: string,
    imageId: string,
    prompt: string
  ) => {
    const response = await client.post(
      `/api/projects/${projectId}/images/${imageId}/segment?prompt=${encodeURIComponent(
        prompt
      )}`
    );
    return response.data;
  },

  // Video annotation
  annotateVideo: async (projectId: string, videoId: string, prompt: string) => {
    const response = await client.post(
      `/api/projects/${projectId}/videos/${videoId}/annotate?prompt=${encodeURIComponent(
        prompt
      )}`
    );
    return response.data;
  },

  // LLM Auto-Prompt Generation
  generatePrompt: async (imageUrl: string) => {
    const response = await client.post("/api/ai/generate-prompt", { imageUrl });
    return response.data;
  },

  // RAG Label Suggestions
  getLabelSuggestions: async (imageUrl: string, currentLabels?: string[]) => {
    const response = await client.post("/api/ai/label-suggestions", {
      imageUrl,
      currentLabels,
    });
    return response.data;
  },

  // Batch Annotation
  batchAnnotate: async (
    projectId: string,
    imageIds: string[],
    prompt: string
  ) => {
    const response = await client.post(
      `/api/projects/${projectId}/batch-annotate`,
      { imageIds, prompt }
    );
    return response.data;
  },

  // Export to COCO format
  exportCoco: async (projectId: string) => {
    const response = await client.get(
      `/api/projects/${projectId}/export/coco`,
      { responseType: "blob" }
    );
    return response.data;
  },

  // Health check
  healthCheck: async () => {
    try {
      const response = await client.get("/health");
      return { ok: true, data: response.data };
    } catch {
      return { ok: false, data: null };
    }
  },
};
