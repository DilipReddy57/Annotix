import axios from "axios";

export const API_BASE_URL = "http://localhost:8000";

export const client = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    "Content-Type": "application/json",
  },
});

export const api = {
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

  annotateVideo: async (projectId: string, videoId: string, prompt: string) => {
    const response = await client.post(
      `/api/projects/${projectId}/videos/${videoId}/annotate?prompt=${encodeURIComponent(
        prompt
      )}`
    );
    return response.data;
  },
};
