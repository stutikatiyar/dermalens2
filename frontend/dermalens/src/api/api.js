import axios from "axios";

function getBackendURL() {
  const hostname = window.location.hostname;

  // GitHub Codespaces
  if (hostname.includes("-5173.app.github.dev")) {
    return `https://${hostname.replace(
      "-5173.app.github.dev",
      "-8000.app.github.dev"
    )}`;
  }

  // Local development
  return "http://127.0.0.1:8000";
}

const API = axios.create();

/**
 * Upload image to backend and normalize the response
 * for the frontend.
 */
export const uploadImage = async (file) => {
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await API.post(
      `${getBackendURL()}/upload`,
      formData
    );

    const data = response.data;

    // Backend returns the enhanced image as `image`.
    // Frontend expects `enhanced_image`.
    if (data.image && !data.enhanced_image) {
      data.enhanced_image = data.image;
    }

    return data;
  } catch (error) {
    console.error("API upload error:", error);
    throw error;
  }
};

export { getBackendURL };