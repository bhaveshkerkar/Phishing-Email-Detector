import axios from "axios";

const BASE_URL = "http://127.0.0.1:8000";

export async function analyzeEmail(emailData) {
  try {
    const response = await axios.post(`${BASE_URL}/analyze`, emailData);
    return { success: true, data: response.data };
  } catch (error) {
    const msg =
      error?.response?.data?.detail ||
      "Could not connect to backend. Make sure the server is running on port 8000.";
    return { success: false, error: msg };
  }
}
