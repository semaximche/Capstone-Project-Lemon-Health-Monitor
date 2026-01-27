/**
 * API Configuration
 * 
 * Centralized configuration for API base URL.
 * Change the API_BASE_URL to point to your backend server.
 * 
 * For WebSocket connections, the protocol will be automatically converted (http -> ws, https -> wss)
 */

// API Base URL - Change this to your backend server address
export const API_BASE_URL = "http://127.0.0.1:8000";

// API Version prefix
export const API_VERSION = "/v1";

// Full API base URL with version
export const API_URL = `${API_BASE_URL}${API_VERSION}`;

/**
 * Get WebSocket URL from HTTP URL
 * Converts http:// to ws:// and https:// to wss://
 */
export function getWebSocketUrl(path: string): string {
  const wsBase = API_BASE_URL.replace(/^http/, "ws");
  return `${wsBase}${API_VERSION}${path}`;
}

/**
 * Helper function to build full API endpoint URL
 */
export function apiEndpoint(endpoint: string): string {
  // Remove leading slash if present to avoid double slashes
  const cleanEndpoint = endpoint.startsWith("/") ? endpoint.slice(1) : endpoint;
  return `${API_URL}/${cleanEndpoint}`;
}
