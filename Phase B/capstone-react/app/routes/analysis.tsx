import { useEffect, useState, useMemo } from "react";
import AnalysisDisplay from "~/components/analysisDisplay";
import { fakeResultsData } from "~/fakeData/fakeAnalysisData";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";
import { apiEndpoint } from "~/lib/api-config";

export default function Analysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [waitingForNotification, setWaitingForNotification] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [atoken, setAtoken] = useState<string>("");

  const { token, notifications } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!token) {
      navigate("/signin");
      return;
    }
    setAtoken(token || "");
  }, [token]);

  useEffect(() => {
    if (!file) {
      setPreview(null);
      return;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  // Clear waiting state when a new notification arrives (analysis completed)
  useEffect(() => {
    if (waitingForNotification && notifications.length > 0) {
      // Check if the most recent notification is unread (likely the one we're waiting for)
      const latestNotification = notifications[0];
      if (!latestNotification.read) {
        // Analysis completed, clear waiting state
        setWaitingForNotification(false);
      }
    }
  }, [notifications, waitingForNotification]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", file);

    try {
      setLoading(true);
      setError(null);
      const response = await fetch(apiEndpoint("analysis"), {
        method: "POST",
        headers: {
          Accept: "application/json",
          Authorization: `Bearer ` + atoken,
        },
        body: formData,
      });

      if (!response.ok) {
        if (response.status === 401) {
          throw new Error("Session expired or unauthorized. Please sign in again.");
        }
        let msg = `Request failed (${response.status})`;
        try {
          const contentType = response.headers.get("content-type") || "";
          if (contentType.includes("application/json")) {
            const err = await response.json();
            msg = err.message || err.detail || JSON.stringify(err) || msg;
          } else {
            const text = await response.text();
            msg = text || msg;
          }
        } catch (e) {}
        throw new Error(msg);
      }

      const data = await response.json();
      // TODO: replace fakeResultsData with `data` when backend returns structured results
      console.log("analysis result", data);
      
      // After successful upload, show waiting state until notification arrives
      setWaitingForNotification(true);
    } catch (err: any) {
      setError(err instanceof Error ? err.message : String(err));
      setWaitingForNotification(false);
      if (err instanceof Error && err.message.toLowerCase().includes("session expired")) {
        setTimeout(() => {
          navigate("/signin");
        }, 1200);
      }
    } finally {
      setLoading(false);
    }
  };

  const resultContent = useMemo(() => {
    return <AnalysisDisplay data={fakeResultsData} />;
  }, []);

  return (
    <div className="min-h-screen p-6 md:p-8">
      <div className="mx-auto max-w-6xl space-y-6">
        <div className="glass-panel rounded-2xl p-6 shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
          <h1 className="text-xl sm:text-2xl md:text-3xl font-display font-bold text-gradient-cyan relative z-10">Image Analysis</h1>
        </div>

        <div className="flex flex-col md:flex-row gap-6">
          <div className="md:w-1/2 glass-panel rounded-xl p-6 shadow-2xl relative overflow-hidden">
            <div className="absolute top-0 right-0 w-48 h-48 bg-cyan-500/5 rounded-full blur-3xl" />
            <h2 className="text-lg font-display font-semibold text-cyan-100 mb-3 relative z-10">Upload an image</h2>

            {error && (
              <div className="mb-4">
                <Alert type="error">{error}</Alert>
              </div>
            )}

            <label className="block mb-4 relative z-10">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="block w-full text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-gradient-to-r file:from-cyan-500 file:to-teal-500 file:px-4 file:py-2 file:text-white hover:file:from-cyan-400 hover:file:to-teal-400"
              />
            </label>

            {preview ? (
              <div className="mb-4 relative z-10">
                <img src={preview} alt="preview" className="w-full rounded-xl object-contain max-h-60 mx-auto border border-cyan-500/20 shadow-lg" />
              </div>
            ) : (
              <div className="mb-4 text-sm text-cyan-200/70 relative z-10">No image selected yet. Use the file control above to choose an image.</div>
            )}

            <div className="flex flex-col sm:flex-row gap-3 relative z-10">
              <button
                onClick={handleUpload}
                disabled={loading || waitingForNotification}
                className="flex-1 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white py-3 text-sm sm:text-base font-semibold disabled:opacity-60 disabled:cursor-not-allowed shadow-lg neon-glow hover-lift transition-all duration-300">
                {loading ? "Uploading..." : waitingForNotification ? "Processing..." : "Upload & Analyze"}
              </button>
              <button
                onClick={() => { 
                  setFile(null); 
                  setPreview(null); 
                  setError(null);
                  setWaitingForNotification(false);
                }}
                disabled={waitingForNotification}
                className="w-full sm:w-auto px-4 py-3 rounded-xl glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 hover-lift transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed">
                Clear
              </button>
            </div>
          </div>

          {/* Loading State - Shows while waiting for notification */}
          {waitingForNotification && (
            <div className="md:w-1/2 glass-panel rounded-xl p-8 shadow-2xl relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-teal-500/10 animate-pulse" />
              <div className="relative z-10 flex flex-col items-center justify-center min-h-[300px] py-8">
                <div className="relative mb-6">
                  <div className="inline-block w-16 h-16 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin"></div>
                  <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-8 h-8 bg-gradient-to-r from-cyan-500 to-teal-500 rounded-full animate-pulse"></div>
                  </div>
                </div>
                <h3 className="text-xl sm:text-2xl font-display font-semibold text-cyan-100 mb-3 text-center">Analysis in Progress</h3>
                <p className="text-cyan-200/80 text-center mb-2 max-w-md">
                  Your image has been uploaded successfully!
                </p>
                <p className="text-cyan-300/70 text-sm text-center mb-4 max-w-md">
                  Our AI models are processing your image. This may take a few moments...
                </p>
                <div className="flex items-center gap-2 text-cyan-400/60 text-xs">
                  <svg className="w-4 h-4 animate-pulse" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6 6 0 10-12 0v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
                  </svg>
                  <span>You'll receive a notification when the analysis is complete</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
