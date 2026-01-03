import { useEffect, useState, useMemo } from "react";
import AnalysisDisplay from "~/components/analysisDisplay";
import { fakeResultsData } from "~/fakeData/fakeAnalysisData";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";

export default function Analysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [atoken, setAtoken] = useState<string>("");

  const { token } = useAuth();
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
      const response = await fetch("http://127.0.0.1:8000/v1/analysis", {
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
    } catch (err: any) {
      setError(err instanceof Error ? err.message : String(err));
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
    <div className="min-h-screen bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 p-6">
      <div className="mx-auto max-w-6xl">
        <h1 className="text-2xl md:text-3xl text-emerald-100 font-bold mb-6">Image Analysis</h1>

        <div className="flex flex-col md:flex-row gap-6">
          <div className="md:w-1/2 bg-white/5 border border-emerald-700 rounded-xl p-6 shadow-lg">
            <h2 className="text-lg font-semibold text-emerald-100 mb-3">Upload an image</h2>

            {error && (
              <div className="mb-4">
                <Alert type="error">{error}</Alert>
              </div>
            )}

            <label className="block mb-4">
              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="block w-full text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-emerald-500 file:px-4 file:py-2 file:text-white hover:file:bg-emerald-400"
              />
            </label>

            {preview ? (
              <div className="mb-4">
                <img src={preview} alt="preview" className="w-full rounded-md object-contain max-h-60 mx-auto border" />
              </div>
            ) : (
              <div className="mb-4 text-sm text-slate-200">No image selected yet. Use the file control above to choose an image.</div>
            )}

            <div className="flex gap-3">
              <button
                onClick={handleUpload}
                disabled={loading}
                className="flex-1 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white py-3 font-medium disabled:opacity-60 disabled:cursor-not-allowed shadow">
                {loading ? "Analyzing..." : "Upload & Analyze"}
              </button>
              <button
                onClick={() => { setFile(null); setPreview(null); setError(null); }}
                className="px-4 py-3 rounded-lg bg-white/5 hover:bg-white/10 text-emerald-100">
                Clear
              </button>
            </div>
          </div>

          <div className="md:w-1/2 bg-white/5 border border-emerald-700 rounded-xl p-6 shadow-lg">
            <h2 className="text-lg font-semibold text-emerald-100 mb-3">Results</h2>
            <div className="min-h-[240px]">
              {resultContent}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
