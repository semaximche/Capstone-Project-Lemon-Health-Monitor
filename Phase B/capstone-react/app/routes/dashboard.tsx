import { useEffect, useState } from "react";
import AnalysisDisplay from "~/components/analysisDisplay";
import { fakeResultsData } from "~/fakeData/fakeAnalysisData";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";

export default function Dashboard() {
    const [file, setFile] = useState<File | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<String | null>(null);
    const [atoken, setAtoken] = useState<String>('');
    const [displayFromServer, setDisplayFromServer] = useState<any | null>(null);

    const { token } = useAuth();
    const { selectedAnalysis } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
      if (!token) {
        navigate('/signin');
        return;
      }
      setAtoken(token);
    }, [token]);

    useEffect(() => {
      let cancelled = false;
      const fetchDetails = async () => {
        if (!token || !selectedAnalysis || !selectedAnalysis.id) return;
        setLoading(true);
        try {
          const resp = await fetch(`http://127.0.0.1:8000/v1/me/analysis/${selectedAnalysis.id}`, {
            headers: token ? { Authorization: `Bearer ${token}`, Accept: 'application/json' } : { Accept: 'application/json' },
          });
          if (!resp.ok) {
            console.warn('failed to fetch /v1/me/analysis', resp.status);
            return;
          }
          const data = await resp.json();
          if (cancelled) return;
          setDisplayFromServer({
            id: data.id,
            user_id: data.user_id || selectedAnalysis.user_id,
            presigned_url: data.presigned_url || selectedAnalysis.presigned_url,
            description: data.description || selectedAnalysis.description,
            summary: data.summary || selectedAnalysis.summary,
          });
        } catch (e) {
          console.error('error fetching analysis details', e);
        } finally {
          if (!cancelled) setLoading(false);
        }
      };

      fetchDetails();
      return () => { cancelled = true; };
    }, [selectedAnalysis?.id, token]);

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
            console.log(atoken);
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
                // unauthorized - token may be invalid
                throw new Error('Session expired or unauthorized. Please sign in again.');
              }
              let msg = `Request failed (${response.status})`;
              try {
                const contentType = response.headers.get('content-type') || '';
                if (contentType.includes('application/json')) {
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
        } catch (err) {
            setError(err instanceof Error ? err.message : "Unknown error");
            if (err instanceof Error && err.message.toLowerCase().includes('session expired')) {
              // navigate to signin after brief delay
              setTimeout(() => { navigate('/signin'); }, 1200);
            }
        } finally {
            setLoading(false);
        }
    };


    return (
    <div className="min-h-screen bg-black p-6">
      <div className="mx-auto max-w-3xl space-y-6">
        <div className="flex aspect-square items-center justify-center rounded-xl bg-black shadow border-2 border-white text-white">
          <div className="w-full max-w-sm space-y-4 p-6 text-center">
            <h2 className="text-xl font-semibold">Upload Image</h2>
        
            <input
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              className="block w-full text-sm file:mr-4 file:rounded-lg file:border-0
                         file:bg-blue-600 file:px-4 file:py-2
                         file:text-white hover:file:bg-blue-700"/>

            <button
              onClick={handleUpload}
              disabled={loading}
              className="w-full rounded-lg bg-blue-600 px-4 py-2
                         font-medium text-white hover:bg-blue-700
                         disabled:cursor-not-allowed disabled:opacity-50">
              {loading ? "Analyzing..." : "Upload & Analyze"}
            </button>

            {error && <p className="text-sm text-red-600">{error}</p>}
          </div>
        </div>

        <div className="flex aspect-square items-center justify-center rounded-xl bg-black shadow border-2 border-white text-white ">
          <div className="w-full p-6">
            <h2 className="mb-4 text-xl font-semibold">Results</h2>

            {selectedAnalysis ? (
              <div className="space-y-4 text-left text-white">
                {selectedAnalysis.presigned_url && (
                  <img src={selectedAnalysis.presigned_url} alt="analysis" className="max-w-full rounded" />
                )}
                <div className="bg-white/5 p-4 rounded">
                  <h3 className="font-semibold">Summary</h3>
                  <p className="text-sm">{selectedAnalysis.summary}</p>
                </div>
                <div className="bg-white/5 p-4 rounded">
                  <h3 className="font-semibold">Details</h3>
                  <p className="text-sm">{selectedAnalysis.description}</p>
                </div>
              </div>
            ) : (
              <AnalysisDisplay data={fakeResultsData} />
            )}

          </div>
        </div>

      </div>
    </div>
  );
}