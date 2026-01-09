import { useEffect, useState } from "react";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";

interface AnalysisDetails {
  id: string;
  description: string;
  summary: string;
}

export default function Dashboard() {
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);
    const [analysisDetails, setAnalysisDetails] = useState<AnalysisDetails | null>(null);

    const { token, selectedAnalysis } = useAuth();
    const navigate = useNavigate();

    useEffect(() => {
      if (!token) {
        navigate('/signin');
        return;
      }
    }, [token, navigate]);

    useEffect(() => {
      let cancelled = false;
      const fetchDetails = async () => {
        if (!token || !selectedAnalysis || !selectedAnalysis.id) {
          // If no selected analysis, show empty state
          setAnalysisDetails(null);
          return;
        }

        setLoading(true);
        setError(null);
        
        try {
          const resp = await fetch(`http://127.0.0.1:8000/v1/me/analysis/${selectedAnalysis.id}`, {
            headers: { 
              Authorization: `Bearer ${token}`, 
              Accept: 'application/json' 
            },
          });
          
          if (!resp.ok) {
            if (resp.status === 401) {
              throw new Error('Session expired or unauthorized. Please sign in again.');
            }
            throw new Error(`Failed to fetch analysis details (${resp.status})`);
          }
          
          const data = await resp.json();
          if (cancelled) return;
          
          // Map the response to match AnalysisResponse structure: id, description, summary
          setAnalysisDetails({
            id: data.id || selectedAnalysis.id || '',
            description: data.description || selectedAnalysis.description || '',
            summary: data.summary || selectedAnalysis.summary || '',
          });
        } catch (e) {
          if (cancelled) return;
          console.error('error fetching analysis details', e);
          setError(e instanceof Error ? e.message : 'Failed to load analysis details');
          
          // Fallback to cached data if available
          if (selectedAnalysis && (selectedAnalysis.description || selectedAnalysis.summary)) {
            setAnalysisDetails({
              id: selectedAnalysis.id || '',
              description: selectedAnalysis.description || '',
              summary: selectedAnalysis.summary || '',
            });
          }
        } finally {
          if (!cancelled) setLoading(false);
        }
      };

      fetchDetails();
      return () => { cancelled = true; };
    }, [selectedAnalysis?.id, token]);

    if (!selectedAnalysis) {
      return (
        <div className="min-h-screen bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 p-6">
          <div className="mx-auto max-w-4xl">
            <div className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg text-center">
              <h1 className="text-2xl font-bold text-emerald-100 mb-4">Analysis Dashboard</h1>
              <p className="text-emerald-200">No analysis selected. Click on a notification to view analysis details.</p>
            </div>
          </div>
        </div>
      );
    }

    return (
      <div className="min-h-screen bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 p-6">
        <div className="mx-auto max-w-4xl space-y-6">
          <div className="flex items-center justify-between">
            <h1 className="text-2xl md:text-3xl text-emerald-100 font-bold">Analysis Details</h1>
            <button
              onClick={() => navigate('/analysis')}
              className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white font-medium"
            >
              New Analysis
            </button>
          </div>

          {error && (
            <Alert type="error">{error}</Alert>
          )}

          {loading && !analysisDetails ? (
            <div className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg text-center">
              <div className="text-emerald-200">Loading analysis details...</div>
            </div>
          ) : analysisDetails ? (
            <div className="space-y-6">
              {/* Analysis ID */}
              <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 shadow-lg">
                <h2 className="text-sm font-semibold text-emerald-300 uppercase tracking-wide mb-2">Analysis ID</h2>
                <p className="text-lg text-emerald-100 font-mono">{analysisDetails.id}</p>
              </div>

              {/* Summary */}
              <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 shadow-lg">
                <h2 className="text-lg font-semibold text-emerald-100 mb-4">Summary</h2>
                <p className="text-emerald-200 whitespace-pre-wrap leading-relaxed">
                  {analysisDetails.summary || 'No summary available.'}
                </p>
              </div>

              {/* Description */}
              <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 shadow-lg">
                <h2 className="text-lg font-semibold text-emerald-100 mb-4">Description</h2>
                <p className="text-emerald-200 whitespace-pre-wrap leading-relaxed">
                  {analysisDetails.description || 'No description available.'}
                </p>
              </div>
            </div>
          ) : (
            <div className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg text-center">
              <p className="text-emerald-200">Unable to load analysis details.</p>
            </div>
          )}
        </div>
      </div>
    );
}

