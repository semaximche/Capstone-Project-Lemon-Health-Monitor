import { useEffect, useState } from "react";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";
import AnalysisDisplay from "~/components/analysisDisplay";
import Markdown from "react-markdown";
import type { AnalysisBox } from "~/types/analysis";

interface AnalysisDetails {
  id: string;
  description: string;
  summary: string;
  image?: string; // Base64 encoded image
}

export default function Dashboard() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [analysisDetails, setAnalysisDetails] = useState<AnalysisDetails | null>(null);
  const [resultsArrayData, setResultsArrayData] = useState<Array<AnalysisBox> | null>(null);

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

        // Map the response to match AnalysisResponse structure: id, description, summary, image
        setAnalysisDetails({
          id: data.id || selectedAnalysis.id || '',
          description: data.description || selectedAnalysis.description || '',
          summary: data.summary || selectedAnalysis.summary || '',
          image: data.image || undefined, // Base64 encoded image
        });
      } catch (e) {
        if (cancelled) return;
        console.error('error fetching analysis details', e);
        setError(e instanceof Error ? e.message : 'Failed to load analysis details');

        // Fallback to cached data if available
        if (selectedAnalysis && (selectedAnalysis.description || selectedAnalysis.summary || selectedAnalysis.image)) {
          setAnalysisDetails({
            id: selectedAnalysis.id || '',
            description: selectedAnalysis.description || '',
            summary: selectedAnalysis.summary || '',
            image: selectedAnalysis.image, // Use cached image if available
          });
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    };

    fetchDetails();
    return () => { cancelled = true; };
  }, [selectedAnalysis?.id, token]);

  // parsing analysisDetails safetly
  useEffect(() => {
    if (!analysisDetails?.description) {
      setResultsArrayData(null);
      return;
    }

    try {
      analysisDetails.description = analysisDetails.description.replace(/'/g, '"');
      const parsed = JSON.parse(analysisDetails.description);
      console.log(parsed);
      setResultsArrayData(parsed);

    } catch (err) {
      console.error("Failed to parse analysis description", err);
      setResultsArrayData(null);
    }
  }, [analysisDetails]);

  if (!selectedAnalysis) {
    return (
      <div className="min-h-screen p-6 relative">
        <div className="mx-auto max-w-5xl">
          <div className="glass-panel rounded-2xl p-12 shadow-2xl text-center relative overflow-hidden scan-line">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent" />
            <h1 className="text-2xl sm:text-3xl md:text-4xl font-display font-bold text-gradient-cyan mb-4 relative z-10">Analysis Dashboard</h1>
            <p className="text-cyan-200/80 text-base sm:text-lg relative z-10 px-4">No analysis selected. Click on a notification to view analysis details.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen p-6 md:p-8 relative">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div className="flex-1 min-w-0">
            <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-display font-bold text-gradient-cyan mb-2 break-words">Analysis Details</h1>
            <p className="text-cyan-300/60 text-xs sm:text-sm font-mono tracking-wider break-all">ID: {analysisDetails?.id || selectedAnalysis.id}</p>
          </div>
          <button
            onClick={() => navigate('/analysis')}
            className="w-full sm:w-auto px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-semibold shadow-lg neon-glow hover-lift transition-all duration-300"
          >
            <span className="flex items-center justify-center gap-2">
              <span>+</span>
              <span>New Analysis</span>
            </span>
          </button>
        </div>

        {error && (
          <Alert type="error">{error}</Alert>
        )}

        {loading && !analysisDetails ? (
          <div className="glass-panel rounded-2xl p-12 shadow-2xl text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-transparent animate-pulse" />
            <div className="relative z-10">
              <div className="inline-block w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin mb-4" />
              <div className="text-cyan-300 text-lg font-medium">Loading analysis details...</div>
            </div>
          </div>
        ) : analysisDetails ? (
          <div className="space-y-8">
            {/* Analysis Image - Featured prominently */}
            <div className="glass-panel rounded-2xl p-8 shadow-2xl relative overflow-hidden hover-lift">
              <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
              <div className="relative z-10">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-1 h-8 bg-gradient-to-b from-cyan-500 to-teal-500 rounded-full" />
                  <h2 className="text-xl sm:text-2xl font-display font-semibold text-cyan-100">Visual Analysis</h2>
                </div>
                {(analysisDetails.image && resultsArrayData) ? (
                  <div className="rounded-xl overflow-hidden border border-cyan-500/20 shadow-xl">
                    <AnalysisDisplay data={{ image: analysisDetails.image, classifications: resultsArrayData }} />
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-64 border-2 border-dashed border-cyan-500/30 rounded-xl">
                    <p className="text-cyan-300/60">No visual analysis data available.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Summary - Rich content area */}
            <div className="glass-panel rounded-2xl p-8 shadow-2xl relative overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-br from-teal-500/5 to-transparent" />
              <div className="relative z-10">
                <div className="flex items-center gap-3 mb-6">
                  <div className="w-1 h-8 bg-gradient-to-b from-teal-500 to-cyan-500 rounded-full" />
                  <h2 className="text-xl sm:text-2xl font-display font-semibold text-cyan-100">Analysis Summary</h2>
                </div>
                <div className="prose prose-invert prose-cyan max-w-none">
                  <div className="text-cyan-100/90 leading-relaxed space-y-4">
                    <Markdown>
                      {analysisDetails.summary || 'No summary available.'}
                    </Markdown>
                  </div>
                </div>
              </div>
            </div>
          </div>
        ) : (
          <div className="glass-panel rounded-2xl p-12 shadow-2xl text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-red-500/10 to-transparent" />
            <p className="text-cyan-300/80 text-lg relative z-10">Unable to load analysis details.</p>
          </div>
        )}
      </div>
    </div>
  );
}

