import { useEffect, useState } from "react";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";

interface AnalysisItem {
  id: string;
  description: string;
  summary: string;
  created_at?: string;
}

export default function Analyses() {
  const [analyses, setAnalyses] = useState<AnalysisItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hasMore, setHasMore] = useState(true);
  const [offset, setOffset] = useState(0);
  const limit = 10;

  const { token, setSelectedAnalysis } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    if (!token) {
      navigate('/signin');
      return;
    }
  }, [token, navigate]);

  const fetchAnalyses = async (currentOffset: number, append: boolean = false) => {
    if (!token) return;

    const loadingState = append ? setLoadingMore : setLoading;
    loadingState(true);
    setError(null);

    try {
      const resp = await fetch(`http://127.0.0.1:8000/v1/me/analysis/?limit=${limit}&offset=${currentOffset}`, {
        headers: {
          Authorization: `Bearer ${token}`,
          Accept: 'application/json'
        },
      });

      if (!resp.ok) {
        if (resp.status === 401) {
          throw new Error('Session expired or unauthorized. Please sign in again.');
        }
        throw new Error(`Failed to fetch analyses (${resp.status})`);
      }

      const data = await resp.json();
      
      // Handle different response formats
      // Could be { items: [...], total: number } or just an array
      const items = Array.isArray(data) ? data : (data.items || data.analyses || []);
      const total = data.total || data.count;
      
      if (append) {
        setAnalyses(prev => [...prev, ...items]);
      } else {
        setAnalyses(items);
      }

      // Check if there are more items to load
      if (Array.isArray(data)) {
        // If it's just an array, check if we got less than limit
        setHasMore(items.length === limit);
      } else {
        // If we have total count, compare with current offset + items length
        const newOffset = currentOffset + items.length;
        setHasMore(total ? newOffset < total : items.length === limit);
      }

      setOffset(currentOffset + items.length);
    } catch (e) {
      console.error('error fetching analyses', e);
      setError(e instanceof Error ? e.message : 'Failed to load analyses');
    } finally {
      loadingState(false);
    }
  };

  useEffect(() => {
    if (token) {
      fetchAnalyses(0, false);
    }
  }, [token]);

  const handleLoadMore = () => {
    fetchAnalyses(offset, true);
  };

  const handleAnalysisClick = (analysis: AnalysisItem) => {
    // Set selected analysis and navigate to dashboard
    setSelectedAnalysis({
      id: analysis.id,
      user_id: '', // Will be fetched from API if needed
      description: analysis.description,
      summary: analysis.summary,
    });
    navigate('/dashboard');
  };

  const formatDate = (dateString?: string) => {
    if (!dateString) return 'Date unknown';
    try {
      return new Date(dateString).toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return dateString;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 p-6">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="flex items-center justify-between">
          <h1 className="text-2xl md:text-3xl text-emerald-100 font-bold">My Analyses</h1>
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

        {loading && analyses.length === 0 ? (
          <div className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg text-center">
            <div className="text-emerald-200">Loading analyses...</div>
          </div>
        ) : analyses.length === 0 ? (
          <div className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg text-center">
            <h2 className="text-xl font-semibold text-emerald-100 mb-2">No analyses found</h2>
            <p className="text-emerald-200 mb-4">You haven't created any analyses yet.</p>
            <button
              onClick={() => navigate('/analysis')}
              className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white font-medium"
            >
              Create Your First Analysis
            </button>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {analyses.map((analysis) => (
                <div
                  key={analysis.id}
                  onClick={() => handleAnalysisClick(analysis)}
                  className="bg-white/5 border border-emerald-700 rounded-xl p-5 shadow-lg hover:bg-white/10 hover:border-emerald-500 cursor-pointer transition-all duration-200 flex flex-col"
                >
                  <div className="flex items-start justify-between mb-3">
                    <h3 className="text-sm font-semibold text-emerald-300 uppercase tracking-wide">
                      Analysis {analysis.id.slice(0, 8)}
                    </h3>
                    <span className="text-xs text-emerald-400">
                      {formatDate(analysis.created_at)}
                    </span>
                  </div>
                  
                  {analysis.summary && (
                    <p className="text-emerald-200 text-sm mb-3 line-clamp-3 flex-grow">
                      {analysis.summary}
                    </p>
                  )}
                  
                  {analysis.description && (
                    <p className="text-emerald-300 text-xs line-clamp-2 opacity-75">
                      {analysis.description}
                    </p>
                  )}
                  
                  {!analysis.summary && !analysis.description && (
                    <p className="text-emerald-400 text-sm italic">No details available</p>
                  )}
                  
                  <div className="mt-4 pt-3 border-t border-emerald-700/50">
                    <span className="text-xs text-emerald-400">Click to view details →</span>
                  </div>
                </div>
              ))}
            </div>

            {hasMore && (
              <div className="flex justify-center pt-4">
                <button
                  onClick={handleLoadMore}
                  disabled={loadingMore}
                  className="px-6 py-3 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white font-medium disabled:opacity-60 disabled:cursor-not-allowed shadow-lg transition-all"
                >
                  {loadingMore ? (
                    <span className="flex items-center gap-2">
                      <span className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></span>
                      Loading...
                    </span>
                  ) : (
                    'Load More'
                  )}
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
