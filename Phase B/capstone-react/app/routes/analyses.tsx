import { useEffect, useState } from "react";
import { useAuth } from "~/provider/auth-context";
import { useNavigate } from "react-router";
import Alert from "~/components/alert";
import { apiEndpoint } from "~/lib/api-config";

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
      const resp = await fetch(apiEndpoint(`me/analysis/?limit=${limit}&offset=${currentOffset}`), {
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
    <div className="min-h-screen p-6 md:p-8">
      <div className="mx-auto max-w-7xl space-y-6">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 glass-panel rounded-2xl p-4 sm:p-6 shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
          <div className="relative z-10 flex flex-col sm:flex-row items-start sm:items-center justify-between w-full gap-4">
            <h1 className="text-xl sm:text-2xl md:text-3xl font-display font-bold text-gradient-cyan">My Analyses</h1>
            <button
              onClick={() => navigate('/analysis')}
              className="w-full sm:w-auto px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-semibold shadow-lg neon-glow hover-lift transition-all duration-300"
            >
              New Analysis
            </button>
          </div>
        </div>

        {error && (
          <Alert type="error">{error}</Alert>
        )}

        {loading && analyses.length === 0 ? (
          <div className="glass-panel rounded-2xl p-8 shadow-2xl text-center relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-transparent animate-pulse" />
            <div className="relative z-10 text-cyan-200">Loading analyses...</div>
          </div>
        ) : analyses.length === 0 ? (
          <div className="glass-panel rounded-2xl p-8 shadow-2xl text-center relative overflow-hidden">
            <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
            <div className="relative z-10">
              <h2 className="text-xl font-display font-semibold text-cyan-100 mb-2">No analyses found</h2>
              <p className="text-cyan-200/80 mb-4">You haven't created any analyses yet.</p>
              <button
                onClick={() => navigate('/analysis')}
                className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-semibold shadow-lg neon-glow hover-lift transition-all duration-300"
              >
                Create Your First Analysis
              </button>
            </div>
          </div>
        ) : (
          <>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {analyses.map((analysis) => (
                <div
                  key={analysis.id}
                  onClick={() => handleAnalysisClick(analysis)}
                  className="glass-panel rounded-xl p-5 shadow-2xl hover-lift cursor-pointer transition-all duration-200 flex flex-col border border-cyan-500/20 hover:border-cyan-500/40 relative overflow-hidden group"
                >
                  <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
                  <div className="relative z-10">
                    <div className="flex items-start justify-between mb-3">
                      <h3 className="text-sm font-display font-semibold text-cyan-300 uppercase tracking-wide">
                        Analysis {analysis.id.slice(0, 8)}
                      </h3>
                      <span className="text-xs text-cyan-400">
                        {formatDate(analysis.created_at)}
                      </span>
                    </div>
                    
                    {analysis.summary && (
                      <p className="text-cyan-200/80 text-sm mb-3 line-clamp-3 flex-grow">
                        {analysis.summary}
                      </p>
                    )}
                    
                    {analysis.description && (
                      <p className="text-cyan-300/70 text-xs line-clamp-2">
                        {analysis.description}
                      </p>
                    )}
                    
                    {!analysis.summary && !analysis.description && (
                      <p className="text-cyan-400/60 text-sm italic">No details available</p>
                    )}
                    
                    <div className="mt-4 pt-3 border-t border-cyan-500/20">
                      <span className="text-xs text-cyan-400">Click to view details →</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {hasMore && (
              <div className="flex justify-center pt-4">
                <button
                  onClick={handleLoadMore}
                  disabled={loadingMore}
                  className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-semibold disabled:opacity-60 disabled:cursor-not-allowed shadow-lg neon-glow hover-lift transition-all duration-300"
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
