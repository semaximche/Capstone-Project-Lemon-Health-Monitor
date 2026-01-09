import { useNavigate } from "react-router";
import { useState, useEffect } from "react";
import { useAuth } from "~/provider/auth-context";

const features = [
  {
    title: "AI-Powered Detection",
    description: "Advanced machine learning models analyze lemon images to identify diseases with high accuracy.",
    icon: "🔬"
  },
  {
    title: "Real-Time Analysis",
    description: "Get instant results after uploading your images. Our system processes images quickly and efficiently.",
    icon: "⚡"
  },
  {
    title: "Detailed Reports",
    description: "Receive comprehensive analysis reports with disease identification, severity assessment, and recommendations.",
    icon: "📊"
  },
  {
    title: "History Tracking",
    description: "Keep track of all your analyses in one place. Review past results and monitor plant health over time.",
    icon: "📝"
  },
  {
    title: "Easy to Use",
    description: "Simple interface designed for farmers and researchers. No technical expertise required.",
    icon: "👨‍🌾"
  },
  {
    title: "Secure & Private",
    description: "Your data is securely stored and protected. All analyses are private to your account.",
    icon: "🔒"
  }
];

export default function Welcome() {
  const navigate = useNavigate();
  const { token, unreadCount } = useAuth();
  const [currentFeature, setCurrentFeature] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setCurrentFeature((prev) => (prev + 1) % features.length);
    }, 4000);
    return () => clearInterval(interval);
  }, []);

  // Logged in user view
  if (token) {
    return (
      <div className="space-y-8">
        <section className="max-w-4xl mx-auto p-8 bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 rounded-xl border border-emerald-700 text-white shadow-lg">
          <div className="mb-6">
            <h1 className="text-4xl font-bold mb-2 text-emerald-100">Welcome Back! 👋</h1>
            <p className="text-emerald-200">Ready to analyze your lemon images? Get started with a new analysis or view your previous results.</p>
          </div>

          <div className="grid gap-6 md:grid-cols-2 mb-8">
            <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 hover:bg-white/10 transition-colors">
              <div className="text-4xl mb-3">🔬</div>
              <h2 className="text-xl font-semibold text-emerald-100 mb-2">New Analysis</h2>
              <p className="text-emerald-200 text-sm mb-4">Upload a new image to analyze for diseases and get detailed insights.</p>
              <button 
                onClick={() => navigate('/analysis')} 
                className="w-full px-4 py-2 rounded bg-emerald-500 hover:bg-emerald-400 text-white shadow transition-colors"
              >
                Start Analysis
              </button>
            </div>

            <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 hover:bg-white/10 transition-colors">
              <div className="text-4xl mb-3">📊</div>
              <h2 className="text-xl font-semibold text-emerald-100 mb-2">View All Analyses</h2>
              <p className="text-emerald-200 text-sm mb-4">Browse through your analysis history and review past results.</p>
              <button 
                onClick={() => navigate('/analyses')} 
                className="w-full px-4 py-2 rounded bg-emerald-500 hover:bg-emerald-400 text-white shadow transition-colors"
              >
                My Analyses
              </button>
            </div>
          </div>

          {unreadCount > 0 && (
            <div className="bg-emerald-500/20 border border-emerald-500 rounded-xl p-4 mb-6">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-lg font-semibold text-emerald-100 mb-1">You have {unreadCount} new notification{unreadCount > 1 ? 's' : ''}</h3>
                  <p className="text-emerald-200 text-sm">Check the notification bell in the header to view your completed analyses.</p>
                </div>
                <div className="text-3xl">🔔</div>
              </div>
            </div>
          )}

          <div className="grid md:grid-cols-3 gap-4">
            <div className="bg-white/5 border border-emerald-700 rounded-xl p-5 text-center">
              <div className="text-3xl mb-2">⚡</div>
              <h3 className="text-sm font-semibold text-emerald-100 mb-1">Quick Analysis</h3>
              <p className="text-xs text-emerald-200">Fast and accurate disease detection</p>
            </div>
            
            <div className="bg-white/5 border border-emerald-700 rounded-xl p-5 text-center">
              <div className="text-3xl mb-2">📝</div>
              <h3 className="text-sm font-semibold text-emerald-100 mb-1">Track History</h3>
              <p className="text-xs text-emerald-200">Monitor your plant health over time</p>
            </div>
            
            <div className="bg-white/5 border border-emerald-700 rounded-xl p-5 text-center">
              <div className="text-3xl mb-2">🎯</div>
              <h3 className="text-sm font-semibold text-emerald-100 mb-1">Detailed Reports</h3>
              <p className="text-xs text-emerald-200">Comprehensive analysis results</p>
            </div>
          </div>
        </section>
      </div>
    );
  }

  // Logged out user view (existing content)
  return (
    <div className="space-y-8">
      <section className="max-w-4xl mx-auto p-8 bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 rounded-xl border border-emerald-700 text-white shadow-lg">
        <div className="mb-6">
          <h1 className="text-4xl font-bold mb-2 text-emerald-100">Welcome to Lemon Disease Detection</h1>
          <p className="text-emerald-200">A lightweight tool to detect diseases in lemon images using an image analysis model. This project helps farmers and researchers quickly identify plant issues and take action.</p>
        </div>

        <div className="grid gap-6 md:grid-cols-2">
          <div className="space-y-3">
            <h2 className="text-xl font-semibold text-emerald-100">What this app does</h2>
            <ul className="list-disc list-inside text-slate-200">
              <li className="leading-relaxed">Upload an image of a lemon or leaf.</li>
              <li className="leading-relaxed">Run a server-side analysis to detect disease type and severity.</li>
              <li className="leading-relaxed">View results, recommendations, and history.</li>
            </ul>
          </div>

          <div className="space-y-3">
            <h2 className="text-xl font-semibold text-emerald-100">What you'll do</h2>
            <ol className="list-decimal list-inside text-slate-200">
              <li className="leading-relaxed">Create an account or sign in.</li>
              <li className="leading-relaxed">Upload clear photos of the fruit or leaves.</li>
              <li className="leading-relaxed">Review the analysis and follow suggested actions.</li>
            </ol>
          </div>
        </div>

        <div className="mt-8 flex flex-wrap gap-3">
          <button onClick={() => navigate('/signin')} className="px-4 py-2 rounded bg-emerald-500 hover:bg-emerald-400 text-white shadow transition-colors">Sign in</button>
          <button onClick={() => navigate('/register')} className="px-4 py-2 rounded bg-white/10 hover:bg-white/20 text-emerald-100 transition-colors">Create account</button>
          <button onClick={() => navigate('/analysis')} className="px-4 py-2 rounded bg-sky-500 hover:bg-sky-400 text-white shadow transition-colors">Try demo (analysis)</button>
        </div>
      </section>

      {/* Features Carousel Section */}
      <section className="max-w-4xl mx-auto">
        <h2 className="text-3xl font-bold text-emerald-100 mb-6 text-center">Key Features</h2>
        
        <div className="bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 rounded-xl border border-emerald-700 p-8 shadow-lg">
          {/* Carousel Container */}
          <div className="relative overflow-hidden rounded-lg h-64">
            {features.map((feature, index) => (
              <div
                key={index}
                className={`absolute inset-0 px-6 py-8 text-center transition-opacity duration-500 ${
                  index === currentFeature ? 'opacity-100' : 'opacity-0'
                }`}
              >
                <div className="text-6xl mb-4">{feature.icon}</div>
                <h3 className="text-2xl font-semibold text-emerald-100 mb-3">{feature.title}</h3>
                <p className="text-emerald-200 text-lg max-w-2xl mx-auto">{feature.description}</p>
              </div>
            ))}
          </div>

          {/* Carousel Indicators */}
          <div className="flex justify-center gap-2 mt-6">
            {features.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentFeature(index)}
                className={`h-2 rounded-full transition-all ${
                  index === currentFeature
                    ? 'w-8 bg-emerald-400'
                    : 'w-2 bg-emerald-700 hover:bg-emerald-600'
                }`}
                aria-label={`Go to feature ${index + 1}`}
              />
            ))}
          </div>

          {/* Navigation Arrows */}
          <div className="flex justify-between items-center mt-4">
            <button
              onClick={() => setCurrentFeature((prev) => (prev - 1 + features.length) % features.length)}
              className="px-4 py-2 rounded bg-emerald-500/50 hover:bg-emerald-500 text-white transition-colors"
              aria-label="Previous feature"
            >
              ← Previous
            </button>
            <button
              onClick={() => setCurrentFeature((prev) => (prev + 1) % features.length)}
              className="px-4 py-2 rounded bg-emerald-500/50 hover:bg-emerald-500 text-white transition-colors"
              aria-label="Next feature"
            >
              Next →
            </button>
          </div>
        </div>

        {/* Additional Information Cards */}
        <div className="grid md:grid-cols-3 gap-4 mt-8">
          <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 text-center">
            <div className="text-4xl mb-3">🌿</div>
            <h3 className="text-lg font-semibold text-emerald-100 mb-2">Plant Health</h3>
            <p className="text-sm text-emerald-200">Monitor and maintain the health of your lemon trees with regular analysis.</p>
          </div>
          
          <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 text-center">
            <div className="text-4xl mb-3">📱</div>
            <h3 className="text-lg font-semibold text-emerald-100 mb-2">Easy Access</h3>
            <p className="text-sm text-emerald-200">Access your analysis history and reports from anywhere, anytime.</p>
          </div>
          
          <div className="bg-white/5 border border-emerald-700 rounded-xl p-6 text-center">
            <div className="text-4xl mb-3">🎯</div>
            <h3 className="text-lg font-semibold text-emerald-100 mb-2">Accurate Results</h3>
            <p className="text-sm text-emerald-200">Powered by state-of-the-art AI models trained on thousands of images.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
