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
      <div className="space-y-10 p-6 md:p-8">
        <section className="max-w-6xl mx-auto glass-panel rounded-2xl p-10 md:p-12 shadow-2xl relative overflow-hidden scan-line">
          <div className="absolute top-0 right-0 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl" />
          <div className="relative z-10">
            <div className="mb-8">
              <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-display font-bold text-gradient-cyan mb-4 animate-float">Welcome Back! 👋</h1>
              <p className="text-cyan-200/80 text-base sm:text-lg md:text-xl max-w-2xl">Ready to analyze your lemon images? Get started with a new analysis or view your previous results.</p>
            </div>

          <div className="grid gap-6 md:grid-cols-2 mb-10">
            <div className="glass-panel rounded-xl p-8 hover-lift relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative z-10">
                <div className="text-5xl mb-4 animate-float" style={{ animationDelay: '0s' }}>🔬</div>
                <h2 className="text-2xl font-display font-semibold text-cyan-100 mb-3">New Analysis</h2>
                <p className="text-cyan-200/70 text-sm mb-6 leading-relaxed">Upload a new image to analyze for diseases and get detailed insights.</p>
                <button 
                  onClick={() => navigate('/analysis')} 
                  className="w-full px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-semibold shadow-lg neon-glow transition-all duration-300"
                >
                  Start Analysis
                </button>
              </div>
            </div>

            <div className="glass-panel rounded-xl p-8 hover-lift relative overflow-hidden group">
              <div className="absolute inset-0 bg-gradient-to-br from-teal-500/10 to-transparent opacity-0 group-hover:opacity-100 transition-opacity" />
              <div className="relative z-10">
                <div className="text-5xl mb-4 animate-float" style={{ animationDelay: '0.2s' }}>📊</div>
                <h2 className="text-2xl font-display font-semibold text-cyan-100 mb-3">View All Analyses</h2>
                <p className="text-cyan-200/70 text-sm mb-6 leading-relaxed">Browse through your analysis history and review past results.</p>
                <button 
                  onClick={() => navigate('/analyses')} 
                  className="w-full px-6 py-3 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-white font-semibold shadow-lg neon-glow transition-all duration-300"
                >
                  My Analyses
                </button>
              </div>
            </div>
          </div>

          {unreadCount > 0 && (
            <div className="glass-panel border-cyan-500/30 rounded-xl p-4 sm:p-6 mb-8 relative overflow-hidden neon-glow-cyan">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-500/20 to-teal-500/20 animate-pulse-glow" />
              <div className="relative z-10 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex-1">
                  <h3 className="text-lg sm:text-xl font-display font-semibold text-cyan-100 mb-2">You have {unreadCount} new notification{unreadCount > 1 ? 's' : ''}</h3>
                  <p className="text-cyan-200/80 text-xs sm:text-sm">Check the notification bell in the header to view your completed analyses.</p>
                </div>
                <div className="text-3xl sm:text-4xl animate-float flex-shrink-0">🔔</div>
              </div>
            </div>
          )}

          <div className="grid md:grid-cols-3 gap-4">
            <div className="glass-panel rounded-xl p-6 text-center hover-lift group">
              <div className="text-4xl mb-3 animate-float group-hover:scale-110 transition-transform" style={{ animationDelay: '0s' }}>⚡</div>
              <h3 className="text-base font-display font-semibold text-cyan-100 mb-2">Quick Analysis</h3>
              <p className="text-xs text-cyan-200/70">Fast and accurate disease detection</p>
            </div>
            
            <div className="glass-panel rounded-xl p-6 text-center hover-lift group">
              <div className="text-4xl mb-3 animate-float group-hover:scale-110 transition-transform" style={{ animationDelay: '0.1s' }}>📝</div>
              <h3 className="text-base font-display font-semibold text-cyan-100 mb-2">Track History</h3>
              <p className="text-xs text-cyan-200/70">Monitor your plant health over time</p>
            </div>
            
            <div className="glass-panel rounded-xl p-6 text-center hover-lift group">
              <div className="text-4xl mb-3 animate-float group-hover:scale-110 transition-transform" style={{ animationDelay: '0.2s' }}>🎯</div>
              <h3 className="text-base font-display font-semibold text-cyan-100 mb-2">Detailed Reports</h3>
              <p className="text-xs text-cyan-200/70">Comprehensive analysis results</p>
            </div>
          </div>
          </div>
        </section>
      </div>
    );
  }

  // Logged out user view (existing content)
  return (
    <div className="space-y-10 p-6 md:p-8">
      <section className="max-w-6xl mx-auto glass-panel rounded-2xl p-10 md:p-12 shadow-2xl relative overflow-hidden scan-line">
        <div className="absolute top-0 left-0 w-96 h-96 bg-teal-500/10 rounded-full blur-3xl" />
        <div className="relative z-10">
          <div className="mb-8">
            <h1 className="text-3xl sm:text-4xl md:text-5xl lg:text-6xl font-display font-bold text-gradient-cyan mb-4 sm:mb-6">Welcome to Lemon Disease Detection</h1>
            <p className="text-cyan-200/80 text-base sm:text-lg md:text-xl max-w-3xl leading-relaxed">A lightweight tool to detect diseases in lemon images using an image analysis model. This project helps farmers and researchers quickly identify plant issues and take action.</p>
          </div>

        <div className="grid gap-8 md:grid-cols-2 mb-10">
          <div className="space-y-4">
            <h2 className="text-2xl font-display font-semibold text-cyan-100 flex items-center gap-3">
              <div className="w-1 h-6 bg-gradient-to-b from-cyan-500 to-teal-500 rounded-full" />
              What this app does
            </h2>
            <ul className="space-y-3 text-cyan-200/80">
              <li className="flex items-start gap-3 leading-relaxed">
                <span className="text-cyan-400 mt-1">→</span>
                <span>Upload an image of a lemon or leaf.</span>
              </li>
              <li className="flex items-start gap-3 leading-relaxed">
                <span className="text-cyan-400 mt-1">→</span>
                <span>Run a server-side analysis to detect disease type and severity.</span>
              </li>
              <li className="flex items-start gap-3 leading-relaxed">
                <span className="text-cyan-400 mt-1">→</span>
                <span>View results, recommendations, and history.</span>
              </li>
            </ul>
          </div>

          <div className="space-y-4">
            <h2 className="text-2xl font-display font-semibold text-cyan-100 flex items-center gap-3">
              <div className="w-1 h-6 bg-gradient-to-b from-teal-500 to-cyan-500 rounded-full" />
              What you'll do
            </h2>
            <ol className="space-y-3 text-cyan-200/80 list-decimal list-inside">
              <li className="leading-relaxed">Create an account or sign in.</li>
              <li className="leading-relaxed">Upload clear photos of the fruit or leaves.</li>
              <li className="leading-relaxed">Review the analysis and follow suggested actions.</li>
            </ol>
          </div>
        </div>

        <div className="mt-10 flex flex-col sm:flex-row flex-wrap gap-3 sm:gap-4">
          <button 
            onClick={() => navigate('/signin')} 
            className="w-full sm:w-auto px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-semibold shadow-lg neon-glow hover-lift transition-all duration-300"
          >
            Sign in
          </button>
          <button 
            onClick={() => navigate('/register')} 
            className="w-full sm:w-auto px-6 py-3 rounded-xl glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 font-semibold hover-lift transition-all duration-300"
          >
            Create account
          </button>
          <button 
            onClick={() => navigate('/analysis')} 
            className="w-full sm:w-auto px-6 py-3 rounded-xl bg-gradient-to-r from-teal-500 to-cyan-500 hover:from-teal-400 hover:to-cyan-400 text-white font-semibold shadow-lg neon-glow hover-lift transition-all duration-300"
          >
            Try demo (analysis)
          </button>
        </div>
        <div className="mt-6 text-center">
          <p className="text-sm text-cyan-300/80">💬 Need help? Click the chatbot button in the bottom right corner!</p>
        </div>
        </div>
      </section>

      {/* Features Carousel Section */}
      <section className="max-w-6xl mx-auto">
        <h2 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-display font-bold text-gradient-cyan mb-6 sm:mb-10 text-center">Key Features</h2>
        
        <div className="glass-panel rounded-2xl p-8 md:p-12 shadow-2xl relative overflow-hidden">
          <div className="absolute inset-0 bg-gradient-to-br from-cyan-500/5 to-teal-500/5" />
          {/* Carousel Container */}
          <div className="relative overflow-hidden rounded-xl h-64 sm:h-72 md:h-80">
            {features.map((feature, index) => (
              <div
                key={index}
                className={`absolute inset-0 px-6 py-8 md:px-12 md:py-12 text-center transition-all duration-700 ${
                  index === currentFeature 
                    ? 'opacity-100 translate-y-0' 
                    : 'opacity-0 translate-y-4 pointer-events-none'
                }`}
              >
                <div className="text-7xl md:text-8xl mb-6 animate-float">{feature.icon}</div>
                <h3 className="text-3xl md:text-4xl font-display font-semibold text-cyan-100 mb-4">{feature.title}</h3>
                <p className="text-cyan-200/80 text-lg md:text-xl max-w-3xl mx-auto leading-relaxed">{feature.description}</p>
              </div>
            ))}
          </div>

          {/* Carousel Indicators */}
          <div className="flex justify-center gap-2 mt-8">
            {features.map((_, index) => (
              <button
                key={index}
                onClick={() => setCurrentFeature(index)}
                className={`h-2 rounded-full transition-all duration-300 ${
                  index === currentFeature
                    ? 'w-10 bg-gradient-to-r from-cyan-400 to-teal-400 neon-glow'
                    : 'w-2 bg-cyan-500/30 hover:bg-cyan-500/50'
                }`}
                aria-label={`Go to feature ${index + 1}`}
              />
            ))}
          </div>

          {/* Navigation Arrows */}
          <div className="flex justify-between items-center mt-6 gap-3">
            <button
              onClick={() => setCurrentFeature((prev) => (prev - 1 + features.length) % features.length)}
              className="flex-1 sm:flex-none px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base rounded-xl glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 font-semibold hover-lift transition-all duration-300"
              aria-label="Previous feature"
            >
              ← Previous
            </button>
            <button
              onClick={() => setCurrentFeature((prev) => (prev + 1) % features.length)}
              className="flex-1 sm:flex-none px-4 sm:px-6 py-2 sm:py-3 text-sm sm:text-base rounded-xl glass-panel border-cyan-500/30 text-cyan-100 hover:bg-cyan-500/10 font-semibold hover-lift transition-all duration-300"
              aria-label="Next feature"
            >
              Next →
            </button>
          </div>
        </div>

        {/* Additional Information Cards */}
        <div className="grid md:grid-cols-3 gap-6 mt-10">
          <div className="glass-panel rounded-xl p-8 text-center hover-lift group">
            <div className="text-5xl mb-4 animate-float group-hover:scale-110 transition-transform">🌿</div>
            <h3 className="text-xl font-display font-semibold text-cyan-100 mb-3">Plant Health</h3>
            <p className="text-sm text-cyan-200/70 leading-relaxed">Monitor and maintain the health of your lemon trees with regular analysis.</p>
          </div>
          
          <div className="glass-panel rounded-xl p-8 text-center hover-lift group">
            <div className="text-5xl mb-4 animate-float group-hover:scale-110 transition-transform" style={{ animationDelay: '0.1s' }}>📱</div>
            <h3 className="text-xl font-display font-semibold text-cyan-100 mb-3">Easy Access</h3>
            <p className="text-sm text-cyan-200/70 leading-relaxed">Access your analysis history and reports from anywhere, anytime.</p>
          </div>
          
          <div className="glass-panel rounded-xl p-8 text-center hover-lift group">
            <div className="text-5xl mb-4 animate-float group-hover:scale-110 transition-transform" style={{ animationDelay: '0.2s' }}>🎯</div>
            <h3 className="text-xl font-display font-semibold text-cyan-100 mb-3">Accurate Results</h3>
            <p className="text-sm text-cyan-200/70 leading-relaxed">Powered by state-of-the-art AI models trained on thousands of images.</p>
          </div>
        </div>
      </section>
    </div>
  );
}
