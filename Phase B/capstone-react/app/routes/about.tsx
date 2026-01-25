import { useNavigate } from "react-router";

export default function About() {
  const navigate = useNavigate();

  const technologies = [
    {
      category: "Backend Framework",
      items: ["Python", "FastAPI", "Microservices Architecture"]
    },
    {
      category: "Machine Learning",
      items: ["YOLOv8 (Object Detection)", "EfficientNet (Disease Classification)", "Gemini GenAI Flash (LLM)"]
    },
    {
      category: "Message Queue & Real-time",
      items: ["RabbitMQ", "WebSockets"]
    },
    {
      category: "Storage",
      items: ["SQLite (Database)", "S3 Bucket (Object Storage)"]
    },
    {
      category: "Frontend",
      items: ["React", "TypeScript", "React Router", "Tailwind CSS"]
    },
    {
      category: "Development Tools",
      items: ["GitHub Copilot", "Cursor AI"]
    }
  ];

  return (
    <div className="min-h-screen p-6 md:p-8">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="text-center mb-8">
          <h1 className="text-2xl sm:text-3xl md:text-4xl lg:text-5xl font-display font-bold text-gradient-cyan mb-4 px-4">About the Project</h1>
          <p className="text-base sm:text-lg text-cyan-200/80 max-w-3xl mx-auto px-4">
            Lemon Disease Detection is a comprehensive AI-powered system for identifying and analyzing diseases in lemon plants using advanced machine learning models and modern microservices architecture.
          </p>
        </div>

        {/* Architecture Diagram */}
        <section className="glass-panel rounded-2xl p-8 shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
          <h2 className="text-xl sm:text-2xl font-display font-bold text-cyan-100 mb-4 sm:mb-6 text-center relative z-10 px-4">System Architecture & Flow</h2>

          <div className="space-y-6">
            {/* Flow Diagram */}
            <div className="relative">
              {/* User Layer */}
              <div className="bg-cyan-600/30 border-2 border-cyan-500 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">👤</div>
                  <h3 className="text-lg font-semibold text-cyan-100">User (React Frontend)</h3>
                  <p className="text-sm text-cyan-200">Uploads lemon images</p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-cyan-400">↓</div>
              </div>

              {/* API Gateway Layer */}
              <div className="bg-blue-600/30 border-2 border-blue-500 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">🚪</div>
                  <h3 className="text-lg font-semibold text-cyan-100">FastAPI Gateway</h3>
                  <p className="text-sm text-cyan-200">Microservices Architecture</p>
                  <p className="text-xs text-cyan-300 mt-1">Receives image, validates, stores in S3</p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-cyan-400">↓</div>
              </div>

              {/* Message Queue */}
              <div className="bg-indigo-600/30 border-2 border-indigo-500 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">📬</div>
                  <h3 className="text-lg font-semibold text-cyan-100">RabbitMQ</h3>
                  <p className="text-sm text-cyan-200">Message Queue</p>
                  <p className="text-xs text-cyan-300 mt-1">Queues analysis tasks</p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-cyan-400">↓</div>
              </div>

              {/* ML Processing Layer */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 sm:gap-4 mb-4">
                <div className="bg-orange-600/30 border-2 border-orange-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">🎯</div>
                    <h4 className="text-sm font-semibold text-cyan-100">YOLOv8</h4>
                    <p className="text-xs text-cyan-200">Leaf Detection</p>
                  </div>
                </div>
                <div className="bg-teal-600/30 border-2 border-teal-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">🔬</div>
                    <h4 className="text-sm font-semibold text-cyan-100">EfficientNet</h4>
                    <p className="text-xs text-cyan-200">Disease Classification</p>
                  </div>
                </div>
                <div className="bg-slate-600/30 border-2 border-slate-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">🤖</div>
                    <h4 className="text-sm font-semibold text-cyan-100">Gemini Flash</h4>
                    <p className="text-xs text-cyan-200">LLM Analysis</p>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-cyan-400">↓</div>
              </div>

              {/* Storage Layer */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 sm:gap-4 mb-4">
                <div className="bg-slate-600/30 border-2 border-slate-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">💾</div>
                    <h4 className="text-sm font-semibold text-cyan-100">SQLite</h4>
                    <p className="text-xs text-cyan-200">Analysis Metadata</p>
                  </div>
                </div>
                <div className="bg-cyan-600/30 border-2 border-cyan-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">☁️</div>
                    <h4 className="text-sm font-semibold text-cyan-100">S3 Bucket</h4>
                    <p className="text-xs text-cyan-200">Image Storage</p>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-cyan-400">↓</div>
              </div>

              {/* Notification Layer */}
              <div className="bg-indigo-600/30 border-2 border-indigo-500 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">🔔</div>
                  <h3 className="text-lg font-semibold text-cyan-100">WebSocket Notification</h3>
                  <p className="text-sm text-cyan-200">Real-time updates to user</p>
                  <p className="text-xs text-cyan-300 mt-1">Analysis complete notification</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Technologies Section */}
        <section className="glass-panel rounded-2xl p-8 shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 left-0 w-64 h-64 bg-teal-500/5 rounded-full blur-3xl" />
          <h2 className="text-xl sm:text-2xl font-display font-bold text-cyan-100 mb-4 sm:mb-6 text-center relative z-10 px-4">Technologies Used</h2>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6 relative z-10">
            {technologies.map((tech, index) => (
              <div key={index} className="glass-panel rounded-xl p-5 hover-lift border border-cyan-500/20">
                <h3 className="text-lg font-display font-semibold text-cyan-100 mb-3">{tech.category}</h3>
                <ul className="space-y-2">
                  {tech.items.map((item, itemIndex) => (
                    <li key={itemIndex} className="text-cyan-200/80 text-sm flex items-start gap-2">
                      <span className="text-cyan-400 mt-1">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Project Flow Section */}
        <section className="glass-panel rounded-2xl p-8 shadow-2xl relative overflow-hidden">
          <div className="absolute bottom-0 right-0 w-64 h-64 bg-indigo-500/5 rounded-full blur-3xl" />
          <h2 className="text-xl sm:text-2xl font-display font-bold text-cyan-100 mb-4 sm:mb-6 text-center relative z-10 px-4">Project Flow</h2>

          <div className="space-y-4 relative z-10">
            <div className="flex gap-3 sm:gap-4 items-start">
              <div className="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-cyan-500 to-teal-500 flex items-center justify-center text-white text-sm sm:text-base font-bold shadow-lg">1</div>
              <div className="flex-1 min-w-0">
                <h3 className="text-base sm:text-lg font-display font-semibold text-cyan-100 mb-1">Image Upload</h3>
                <p className="text-sm sm:text-base text-cyan-200/80">User uploads a lemon/leaf image through the React frontend interface.</p>
              </div>
            </div>

            <div className="flex gap-3 sm:gap-4 items-start">
              <div className="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-cyan-500 to-teal-500 flex items-center justify-center text-white text-sm sm:text-base font-bold shadow-lg">2</div>
              <div className="flex-1 min-w-0">
                <h3 className="text-base sm:text-lg font-display font-semibold text-cyan-100 mb-1">API Processing</h3>
                <p className="text-sm sm:text-base text-cyan-200/80">FastAPI gateway receives the image, validates it, and stores it in S3 bucket. Analysis metadata is saved in SQLite.</p>
              </div>
            </div>

            <div className="flex gap-3 sm:gap-4 items-start">
              <div className="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-cyan-500 to-teal-500 flex items-center justify-center text-white text-sm sm:text-base font-bold shadow-lg">3</div>
              <div className="flex-1 min-w-0">
                <h3 className="text-base sm:text-lg font-display font-semibold text-cyan-100 mb-1">Task Queuing</h3>
                <p className="text-sm sm:text-base text-cyan-200/80">Analysis task is queued in RabbitMQ for asynchronous processing.</p>
              </div>
            </div>

            <div className="flex gap-3 sm:gap-4 items-start">
              <div className="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-cyan-500 to-teal-500 flex items-center justify-center text-white text-sm sm:text-base font-bold shadow-lg">4</div>
              <div className="flex-1 min-w-0">
                <h3 className="text-base sm:text-lg font-display font-semibold text-cyan-100 mb-1">ML Model Processing</h3>
                <p className="text-sm sm:text-base text-cyan-200/80">ML worker processes the image: YOLOv8 detects leaves, EfficientNet classifies diseases, and Gemini Flash generates detailed analysis and recommendations.</p>
              </div>
            </div>

            <div className="flex gap-3 sm:gap-4 items-start">
              <div className="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-cyan-500 to-teal-500 flex items-center justify-center text-white text-sm sm:text-base font-bold shadow-lg">5</div>
              <div className="flex-1 min-w-0">
                <h3 className="text-base sm:text-lg font-display font-semibold text-cyan-100 mb-1">Results Storage</h3>
                <p className="text-sm sm:text-base text-cyan-200/80">Analysis results (description, summary) are stored in SQLite database and linked to the original image in S3.</p>
              </div>
            </div>

            <div className="flex gap-3 sm:gap-4 items-start">
              <div className="flex-shrink-0 w-8 h-8 sm:w-10 sm:h-10 rounded-full bg-gradient-to-r from-cyan-500 to-teal-500 flex items-center justify-center text-white text-sm sm:text-base font-bold shadow-lg">6</div>
              <div className="flex-1 min-w-0">
                <h3 className="text-base sm:text-lg font-display font-semibold text-cyan-100 mb-1">Real-time Notification</h3>
                <p className="text-sm sm:text-base text-cyan-200/80">WebSocket sends real-time notification to the user when analysis is complete. User can click notification to view detailed results on dashboard.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Back Button */}
        <div className="text-center">
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 rounded-xl bg-gradient-to-r from-cyan-500 to-teal-500 hover:from-cyan-400 hover:to-teal-400 text-white font-semibold shadow-lg neon-glow hover-lift transition-all duration-300"
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
