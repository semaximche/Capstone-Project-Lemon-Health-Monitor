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
    <div className="min-h-screen bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 p-6">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="text-center mb-8">
          <h1 className="text-4xl md:text-5xl font-bold text-emerald-100 mb-4">About the Project</h1>
          <p className="text-lg text-emerald-200 max-w-3xl mx-auto">
            Lemon Disease Detection is a comprehensive AI-powered system for identifying and analyzing diseases in lemon plants using advanced machine learning models and modern microservices architecture.
          </p>
        </div>

        {/* Architecture Diagram */}
        <section className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-emerald-100 mb-6 text-center">System Architecture & Flow</h2>
          
          <div className="space-y-6">
            {/* Flow Diagram */}
            <div className="relative">
              {/* User Layer */}
              <div className="bg-emerald-600/30 border-2 border-emerald-500 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">👤</div>
                  <h3 className="text-lg font-semibold text-emerald-100">User (React Frontend)</h3>
                  <p className="text-sm text-emerald-200">Uploads lemon images</p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-emerald-400">↓</div>
              </div>

              {/* API Gateway Layer */}
              <div className="bg-blue-600/30 border-2 border-blue-500 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">🚪</div>
                  <h3 className="text-lg font-semibold text-emerald-100">FastAPI Gateway</h3>
                  <p className="text-sm text-emerald-200">Microservices Architecture</p>
                  <p className="text-xs text-emerald-300 mt-1">Receives image, validates, stores in S3</p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-emerald-400">↓</div>
              </div>

              {/* Message Queue */}
              <div className="bg-purple-600/30 border-2 border-purple-500 rounded-lg p-4 mb-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">📬</div>
                  <h3 className="text-lg font-semibold text-emerald-100">RabbitMQ</h3>
                  <p className="text-sm text-emerald-200">Message Queue</p>
                  <p className="text-xs text-emerald-300 mt-1">Queues analysis tasks</p>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-emerald-400">↓</div>
              </div>

              {/* ML Processing Layer */}
              <div className="grid md:grid-cols-3 gap-4 mb-4">
                <div className="bg-orange-600/30 border-2 border-orange-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">🎯</div>
                    <h4 className="text-sm font-semibold text-emerald-100">YOLOv8</h4>
                    <p className="text-xs text-emerald-200">Leaf Detection</p>
                  </div>
                </div>
                <div className="bg-red-600/30 border-2 border-red-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">🔬</div>
                    <h4 className="text-sm font-semibold text-emerald-100">EfficientNet</h4>
                    <p className="text-xs text-emerald-200">Disease Classification</p>
                  </div>
                </div>
                <div className="bg-yellow-600/30 border-2 border-yellow-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">🤖</div>
                    <h4 className="text-sm font-semibold text-emerald-100">Gemini Flash</h4>
                    <p className="text-xs text-emerald-200">LLM Analysis</p>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-emerald-400">↓</div>
              </div>

              {/* Storage Layer */}
              <div className="grid md:grid-cols-2 gap-4 mb-4">
                <div className="bg-green-600/30 border-2 border-green-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">💾</div>
                    <h4 className="text-sm font-semibold text-emerald-100">SQLite</h4>
                    <p className="text-xs text-emerald-200">Analysis Metadata</p>
                  </div>
                </div>
                <div className="bg-cyan-600/30 border-2 border-cyan-500 rounded-lg p-4">
                  <div className="text-center">
                    <div className="text-2xl mb-2">☁️</div>
                    <h4 className="text-sm font-semibold text-emerald-100">S3 Bucket</h4>
                    <p className="text-xs text-emerald-200">Image Storage</p>
                  </div>
                </div>
              </div>

              {/* Arrow */}
              <div className="flex justify-center mb-4">
                <div className="text-3xl text-emerald-400">↓</div>
              </div>

              {/* Notification Layer */}
              <div className="bg-pink-600/30 border-2 border-pink-500 rounded-lg p-4">
                <div className="text-center">
                  <div className="text-3xl mb-2">🔔</div>
                  <h3 className="text-lg font-semibold text-emerald-100">WebSocket Notification</h3>
                  <p className="text-sm text-emerald-200">Real-time updates to user</p>
                  <p className="text-xs text-emerald-300 mt-1">Analysis complete notification</p>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Technologies Section */}
        <section className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-emerald-100 mb-6 text-center">Technologies Used</h2>
          
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
            {technologies.map((tech, index) => (
              <div key={index} className="bg-white/5 border border-emerald-700 rounded-lg p-5">
                <h3 className="text-lg font-semibold text-emerald-100 mb-3">{tech.category}</h3>
                <ul className="space-y-2">
                  {tech.items.map((item, itemIndex) => (
                    <li key={itemIndex} className="text-emerald-200 text-sm flex items-start gap-2">
                      <span className="text-emerald-400 mt-1">•</span>
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </section>

        {/* Project Flow Section */}
        <section className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg">
          <h2 className="text-2xl font-bold text-emerald-100 mb-6 text-center">Project Flow</h2>
          
          <div className="space-y-4">
            <div className="flex gap-4 items-start">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white font-bold">1</div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-emerald-100 mb-1">Image Upload</h3>
                <p className="text-emerald-200">User uploads a lemon/leaf image through the React frontend interface.</p>
              </div>
            </div>

            <div className="flex gap-4 items-start">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white font-bold">2</div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-emerald-100 mb-1">API Processing</h3>
                <p className="text-emerald-200">FastAPI gateway receives the image, validates it, and stores it in S3 bucket. Analysis metadata is saved in SQLite.</p>
              </div>
            </div>

            <div className="flex gap-4 items-start">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white font-bold">3</div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-emerald-100 mb-1">Task Queuing</h3>
                <p className="text-emerald-200">Analysis task is queued in RabbitMQ for asynchronous processing.</p>
              </div>
            </div>

            <div className="flex gap-4 items-start">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white font-bold">4</div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-emerald-100 mb-1">ML Model Processing</h3>
                <p className="text-emerald-200">ML worker processes the image: YOLOv8 detects leaves, EfficientNet classifies diseases, and Gemini Flash generates detailed analysis and recommendations.</p>
              </div>
            </div>

            <div className="flex gap-4 items-start">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white font-bold">5</div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-emerald-100 mb-1">Results Storage</h3>
                <p className="text-emerald-200">Analysis results (description, summary) are stored in SQLite database and linked to the original image in S3.</p>
              </div>
            </div>

            <div className="flex gap-4 items-start">
              <div className="flex-shrink-0 w-10 h-10 rounded-full bg-emerald-500 flex items-center justify-center text-white font-bold">6</div>
              <div className="flex-1">
                <h3 className="text-lg font-semibold text-emerald-100 mb-1">Real-time Notification</h3>
                <p className="text-emerald-200">WebSocket sends real-time notification to the user when analysis is complete. User can click notification to view detailed results on dashboard.</p>
              </div>
            </div>
          </div>
        </section>

        {/* Back Button */}
        <div className="text-center">
          <button
            onClick={() => navigate('/')}
            className="px-6 py-3 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white font-medium transition-colors shadow-lg"
          >
            ← Back to Home
          </button>
        </div>
      </div>
    </div>
  );
}
