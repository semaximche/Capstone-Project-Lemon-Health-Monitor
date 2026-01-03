import { useNavigate } from "react-router";

export default function Welcome() {
  const navigate = useNavigate();

  return (
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
        <button onClick={() => navigate('/signin')} className="px-4 py-2 rounded bg-emerald-500 hover:bg-emerald-400 text-white shadow">Sign in</button>
        <button onClick={() => navigate('/register')} className="px-4 py-2 rounded bg-white/10 hover:bg-white/20 text-emerald-100">Create account</button>
        <button onClick={() => navigate('/analysis')} className="px-4 py-2 rounded bg-sky-500 hover:bg-sky-400 text-white shadow">Try demo (analysis)</button>
      </div>
    </section>
  );
}
