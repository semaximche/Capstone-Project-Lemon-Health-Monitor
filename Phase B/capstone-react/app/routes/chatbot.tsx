export default function Chatbot() {
  return (
    <div className="min-h-screen p-4 md:p-6 flex items-center justify-center">
      <div className="max-w-2xl text-center">
        <div className="glass-panel rounded-2xl p-8 shadow-2xl relative overflow-hidden">
          <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/5 rounded-full blur-3xl" />
          <div className="relative z-10">
            <div className="text-6xl mb-4">💬</div>
            <h1 className="text-2xl md:text-3xl font-display font-bold text-gradient-cyan mb-4">Chatbot Available</h1>
            <p className="text-cyan-200/80 mb-6">
              The chatbot is now accessible via the floating button in the bottom right corner of the screen.
              Click the chat icon to open the chatbot popup from anywhere in the application!
            </p>
            <div className="flex justify-center">
              <div className="glass-panel border border-cyan-500/30 rounded-lg p-4 inline-block">
                <p className="text-cyan-100 text-sm">💡 Look for the 💬 button in the bottom right corner</p>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
