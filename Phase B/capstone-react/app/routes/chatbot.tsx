export default function Chatbot() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/80 p-4 md:p-6 flex items-center justify-center">
      <div className="max-w-2xl text-center">
        <div className="bg-white/5 border border-emerald-700 rounded-xl p-8 shadow-lg">
          <div className="text-6xl mb-4">💬</div>
          <h1 className="text-2xl md:text-3xl font-bold text-emerald-100 mb-4">Chatbot Available</h1>
          <p className="text-emerald-200 mb-6">
            The chatbot is now accessible via the floating button in the bottom right corner of the screen.
            Click the chat icon to open the chatbot popup from anywhere in the application!
          </p>
          <div className="flex justify-center">
            <div className="bg-emerald-500/20 border border-emerald-500 rounded-lg p-4 inline-block">
              <p className="text-emerald-100 text-sm">💡 Look for the 💬 button in the bottom right corner</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
