import { useState } from "react";
import { ChatbotModal } from "./chatbot-modal";

export function ChatbotButton() {
  const [isOpen, setIsOpen] = useState(false);

  return (
    <>
      {/* Floating Button */}
      <button
        onClick={() => setIsOpen(true)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-emerald-500 hover:bg-emerald-400 text-white shadow-2xl flex items-center justify-center transition-all duration-200 hover:scale-110 z-30"
        aria-label="Open chatbot"
      >
        <span className="text-2xl">💬</span>
      </button>

      {/* Chatbot Modal */}
      <ChatbotModal isOpen={isOpen} onClose={() => setIsOpen(false)} />
    </>
  );
}
