import { useState, useRef, useEffect } from "react";
import { useAuth } from "~/provider/auth-context";
import Alert from "~/components/alert";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
  sources?: string[]; // For future RAG source citations
}

interface ChatbotModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export function ChatbotModal({ isOpen, onClose }: ChatbotModalProps) {
  const { token } = useAuth();
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "Hello! I'm here to help answer questions about the Lemon Disease Detection project. Ask me anything about the system, technologies, features, or how it works!",
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
      inputRef.current?.focus();
    }
  }, [messages, isOpen]);

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    // TODO: Replace with actual API call when backend is ready
    // Example API call structure:
    // const response = await fetch("http://127.0.0.1:8000/v1/chat/message", {
    //   method: "POST",
    //   headers: {
    //     "Content-Type": "application/json",
    //     ...(token && { Authorization: `Bearer ${token}` }),
    //   },
    //   body: JSON.stringify({
    //     message: userMessage.content,
    //     conversation_id: conversationId, // Optional: for continuing conversations
    //   }),
    // });

    // Simulate API delay
    setTimeout(() => {
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "This is a template response. The chatbot backend is not yet implemented. Once the RAG system is ready, I'll be able to answer questions about the project using the private documentation!",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, assistantMessage]);
      setLoading(false);
    }, 1000);
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const sampleQuestions = [
    "What technologies are used in this project?",
    "How does the analysis workflow work?",
    "What machine learning models are used?",
    "How do I upload an image for analysis?",
  ];

  const handleSampleQuestion = (question: string) => {
    setInput(question);
    inputRef.current?.focus();
  };

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div
        className="fixed inset-0 bg-black/50 z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal */}
      <div className="fixed bottom-20 right-4 md:right-6 w-[calc(100vw-2rem)] md:w-96 h-[calc(100vh-8rem)] md:h-[600px] bg-gradient-to-b from-emerald-900 via-emerald-800 to-black/90 border border-emerald-700 rounded-xl shadow-2xl z-50 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-emerald-700 bg-emerald-900/50 rounded-t-xl">
          <div className="flex items-center gap-3">
            <span className="text-2xl">💬</span>
            <div>
              <h2 className="text-lg font-semibold text-emerald-100">Project Chatbot</h2>
              <p className="text-xs text-emerald-300">Ask questions about the project</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-emerald-700/50 text-emerald-100 transition-colors"
            aria-label="Close chatbot"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Template Mode Alert */}
        <div className="px-4 pt-3">
          <Alert type="info">
            <span className="text-xs">⚠️ Template Mode: Backend not yet implemented</span>
          </Alert>
        </div>

        {/* Messages Container */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-lg p-3 ${
                  message.role === "user"
                    ? "bg-emerald-600 text-white"
                    : "bg-white/10 text-emerald-100 border border-emerald-700"
                }`}
              >
                <div className="flex items-start gap-2">
                  {message.role === "assistant" && (
                    <span className="text-lg flex-shrink-0">🤖</span>
                  )}
                  <div className="flex-1 min-w-0">
                    <p className="whitespace-pre-wrap break-words text-sm">{message.content}</p>
                    {message.sources && message.sources.length > 0 && (
                      <div className="mt-2 pt-2 border-t border-emerald-700/50">
                        <p className="text-xs text-emerald-300 mb-1">Sources:</p>
                        <ul className="text-xs text-emerald-400 space-y-1">
                          {message.sources.map((source, idx) => (
                            <li key={idx}>• {source}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                    <p className="text-xs opacity-70 mt-1">
                      {message.timestamp.toLocaleTimeString([], {
                        hour: "2-digit",
                        minute: "2-digit",
                      })}
                    </p>
                  </div>
                  {message.role === "user" && (
                    <span className="text-lg flex-shrink-0">👤</span>
                  )}
                </div>
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white/10 border border-emerald-700 rounded-lg p-3">
                <div className="flex items-center gap-2">
                  <span className="text-lg">🤖</span>
                  <div className="flex gap-1">
                    <span className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"></span>
                    <span
                      className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.2s" }}
                    ></span>
                    <span
                      className="w-2 h-2 bg-emerald-400 rounded-full animate-bounce"
                      style={{ animationDelay: "0.4s" }}
                    ></span>
                  </div>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Sample Questions */}
        {messages.length === 1 && (
          <div className="px-4 pb-2">
            <p className="text-xs text-emerald-300 mb-2">Try asking:</p>
            <div className="flex flex-wrap gap-1.5">
              {sampleQuestions.map((question, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSampleQuestion(question)}
                  className="px-2 py-1 text-xs rounded-lg bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-100 border border-emerald-700 transition-colors"
                >
                  {question}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="p-4 border-t border-emerald-700 bg-emerald-900/30 rounded-b-xl">
          <div className="flex gap-2">
            <input
              ref={inputRef}
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask a question..."
              disabled={loading}
              className="flex-1 px-3 py-2 text-sm rounded-lg bg-white/10 border border-emerald-700 text-emerald-100 placeholder-emerald-400 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:border-transparent disabled:opacity-50"
            />
            <button
              onClick={handleSend}
              disabled={!input.trim() || loading}
              className="px-4 py-2 rounded-lg bg-emerald-500 hover:bg-emerald-400 text-white font-medium disabled:opacity-50 disabled:cursor-not-allowed transition-colors shadow-lg text-sm"
            >
              Send
            </button>
          </div>
          {error && (
            <div className="mt-2">
              <Alert type="error">{error}</Alert>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
