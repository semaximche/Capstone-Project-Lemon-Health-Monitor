# Chatbot Frontend Implementation - Backend Integration Guide

## Overview

This document describes the frontend chatbot implementation and provides specifications for backend integration. The chatbot UI is fully implemented as a **floating popup modal** accessible from anywhere in the application via a button in the bottom right corner.

## Frontend Implementation Summary

### Files Created/Modified

1. **`app/components/chatbot-modal.tsx`** - Chatbot popup modal component
2. **`app/components/chatbot-button.tsx`** - Floating button component (bottom right)
3. **`app/root.tsx`** - Added ChatbotButton to root layout (available everywhere)
4. **`app/routes/chatbot.tsx`** - Redirect page (shows message about floating button)
5. **`app/routes.ts`** - Chatbot route (kept for direct navigation fallback)
6. **`app/components/navigation-list.tsx`** - Removed chatbot tab (now accessible via button)
7. **`app/components/welcome.tsx`** - Removed chatbot button (now accessible via floating button)

### UI Features Implemented

- ✅ **Floating button** in bottom right corner (accessible from any page)
- ✅ **Popup modal** that opens when button is clicked
- ✅ Chat interface with message history
- ✅ User and assistant message bubbles with distinct styling
- ✅ Input field with send button
- ✅ Loading indicator (typing animation)
- ✅ Sample questions for quick start
- ✅ Welcome message
- ✅ Template mode indicator
- ✅ Responsive design (mobile-friendly)
- ✅ Auto-scroll to latest message
- ✅ Enter key to send messages
- ✅ Source citations display area (ready for RAG sources)
- ✅ Timestamp display for each message
- ✅ Error handling UI
- ✅ Backdrop overlay (click to close)
- ✅ Close button in modal header

## API Integration Points

### Expected Backend Endpoint

**Endpoint**: `POST http://127.0.0.1:8000/v1/chat/message`

**Request Headers**:
```typescript
{
  "Content-Type": "application/json",
  "Authorization": "Bearer {token}" // Optional - for authenticated users
}
```

**Request Body**:
```typescript
{
  message: string;              // User's question
  conversation_id?: string;     // Optional: UUID for continuing conversations
  user_id?: string;             // Optional: if user is authenticated
}
```

**Response Body**:
```typescript
{
  response: string;             // Assistant's answer
  conversation_id: string;      // UUID of the conversation
  sources?: string[];           // Optional: Array of source document names/IDs
  timestamp: string;            // ISO 8601 timestamp
}
```

**Error Response**:
```typescript
{
  detail: string;              // Error message
  status_code: number;         // HTTP status code
}
```

### Frontend Code Location for API Integration

In `app/components/chatbot-modal.tsx`, replace the TODO comment section (around line 50-65) with:

```typescript
const response = await fetch("http://127.0.0.1:8000/v1/chat/message", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    ...(token && { Authorization: `Bearer ${token}` }),
  },
  body: JSON.stringify({
    message: userMessage.content,
    conversation_id: conversationId, // Track this in state if needed
  }),
});

if (!response.ok) {
  throw new Error(`Chat request failed: ${response.status}`);
}

const data = await response.json();

const assistantMessage: Message = {
  id: Date.now().toString(),
  role: "assistant",
  content: data.response,
  timestamp: new Date(data.timestamp),
  sources: data.sources, // Will display if provided
};
```

## Data Structures

### Message Interface

```typescript
interface Message {
  id: string;                    // Unique message ID
  role: "user" | "assistant";   // Message sender
  content: string;               // Message text
  timestamp: Date;                // When message was sent/received
  sources?: string[];             // Optional: RAG source citations
}
```

## Conversation Management

### Frontend State

The frontend currently maintains messages in local state. For conversation persistence:

1. **Store conversation_id** in component state after first API response
2. **Send conversation_id** with subsequent messages to maintain context
3. **Optionally** fetch conversation history on page load if conversation_id exists

### Recommended Backend Support

- **Create conversation**: First message without `conversation_id` creates new conversation
- **Continue conversation**: Subsequent messages with `conversation_id` maintain context
- **Get conversation history**: `GET /v1/chat/conversations/{id}/messages` (optional)

## UI States Handled

### Loading State
- Shows typing indicator animation when waiting for response
- Disables input and send button during loading

### Error State
- Displays error alert if API call fails
- User can retry by sending another message

### Empty State
- Shows welcome message on initial load
- Displays sample questions to help users get started

## Authentication

- **Optional Authentication**: Chatbot works for both authenticated and anonymous users
- **Token Usage**: If user is logged in, token is sent in Authorization header
- **User Context**: Backend can use `user_id` from token to personalize responses or track usage

## WebSocket Support (Optional)

The frontend is structured to support WebSocket streaming in the future. Current implementation uses HTTP POST, but can be extended to:

- Connect to `WS /v1/ws/chat/{user_id}` for streaming responses
- Display tokens as they arrive (token-by-token streaming)
- Show typing indicator while streaming

## Sample Questions Displayed

The frontend shows these sample questions to help users:
- "What technologies are used in this project?"
- "How does the analysis workflow work?"
- "What machine learning models are used?"
- "How do I upload an image for analysis?"

## Source Citations Display

The UI includes a section to display source citations when RAG is implemented:
- Shows below assistant messages
- Lists source documents that were used to generate the answer
- Styled with emerald theme colors

## Styling & Theme

- Uses emerald color scheme consistent with the app
- User messages: emerald-600 background
- Assistant messages: white/10 background with emerald border
- Responsive breakpoints for mobile/tablet/desktop
- Smooth scrolling and animations

## Next Steps for Backend Implementation

1. **Implement RAG Pipeline** (as described in the architecture plan)
2. **Create `/v1/chat/message` endpoint** with the request/response format above
3. **Set up vector database** for document embeddings
4. **Integrate Gemini Flash** for LLM generation
5. **Add conversation storage** in SQLite
6. **Implement document ingestion** for private data
7. **Add source citation** extraction from RAG pipeline
8. **Optional**: Add WebSocket support for streaming

## Testing the Frontend

1. Navigate to `/chatbot` route
2. Type a message and click Send
3. Currently shows template response (backend not connected)
4. Once backend is ready, replace the TODO section with actual API call

## Notes

- The frontend is fully functional as a template
- All UI states are handled (loading, error, empty, messages)
- Ready for immediate backend integration
- No breaking changes needed when connecting to backend
- Simply replace the setTimeout mock with actual fetch call
